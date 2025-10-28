from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import os

from transformers import AutoConfig
import torch
import torch.nn as nn

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.utils.parallelism import ParallelDims
import cosmos_rl.utils.util as util

from .data_packer import CRadioV3DataPacker
from .weight_mapper import CRadioV3WeightMapper, convert_weight_from_hf

from .model.position_encoding import (
    PositionEmbeddingSineHWExport,
    PositionEmbeddingSineHW,
)
from .model.backbone import Joiner, Backbone
from .model.deformable_transformer import DeformableTransformer
from .model.dino import DINO
# from .model.model_utils import load_pretrained_weights


@dataclass
class CRadioV3Args:
    hf_config: AutoConfig
    num_classes: int = 4
    hidden_dim: int = 256
    # pretrained_backbone_path=None,
    # backbone='resnet_50',
    train_backbone: bool = True
    num_feature_levels: int = 2
    nheads: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 1024
    dec_n_points: int = 4
    enc_n_points: int = 4
    num_queries: int = 300
    aux_loss: bool = True
    dilation: bool = False
    dropout_ratio: float = 0.3
    export: bool = False
    activation_checkpoint: bool = False
    return_interm_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    pre_norm: bool = False
    num_patterns: int = 0
    decoder_layer_noise: bool = False
    dln_xy_noise: float = 0.2
    dln_hw_noise: float = 0.2
    add_channel_attention: bool = False
    random_refpoints_xy: bool = False
    two_stage_type: str = "standard"
    two_stage_pat_embed: int = 0
    two_stage_add_query_num: int = 0
    two_stage_learn_wh: bool = False
    two_stage_keep_all_tokens: bool = False
    decoder_sa_type: str = "sa"
    embed_init_tgt: bool = True
    use_detached_boxes_dec_out: bool = False
    fix_refpoints_hw: int = -1
    dec_pred_class_embed_share: bool = False
    dec_pred_bbox_embed_share: bool = False
    two_stage_bbox_embed_share: bool = False
    two_stage_class_embed_share: bool = False
    use_dn: bool = True
    dn_number: int = 100
    dn_box_noise_scale: float = 1.0
    dn_label_noise_ratio: float = 0.5
    pe_temperatureH: int = 20
    pe_temperatureW: int = 20
    lsj_resolution: Optional[int] = None


@ModelRegistry.register(
    CRadioV3WeightMapper, default_data_packer_cls=CRadioV3DataPacker
)
class CRadioV3Model(BaseModel):
    @staticmethod
    def supported_model_types():
        return ["cradio"]

    def __init__(self, args: CRadioV3Args, model_type: str):
        super().__init__(args.hf_config)
        # build positional encoding. only support PositionEmbeddingSine
        if args.export:
            position_embedding = PositionEmbeddingSineHWExport(
                args.hidden_dim // 2,
                temperatureH=args.pe_temperatureH,
                temperatureW=args.pe_temperatureW,
                normalize=True,
            )
        else:
            position_embedding = PositionEmbeddingSineHW(
                args.hidden_dim // 2,
                temperatureH=args.pe_temperatureH,
                temperatureW=args.pe_temperatureW,
                normalize=True,
            )

        # build backbone
        if args.num_feature_levels != len(args.return_interm_indices):
            raise ValueError(
                f"num_feature_levels: {args.num_feature_levels} does not match the size of "
                f"return_interm_indices: {args.return_interm_indices}"
            )

        # Index 4 is not part of the backbone but taken from index 3 with conv 3x3 stride 2
        return_interm_indices = [r for r in args.return_interm_indices if r != 4]
        name = {
            "B": "vit_base_cradiov3",
            "L": "vit_large_cradiov3",
            "H": "vit_huge_cradiov3",
            "g": "vit_giant_cradiov3",
        }[model_type]
        # assert not args.train_backbone
        # with torch.no_grad():
        backbone_only = Backbone(
            name,  # backbone,
            # "/lustre/fs11/portfolios/sw/projects/sw_aidot/users/heslami/.cache/C-RADIOv3-B/c-radio_v3-b_half.pth.tar",  # pretrained_backbone_path,
            args.train_backbone,
            args.lsj_resolution,
            return_interm_indices,
            args.dilation,
            args.export,
            args.activation_checkpoint,
        )

        # Keep joiner for backward compatibility
        joined_backbone = Joiner(backbone_only)

        decoder_query_perturber = None
        if args.decoder_layer_noise:
            from .model.model_utils import RandomBoxPerturber

            decoder_query_perturber = RandomBoxPerturber(
                x_noise_scale=args.dln_xy_noise,
                y_noise_scale=args.dln_xy_noise,
                w_noise_scale=args.dln_hw_noise,
                h_noise_scale=args.dln_hw_noise,
            )

        # build tranformer
        transformer = DeformableTransformer(
            d_model=args.hidden_dim,
            nhead=args.nheads,
            export=args.export,
            activation_checkpoint=args.activation_checkpoint,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout_ratio,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=args.num_feature_levels,
            enc_n_points=args.enc_n_points,
            dec_n_points=args.dec_n_points,
            num_queries=args.num_queries,
            normalize_before=args.pre_norm,
            num_patterns=args.num_patterns,
            modulate_hw_attn=True,
            deformable_decoder=True,
            decoder_query_perturber=decoder_query_perturber,
            add_channel_attention=args.add_channel_attention,
            random_refpoints_xy=args.random_refpoints_xy,
            # two stage
            two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
            two_stage_pat_embed=args.two_stage_pat_embed,
            two_stage_add_query_num=args.two_stage_add_query_num,
            two_stage_learn_wh=args.two_stage_learn_wh,
            two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
            dec_layer_number=None,
            rm_self_attn_layers=None,
            key_aware_type=None,
            layer_share_type=None,
            rm_detach=None,
            decoder_sa_type=args.decoder_sa_type,
            module_seq=["sa", "ca", "ffn"],
            embed_init_tgt=args.embed_init_tgt,
            use_detached_boxes_dec_out=args.use_detached_boxes_dec_out,
        )

        # build deformable detr model
        self.model = DINO(
            joined_backbone,
            position_embedding,
            transformer,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
            export=args.export,
            random_refpoints_xy=args.random_refpoints_xy,
            fix_refpoints_hw=args.fix_refpoints_hw,
            num_feature_levels=args.num_feature_levels,
            nheads=args.nheads,
            dec_pred_class_embed_share=args.dec_pred_class_embed_share,
            dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
            # two stage
            two_stage_type=args.two_stage_type,
            # box_share
            two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
            two_stage_class_embed_share=args.two_stage_class_embed_share,
            decoder_sa_type=args.decoder_sa_type,
            num_patterns=args.num_patterns,
            dn_number=args.dn_number if args.use_dn else 0,
            dn_box_noise_scale=args.dn_box_noise_scale,
            dn_label_noise_ratio=args.dn_label_noise_ratio,
            dn_labelbook_size=args.num_classes,
        )
        self.model_args = args

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "CRadioV3Model":
        return CRadioV3Model(
            CRadioV3Args(
                hf_config=hf_config,
                num_classes=91,
                hidden_dim=256,  # default
                # pretrained_backbone_path=pretrained_backbone,
                # backbone=backbone,
                train_backbone=False,
                num_feature_levels=5,
                nheads=8,  # default
                enc_layers=6,
                dec_layers=6,
                dim_feedforward=2048,
                dec_n_points=4,  # default
                enc_n_points=4,  # default
                num_queries=900,
                aux_loss=True,  # default
                dilation=False,  # default
                dropout_ratio=0.0,
                export=False,
                activation_checkpoint=True,
                return_interm_indices=[0, 1, 2, 3, 4],
                decoder_sa_type="sa",  # default
                embed_init_tgt=True,  # default
                use_dn=True,  # default
                dn_number=100,  # default
                dn_box_noise_scale=1.0,  # default
                dn_label_noise_ratio=0.5,  # default
                pe_temperatureH=20,  # default
                pe_temperatureW=20,  # default
                lsj_resolution=1024,
                pre_norm=False,  # default
                two_stage_type="standard",  # default
                fix_refpoints_hw=-1,  # default
            ),
            model_name_or_path[-1],
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        targets: Optional[Dict[Any, Any]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """model forward function"""
        return self.model(
            input_ids, targets=targets if self.model_args.use_dn else None
        )

    @property
    def parallelize_fn(self):
        from .parallelize import parallelize_model

        return parallelize_model, self

    def post_to_empty_hook(self, config: CosmosConfig):
        nn.MultiheadAttention.reset_parameters = nn.MultiheadAttention._reset_parameters
        self.model.apply(
            lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
        )
        # initialize the ViT adapter again since reset_parameter call above recursively overwrites some of submodules
        self.model.backbone[0].body.reset_parameters()

    def separate_model_parts(self) -> List[nn.Module]:
        return [self]
        # FIXME - split the model to account for different learning rate used for the backbone vs the the rest
        # return [self.model.position_embedding, self.model.transformer, self.model.backbone]

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        model_path = util.resolve_model_path(model_name_or_path, revision=revision)
        safetensors_files = [
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        ]

        # FIXME: check if we need to replace with checkpoint wrapper names

        backbone = self.model.backbone[0].body
        backbone_state_dict = backbone.state_dict()
        used_checkpoint_names = set()
        for f in safetensors_files:
            ckpt = util.safe_open(
                os.path.join(model_path, f), framework="pt", device=str(device)
            )
            for name in ckpt.keys():
                ckpt_tensor = ckpt.get_tensor(name)
                dest_name, sharded_tensor = convert_weight_from_hf(
                    ckpt_tensor, name, parallel_dims
                )
                if dest_name not in backbone_state_dict:
                    logger.info(
                        f"Weight '{dest_name}' is discarded from the HF weights"
                    )
                    continue
                target_tensor = backbone_state_dict[dest_name]
                local_view = (
                    target_tensor.to_local()
                    if isinstance(target_tensor, torch.distributed.tensor.DTensor)
                    else target_tensor
                )
                assert (
                    local_view.shape == sharded_tensor.shape
                ), f"Shape mismatch: {local_view.shape} != {sharded_tensor.shape} for {dest_name}"
                with torch.no_grad():
                    local_view.copy_(sharded_tensor)
                used_checkpoint_names.add(dest_name)

        for name, parameter in backbone.named_parameters():
            if name in used_checkpoint_names and not self.model_args.train_backbone:
                parameter.requires_grad_(False)

        # pretrained_backbone_ckp = (
        #     load_pretrained_weights(pretrained_backbone_path)
        #     if pretrained_backbone_path
        #     else None
        # )

        # if pretrained_backbone_ckp:
        #     pretrained_backbone_ckp = {
        #         k.replace("base_model.", "model."): v
        #         for k, v in pretrained_backbone_ckp.items()
        #     }

        # missing_keys = None
        # if pretrained_backbone_ckp:
        #     _tmp_st_output = backbone.load_state_dict(
        #         pretrained_backbone_ckp, strict=False
        #     )
        #     missing_keys = list(_tmp_st_output[0])
        #     if get_global_rank() == 0:
        #         logger.info(
        #             f"Loaded pretrained weights from {pretrained_backbone_path}"
        #         )
        #         logger.info(f"{_tmp_st_output}")

        # if not missing_keys:
        #     missing_keys = []
        # for name, parameter in backbone.named_parameters():
        #     if not any(p in name for p in missing_keys) and not train_backbone:
        #         parameter.requires_grad_(False)

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        inputs = kwargs["input_ids"]
        return torch.empty(1), inputs, 1

    def apply_pipeline_split(self, pp_rank, pp_size):
        pass

    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        pass
