"""DINO + C-RADIOv3 top-level module."""

from typing import Optional, List, Tuple, Dict, Any, Callable
import os

from transformers import AutoConfig
import torch
import torch.nn as nn

from cosmos_rl.utils.logging import logger
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.utils.parallelism import ParallelDims
import cosmos_rl.utils.util as util

from .utils import get_experiment_config
from .data_packer import DINOCRadioV3DataPacker
from .weight_mapper import DINOCRadioV3WeightMapper
from .weight_converter import convert_weight_from_hf

from .model.build_nn_model import DINOModel
from .model.loss.loss_fn import DINOLoss


@ModelRegistry.register(
    DINOCRadioV3WeightMapper, default_data_packer_cls=DINOCRadioV3DataPacker
)
class DINOCRadioV3Model(BaseModel):
    @staticmethod
    def supported_model_types():
        return ["cradio"]

    def __init__(
        self,
        hf_config: AutoConfig,
        experiment_config: Dict[str, Any],
        backbone_size_label: str,
    ):
        super().__init__(hf_config)
        model_config = experiment_config["model"]
        dataset_config = experiment_config["dataset"]
        num_classes = dataset_config["num_classes"]
        backbone = {
            "B": "vit_base_cradiov3",
            "L": "vit_large_cradiov3",
            "H": "vit_huge_cradiov3",
            "g": "vit_giant_cradiov3",
        }[backbone_size_label]
        dropout_ratio = model_config["dropout_ratio"]
        hidden_dim = model_config["hidden_dim"]
        num_feature_levels = model_config["num_feature_levels"]
        nheads = model_config["nheads"]
        enc_layers = model_config["enc_layers"]
        dec_layers = model_config["dec_layers"]
        dim_feedforward = model_config["dim_feedforward"]
        dec_n_points = model_config["dec_n_points"]
        enc_n_points = model_config["enc_n_points"]
        num_queries = model_config["num_queries"]
        aux_loss = model_config["aux_loss"]
        dilation = model_config["dilation"]
        self.train_backbone = model_config["train_backbone"]

        # DINO specific
        return_interm_indices = model_config["return_interm_indices"]
        pre_norm = model_config["pre_norm"]
        two_stage_type = model_config["two_stage_type"]
        decoder_sa_type = model_config["decoder_sa_type"]
        embed_init_tgt = model_config["embed_init_tgt"]
        fix_refpoints_hw = model_config["fix_refpoints_hw"]
        use_dn = model_config["use_dn"]
        dn_number = model_config["dn_number"]
        dn_box_noise_scale = model_config["dn_box_noise_scale"]
        dn_label_noise_ratio = model_config["dn_label_noise_ratio"]
        pe_temperatureH = model_config["pe_temperatureH"]
        pe_temperatureW = model_config["pe_temperatureW"]

        activation_checkpoint = experiment_config["train"]["activation_checkpoint"]
        lsj_resolution = experiment_config["dataset"]["augmentation"][
            "fixed_random_crop"
        ]

        self.model = DINOModel(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            backbone=backbone,
            num_feature_levels=num_feature_levels,
            nheads=nheads,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            num_queries=num_queries,
            aux_loss=aux_loss,
            dilation=dilation,
            dropout_ratio=dropout_ratio,
            export=experiment_config["export"],
            activation_checkpoint=activation_checkpoint,
            return_interm_indices=return_interm_indices,
            decoder_sa_type=decoder_sa_type,
            embed_init_tgt=embed_init_tgt,
            use_dn=use_dn,
            dn_number=dn_number,
            dn_box_noise_scale=dn_box_noise_scale,
            dn_label_noise_ratio=dn_label_noise_ratio,
            pe_temperatureH=pe_temperatureH,
            pe_temperatureW=pe_temperatureW,
            lsj_resolution=lsj_resolution,
            pre_norm=pre_norm,
            two_stage_type=two_stage_type,
            fix_refpoints_hw=fix_refpoints_hw,
        )

    @classmethod
    def get_loss_fn(cls) -> Callable:
        experiment_config = get_experiment_config()
        return DINOLoss(experiment_config["model"], experiment_config["dataset"])

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "DINOCRadioV3Model":
        return DINOCRadioV3Model(
            hf_config,
            get_experiment_config(),
            model_name_or_path[-1],
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        targets: Optional[Dict[Any, Any]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        return self.model(input_ids, targets=targets)

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
        self.model.model.backbone[0].body.reset_parameters()

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

        backbone = self.model.model.backbone[0].body
        backbone_state_dict = backbone.state_dict()
        used_checkpoint_names = set()
        for f in safetensors_files:
            ckpt = util.safe_open(
                os.path.join(model_path, f), framework="pt", device=str(device)
            )
            for name in ckpt.keys():
                ckpt_tensor = ckpt.get_tensor(name)
                dest_name, tensor = convert_weight_from_hf(
                    ckpt_tensor, name, parallel_dims
                )
                if dest_name not in backbone_state_dict:
                    logger.info(
                        f"Weight '{dest_name}' is discarded from the HF weights"
                    )
                    continue
                target_tensor = backbone_state_dict[dest_name]
                assert (
                    target_tensor.shape == tensor.shape
                ), f"Shape mismatch: {target_tensor.shape} != {tensor.shape} for {dest_name}"
                with torch.no_grad():
                    target_tensor.copy_(tensor)
                used_checkpoint_names.add(dest_name)

        for name, parameter in backbone.named_parameters():
            if name in used_checkpoint_names and not self.train_backbone:
                parameter.requires_grad_(False)

    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        inputs = kwargs["input_ids"]
        return torch.empty(1), inputs, 1

    def separate_model_parts(self) -> List[nn.Module]:
        return [self]

    def apply_pipeline_split(self, pp_rank, pp_size):
        pass

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        pass
