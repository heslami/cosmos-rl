from typing import Optional, Callable

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig


def parallelize_model(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable],
) -> nn.Module:
    if parallel_dims.world_size == 1:
        return None, None

    world_mesh = parallel_dims.mesh

    if parallel_dims.tp_enabled:
        assert config.policy.model_name_or_path.endswith(
            "g"
        ), "TP only for g model is supported"
        _apply_tp(model, world_mesh["tp"])

    if parallel_dims.dp_shard_enabled:
        _apply_fsdp(model, world_mesh["dp_shard"])
    elif parallel_dims.dp_replicate_enabled:
        assert world_mesh.ndim == 1, "DDP does not support > 1D parallelism"
        _apply_ddp(model, world_mesh)

    return None, None


def _apply_ddp(model: nn.Module, dp_mesh: DeviceMesh):
    replicate(model, device_mesh=dp_mesh)
    logger.info("Applied DDP to the model")


def _apply_fsdp(model: nn.Module, dp_mesh: DeviceMesh):
    fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)


def _apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    layer_plan = {}
    # Assume only "g" model will use TP
    for blk_idx in [[0, 9], [10, 19], [20, 29], [30, 39]]:
        start, end = blk_idx
        layer_plan[f"blocks.{start}"] = PrepareModuleInput(
            input_layouts=(Replicate(),), desired_input_layouts=(Shard(1),)
        )
        layer_plan[f"blocks.{end}"] = PrepareModuleOutput(
            output_layouts=(Shard(1)), desired_output_layouts=(Replicate(),)
        )
    parallelize_module(model.model.model.backbone[0].body.model, tp_mesh, layer_plan)

    for block in model.model.model.backbone[0].body.model.blocks:
        layer_plan = {
            "norm1": SequenceParallel(),
            "attn": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attn.q": ColwiseParallel(),
            "attn.k": ColwiseParallel(),
            "attn.v": ColwiseParallel(),
            "attn.proj": RowwiseParallel(output_layouts=Shard(1)),
            "ls1": SequenceParallel(),
            "norm2": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)
            ),
            "mlp.fc1": ColwiseParallel(),
            "mlp.fc2": RowwiseParallel(output_layouts=Shard(1)),
            "ls2": SequenceParallel(),
        }

        parallelize_module(block, tp_mesh, layer_plan)
