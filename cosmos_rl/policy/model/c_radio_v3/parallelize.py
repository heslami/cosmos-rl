from typing import Optional, Callable
import os

import torch
import torch.nn as nn
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh

try:
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
except ImportError:
    print("torch.distributed.fsdp is not available. DeepSeek model will not work.")

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig


def _apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    layer_plan = {}
    # FIXME - this hard-codes the config for the "g" model
    for blk_idx in [[0, 9], [10, 19], [20, 29], [30, 39]]:
        start, end = blk_idx
        layer_plan[f"blocks.{start}"] = PrepareModuleInput(
            input_layouts=(Replicate(),), desired_input_layouts=(Shard(1),)
        )
        layer_plan[f"blocks.{end}"] = PrepareModuleOutput(
            output_layouts=(Shard(1)), desired_output_layouts=(Replicate(),)
        )
    parallelize_module(model.model.backbone[0].body.model, tp_mesh, layer_plan)

    for id, block in enumerate(model.model.backbone[0].body.model.blocks):
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


def _apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
):
    """Apply FSDP sharding to model layers using data parallel mesh."""
    fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)


def _get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module: torch.cuda
    return device_type, device_module


def parallelize_model(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable],
) -> nn.Module:
    if parallel_dims.world_size > 1:
        device_type, device_module = _get_device_info()

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = torch.device(f"{device_type}:{local_rank}")
        device_module.set_device(device)
        mesh = parallel_dims.build_mesh(device_type)
        if parallel_dims.tp_enabled:
            _apply_tp(model, mesh["tp"])
        if parallel_dims.dp_enabled:
            _apply_fsdp(model, mesh["dp_shard"])

        # if local_rank == 0:
        #     import pdb; pdb.set_trace()

    return None, None
