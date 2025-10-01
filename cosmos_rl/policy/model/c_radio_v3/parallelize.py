from typing import Optional, Callable
import os

import torch
import torch.nn as nn
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh

try:
    from torch.distributed.fsdp import fully_shard
except ImportError:
    print("torch.distributed.fsdp is not available. DeepSeek model will not work.")

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig


def _apply_fsdp(
    model: nn.Module,
    mesh: DeviceMesh,
):
    """Apply FSDP sharding to model layers using data parallel mesh."""
    default_dp_mesh = mesh["dp_shard_cp"]
    if default_dp_mesh is None:
        return

    fully_shard(model, mesh=default_dp_mesh, reshard_after_forward=True)


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
    device_type, device_module = _get_device_info()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    mesh = parallel_dims.build_mesh(device_type)
    _apply_fsdp(model, mesh)

    return None, None
