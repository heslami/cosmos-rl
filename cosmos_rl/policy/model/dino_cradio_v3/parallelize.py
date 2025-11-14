from typing import Optional, Callable

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.replicate import replicate

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
    # DDP
    if parallel_dims.dp_replicate_enabled:
        assert world_mesh.ndim == 1, "DDP does not support > 1D parallelism"
        _apply_ddp(model, world_mesh)

    return None, None


def _apply_ddp(model: nn.Module, dp_mesh: DeviceMesh):
    replicate(model, device_mesh=dp_mesh)
    logger.info("Applied DDP to the model")
