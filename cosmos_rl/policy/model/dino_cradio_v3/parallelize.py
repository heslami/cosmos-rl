from typing import Optional, Callable

import torch.nn as nn
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig


def parallelize_model(
    model: nn.Module,
    parallel_dims: ParallelDims,
    config: CosmosConfig,
    pp_loss_fn: Optional[Callable],
) -> nn.Module:
    if parallel_dims.world_size > 1:
        raise NotImplementedError("Only single GPU runs supported!")

    return None, None
