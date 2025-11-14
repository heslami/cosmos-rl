from typing import Tuple

import torch

from cosmos_rl.utils.parallelism import ParallelDims


def map_key_from_hf(name: str):
    return name[len("radio_model.") :].replace(".grandma", ".gamma")


def convert_weight_from_hf(
    tensor: torch.Tensor,
    name: str,
    parallel_dims: ParallelDims,
) -> Tuple[str, torch.Tensor]:
    if parallel_dims.dp_shard_enabled:
        dp_shard_rank = parallel_dims.mesh["dp_shard"].get_local_rank()
        dp_shard_size = parallel_dims.mesh["dp_shard"].size()
    else:
        dp_shard_rank = 0
        dp_shard_size = 1

    dest_name = map_key_from_hf(name)

    if tensor.shape[0] % dp_shard_size == 0:
        shard = tensor.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    else:
        chunk_size = (tensor.shape[0] + dp_shard_size - 1) // dp_shard_size
        shard = tensor[dp_shard_rank * chunk_size : (dp_shard_rank + 1) * chunk_size]

    return dest_name, shard.contiguous()
