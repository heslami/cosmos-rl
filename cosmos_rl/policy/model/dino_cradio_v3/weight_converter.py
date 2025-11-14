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
    dest_name = map_key_from_hf(name)

    return dest_name, tensor
