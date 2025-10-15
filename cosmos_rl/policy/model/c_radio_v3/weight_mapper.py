from typing import Any, Tuple, Dict, List

import torch
from transformers import AutoConfig

from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.logging import logger


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

    logger.info(
        f"dp_shard_rank = {dp_shard_rank}, parallel_dims.mesh = {parallel_dims.mesh}"
    )

    dest_name = map_key_from_hf(name)

    if tensor.shape[0] % dp_shard_size == 0:
        shard = tensor.tensor_split(dp_shard_size, dim=0)[dp_shard_rank]
    else:
        chunk_size = (tensor.shape[0] + dp_shard_size - 1) // dp_shard_size
        shard = tensor[dp_shard_rank * chunk_size : (dp_shard_rank + 1) * chunk_size]

    logger.info(f"tensor shape = {tensor.shape}, final shard shape = {shard.shape}")

    return dest_name, shard.contiguous()


class CRadioV3WeightMapper(WeightMapper):
    def __init__(self, hf_config: AutoConfig):
        pass

    def rollout_prepare_recv(
        self,
        vllm_model: Any,
    ) -> Tuple[Dict[str, torch.Tensor], List[List[Tuple[str, int]]]]:
        pass

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        pass

    def policy_maybe_decompose_weights_to_hf_naming(self, name, param):
        raise NotImplementedError("checkpoint to safetensors not supported yet")
