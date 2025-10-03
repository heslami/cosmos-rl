from typing import Any, Tuple, Dict, List

import torch
from transformers import AutoConfig

from cosmos_rl.policy.model.base import WeightMapper


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
