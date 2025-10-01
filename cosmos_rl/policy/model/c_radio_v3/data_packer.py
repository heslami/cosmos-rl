from typing import List, Any, Tuple, Dict

import torch

from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.policy.config import Config


class CRadioV3DataPacker(DataPacker):
    def setup(self, config: Config, *args, **kwargs):
        pass

    def sft_process_sample(self, sample: Tuple[Any, Any, Any]) -> Dict[str, Any]:
        image, target, image_path = sample
        return {"image": image, "target": target, "image_path": image_path}

    def sft_compute_max_len(self, processed_samples: List[Any]) -> int:
        return 0  # ignored

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        return {
            "input_ids": torch.tensor([x["image"] for x in processed_samples]),
            "label_ids": [x["target"] for x in processed_samples],
            "targets": [x["target"] for x in processed_samples],
        }
