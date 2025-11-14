from typing import List, Any, Tuple, Dict

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker

from .model.deformable_detr.utils import tensor_from_tensor_list


class DINOCRadioV3DataPacker(BaseDataPacker):
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
        targets = [x["target"] for x in processed_samples]
        return {
            "input_ids": tensor_from_tensor_list(
                [x["image"] for x in processed_samples], targets
            ),
            "label_ids": targets,
            "targets": targets,
        }

    def get_rollout_input(self, item: Any) -> Any:
        pass

    def get_policy_input(
        self,
        sample: Any,
        rollout_output: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        pass

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        pass

    def policy_collate_fn(
        self,
        processed_samples: List[Any],
        computed_max_len: int,
    ) -> Dict[str, Any]:
        pass
