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


# def dino_collate_fn(batch):
#     """Custom collate function for DETR-like models.

#     DETR models use mutli-scale resize and random cropping which results in the varying input resolution of a single image.
#     Hence, we need a custom collate_fn to pad additional regions and pass that as mask to transformer.

#     Args:
#         batch (tuple): tuple of a single batch. Contains image and label tensors

#     Returns:
#         batch (tuple): tuple of a single batch with uniform image resolution after padding.
#     """
#     batch = list(zip(*batch))
#     batch[0] = tensor_from_tensor_list(batch[0], batch[1])
#     return tuple(batch)


def _max_by_axis(the_list):
    """Get maximum image shape for padding."""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def tensor_from_tensor_list(tensor_list, targets):
    """Convert list of tensors with different size to fixed resolution.

    The final size is determined by largest height and width.
    In theory, the batch could become [3, 1333, 1333] on dataset with different aspect ratio, e.g. COCO
    A fourth channel dimension is the mask region in which 0 represents the actual image and 1 means the padded region.
    This is to give size information to the transformer archicture. If transform-padding is applied,
    then only the pre-padded regions gets mask value of 1.

    Args:
        tensor_list (List[Tensor]): list of image tensors
        targets (List[dict]): list of labels that contain the size information

    Returns:
        tensors (torch.Tensor): list of image tensors in shape of (B, 4, H, W)
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        temp_tensors = torch.zeros((b, c, h, w), dtype=dtype, device=device)
        mask = torch.ones((b, 1, h, w), dtype=dtype, device=device)
        tensors = torch.concat((temp_tensors, mask), 1)
        for img, target, pad_img in zip(tensor_list, targets, tensors):
            # Get original image size before transform-padding
            # If no transform-padding has been applied,
            # then height == img.shape[1] and width == img.shape[2]
            actual_height, actual_width = target["size"]
            pad_img[: img.shape[0], :actual_height, :actual_width].copy_(
                img[:, :actual_height, :actual_width]
            )
            pad_img[c, :actual_height, :actual_width] = (
                0  # set zeros for mask in non-padded area
            )
    else:
        raise ValueError("Channel size other than 3 is not supported")
    return tensors
