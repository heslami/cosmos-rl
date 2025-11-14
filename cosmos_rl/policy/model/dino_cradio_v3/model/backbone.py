"""Backbone modules."""

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from vision_transformer.vit_adapter import vit_model_dict


class BackboneBase(nn.Module):
    """BackboneBase class."""

    def __init__(
        self,
        model_name,
        backbone: nn.Module,
        num_channels: int,
        return_interm_indices: list,
        export: bool,
    ):
        """Initialize the Backbone Base Class.

        Args:
            model_name (str): backbone model name.
            backbone (nn.Module): backbone torch module.
            num_channels (int): channel size.
            return_interm_indices (list): list of layer indices to reutrn as backbone features.
            export (bool): flag to indicate whehter exporting to onnx or not.
        """
        super().__init__()
        self.export = export
        self.model_name = model_name

        self.body = backbone

        self.num_channels = num_channels
        self.return_interm_indices = return_interm_indices

    def forward(self, input_tensors):
        """Forward function for Backboone base.

        Args:
            input_tensors (torch.Tensor): input tensor.

        Returns:
            out (torch.Tensor): output tensor.
        """
        if self.export:
            batch_shape = input_tensors.shape
            dtype = input_tensors.dtype
            device = input_tensors.device
            # when exporting, the input shape is fixed and no padding mask is needed.
            masks = torch.zeros(
                (batch_shape[0], 1, batch_shape[2], batch_shape[3]),
                dtype=dtype,
                device=device,
            )
            input_tensor = input_tensors
        else:
            masks = input_tensors[:, 3:4]
            input_tensor = input_tensors[:, :3]

        xs = self.body(input_tensor)

        # Handling timm/efficientvit cases
        if isinstance(xs, list):
            new_xs = {}
            for i, x in enumerate(xs):
                new_xs[f"layer{i}"] = x
            xs = new_xs

        out: Dict[str, torch.Tensor] = {}
        for name, x in xs.items():
            mask = F.interpolate(masks.float(), size=x.shape[-2:])
            mask = mask.to(torch.bool)
            out[name] = (x, mask)
        return out


class Backbone(BackboneBase):
    """Backbone for DINO."""

    def __init__(
        self,
        name: str,
        resolution: int,
        return_interm_indices: list,
        dilation: bool,
        export: bool,
        activation_checkpoint: bool,
    ):
        """Initialize the Backbone Class.

        Args:
            resolution (int): input resolution for ViT models.
            return_interm_indices (list): list of layer indices to reutrn as backbone features.
            dilation (bool): flag whether we can to use dilation or not.
            export (bool): flag to indicate whehter exporting to onnx or not.
            activation_checkpoint (bool): flag to indicate whether to run activation checkpointing during training.

        Raises:
            ValueError: If return_interm_indices does not have valid range or has duplicate index.
            NotImplementedError: If invalid backbone name was provided.
        """
        return_interm_indices = np.array(return_interm_indices)
        if not np.logical_and(
            return_interm_indices >= 0, return_interm_indices <= 4
        ).all():
            raise ValueError(
                f"Invalid range for return_interm_indices. "
                f"Provided return_interm_indices is {return_interm_indices}."
            )
        if len(np.unique(return_interm_indices)) != len(return_interm_indices):
            raise ValueError(
                f"Duplicate index in the provided return_interm_indices: {return_interm_indices}"
            )

        if name not in vit_model_dict:
            raise NotImplementedError(
                f"{name} is not supported ViT-Adapter backbone. "
                f"Supported architecutres: {vit_model_dict.keys()}"
            )

        backbone = vit_model_dict[name](
            out_indices=return_interm_indices,
            resolution=resolution,
            activation_checkpoint=activation_checkpoint,
        )
        num_channels = np.array([backbone.embed_dim] * len(return_interm_indices))

        super().__init__(
            name,
            backbone,
            num_channels,
            return_interm_indices,
            export,
        )


class Joiner(nn.Sequential):
    """Joiner Class."""

    def __init__(self, backbone):
        """Initialize the Joiner Class.

        Args:
            backbone (nn.Module): backbone module.

        """
        super().__init__(backbone)
        self.num_channels = backbone.num_channels

    def forward(self, input_tensors):
        """Forward function for Joiner to prepare the backbone output into transformer input format.

        Args:
            input_tensors (torch.Tensor): input tensor.

        Returns:
            out (List[Tensor]): list of tensor (feature vectors from backbone).

        """
        xs = self[0](input_tensors)
        out: List[torch.Tensor] = []
        for _, x in sorted(xs.items()):
            out.append(x)
        return out
