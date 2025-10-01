import copy

import torch.nn as nn

from .loss.matcher import HungarianMatcher
from .loss.criterion import SetCriterion


class DINOLoss(nn.Module):
    def __init__(self):
        """Internal function to build the loss function."""
        # FIXME - make these config
        cls_loss_coef = 2.0
        bbox_loss_coef = 5.0
        giou_loss_coef = 2.0
        use_dn = True
        aux_loss = True
        dec_layers = 6
        two_stage_type = "standard"
        no_interm_box_loss = False
        interm_loss_coef = 1.0
        num_classes = 91
        loss_types = ["labels", "boxes"]
        focal_alpha = 0.25

        matcher = HungarianMatcher(
            cost_class=cls_loss_coef, cost_bbox=bbox_loss_coef, cost_giou=giou_loss_coef
        )
        weight_dict = {
            "loss_ce": cls_loss_coef,
            "loss_bbox": bbox_loss_coef,
            "loss_giou": giou_loss_coef,
        }
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        # for de-noising training
        if use_dn:
            weight_dict["loss_ce_dn"] = cls_loss_coef
            weight_dict["loss_bbox_dn"] = bbox_loss_coef
            weight_dict["loss_giou_dn"] = giou_loss_coef
        clean_weight_dict = copy.deepcopy(weight_dict)

        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in clean_weight_dict.items()}
                )
            weight_dict.update(aux_weight_dict)

        if two_stage_type != "no":
            interm_weight_dict = {}
            _coeff_weight_dict = {
                "loss_ce": 1.0,
                "loss_bbox": 1.0 if not no_interm_box_loss else 0.0,
                "loss_giou": 1.0 if not no_interm_box_loss else 0.0,
            }
            interm_weight_dict.update(
                {
                    f"{k}_interm": v * interm_loss_coef * _coeff_weight_dict[k]
                    for k, v in clean_weight_dict_wo_dn.items()
                }
            )
            weight_dict.update(interm_weight_dict)

        self.weight_dict = copy.deepcopy(weight_dict)

        self.criterion = SetCriterion(
            num_classes, matcher=matcher, losses=loss_types, focal_alpha=focal_alpha
        )

    def forward(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)

        losses = sum(
            loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        )

        return losses
