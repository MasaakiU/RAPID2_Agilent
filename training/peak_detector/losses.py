"""The losses of the detector (SPEC.md 5): of the heat, CenterNet's focal loss; of the offset and the log widths, L1
at the apexes; of the denoised channels, L1 where the sample is."""

import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = {"heat": 1.0, "offset": 1.0, "widths": 0.1, "denoise": 1.0}


def focal_loss(logits, heat, weight, alpha: float = 2.0, beta: float = 4.0):
    """Over the scored cells (weight), by the number of apexes: -(1-p)^a log p at them, -(1-y)^b p^a log(1-p) off."""
    log_p, log_q = F.logsigmoid(logits), F.logsigmoid(-logits)
    p = log_p.exp()
    positive = (heat >= 1.0).float()
    loss = -log_p * (1 - p) ** alpha * positive - log_q * p ** alpha * (1 - heat) ** beta * (1 - positive)
    return (loss * weight).sum() / (positive * weight).sum().clamp(min=1.0)


def masked_l1(prediction, target, mask):
    """The mean absolute difference where mask is 1; mask of prediction's shape less a leading channel axis, if any."""
    if prediction.dim() == mask.dim() + 1:
        mask = mask.unsqueeze(1).expand_as(prediction)
    return ((prediction - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)


def losses(outputs: dict, batch: dict, weights: dict = DEFAULT_WEIGHTS) -> dict:
    """Each loss, and their weighted sum as "total"."""
    values = {
        "heat": focal_loss(outputs["heat"].float(), batch["heat"], batch["weight"]),
        "offset": masked_l1(outputs["offset"].float(), batch["offset"], batch["reg_mask"]),
        "widths": masked_l1(outputs["log_widths"].float(), batch["log_widths"], batch["reg_mask"]),
    }
    if "denoised" in outputs:
        values["denoise"] = masked_l1(outputs["denoised"].float(), batch["clean"], batch["valid"])
    values["total"] = sum(weights.get(name, 0.0) * value for name, value in values.items())
    return values
