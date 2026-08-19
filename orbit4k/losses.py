from __future__ import annotations

import torch
import torch.nn.functional as F


def orbit4k_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    onset_weight: float = 0.35,
    density_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    valid = mask[:, :, None].expand(-1, -1, 4).reshape(-1) > 0.5
    flat_logits = logits.reshape(-1, 4)[valid]
    flat_target = target.reshape(-1)[valid]
    state = F.cross_entropy(flat_logits, flat_target, weight=class_weights)

    true_onset = (target != 0).any(dim=-1).float()
    onset = F.binary_cross_entropy_with_logits(outputs["onset_logits"], true_onset, reduction="none")
    onset = (onset * mask).sum() / mask.sum().clamp_min(1.0)

    nonempty_probability = 1.0 - logits.softmax(dim=-1)[..., 0]
    predicted_count = (nonempty_probability * mask[:, :, None]).sum(dim=(1, 2))
    true_count = ((target != 0).float() * mask[:, :, None]).sum(dim=(1, 2))
    density = F.smooth_l1_loss(predicted_count / 100.0, true_count / 100.0)

    total = state + onset_weight * onset + density_weight * density
    return total, {
        "loss": float(total.detach()),
        "state_loss": float(state.detach()),
        "onset_loss": float(onset.detach()),
        "density_loss": float(density.detach()),
    }
