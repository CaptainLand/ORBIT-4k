from __future__ import annotations

import torch
import torch.nn.functional as F

EMPTY = 0
TAP = 1
LN_START = 2
LN_END = 3


def note_onset_mask(target: torch.Tensor) -> torch.Tensor:
    """Return True for key-down events only: TAP and LN_START.

    LN_END is a release event and must not supervise the onset head or note-density target.
    """

    return (target == TAP) | (target == LN_START)


def note_probability(logits: torch.Tensor) -> torch.Tensor:
    """Expected key-down probability for each lane/tick."""

    probabilities = logits.softmax(dim=-1)
    return probabilities[..., TAP] + probabilities[..., LN_START]


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

    onset_events = note_onset_mask(target)
    true_onset = onset_events.any(dim=-1).float()
    onset = F.binary_cross_entropy_with_logits(
        outputs["onset_logits"], true_onset, reduction="none"
    )
    onset = (onset * mask).sum() / mask.sum().clamp_min(1.0)

    # Density is defined as expected key-down count, not generic non-empty state
    # count. TAP and LN_START contribute; LN_END is a release and contributes 0.
    predicted_count = (note_probability(logits) * mask[:, :, None]).sum(dim=(1, 2))
    true_count = (onset_events.float() * mask[:, :, None]).sum(dim=(1, 2))
    density = F.smooth_l1_loss(predicted_count / 100.0, true_count / 100.0)

    total = state + onset_weight * onset + density_weight * density
    return total, {
        "loss": float(total.detach()),
        "state_loss": float(state.detach()),
        "onset_loss": float(onset.detach()),
        "density_loss": float(density.detach()),
    }
