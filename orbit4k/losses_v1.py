from __future__ import annotations

import torch
import torch.nn.functional as F

EMPTY = 0
TAP = 1
LN_START = 2
LN_END = 3


def note_onset_mask(target: torch.Tensor) -> torch.Tensor:
    return (target == TAP) | (target == LN_START)


def note_probability(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=-1)
    return probabilities[..., TAP] + probabilities[..., LN_START]


def orbit4k_v1_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    cell_mask: torch.Tensor,
    micro_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    cell_onset_weight: float = 0.35,
    micro_onset_weight: float = 0.25,
    density_weight: float = 0.10,
    cell_onset_pos_weight: float = 2.0,
    micro_onset_pos_weight: float = 2.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hierarchical V1 loss on [cell, micro, lane, state]."""
    logits = outputs["logits"]
    if logits.ndim != 5 or logits.shape[2:4] != (3, 4):
        raise ValueError("V1 logits must have shape [B,cells,3,4,4]")

    valid_state = micro_mask[:, :, :, None].expand(-1, -1, -1, 4).reshape(-1) > 0.5
    flat_logits = logits.reshape(-1, 4)[valid_state]
    flat_target = target.reshape(-1)[valid_state]
    state = F.cross_entropy(flat_logits, flat_target, weight=class_weights)

    onset_events = note_onset_mask(target)
    true_micro = onset_events.any(dim=-1).float()
    true_cell = true_micro.bool().any(dim=-1).float()

    micro_bce = F.binary_cross_entropy_with_logits(
        outputs["micro_onset_logits"],
        true_micro,
        reduction="none",
        pos_weight=torch.tensor(
            float(micro_onset_pos_weight),
            device=logits.device,
            dtype=logits.dtype,
        ),
    )
    micro_onset = (micro_bce * micro_mask).sum() / micro_mask.sum().clamp_min(1.0)

    cell_bce = F.binary_cross_entropy_with_logits(
        outputs["cell_onset_logits"],
        true_cell,
        reduction="none",
        pos_weight=torch.tensor(
            float(cell_onset_pos_weight),
            device=logits.device,
            dtype=logits.dtype,
        ),
    )
    cell_onset = (cell_bce * cell_mask).sum() / cell_mask.sum().clamp_min(1.0)

    predicted_count = (
        note_probability(logits) * micro_mask[:, :, :, None]
    ).sum(dim=(1, 2, 3))
    true_count = (
        onset_events.float() * micro_mask[:, :, :, None]
    ).sum(dim=(1, 2, 3))
    density = F.smooth_l1_loss(predicted_count / 100.0, true_count / 100.0)

    total = (
        state
        + float(cell_onset_weight) * cell_onset
        + float(micro_onset_weight) * micro_onset
        + float(density_weight) * density
    )

    valid_cells = cell_mask.sum().clamp_min(1.0)
    occupied_cells = (true_cell * cell_mask).sum() / valid_cells
    valid_micro = micro_mask.sum().clamp_min(1.0)
    occupied_micro = (true_micro * micro_mask).sum() / valid_micro

    return total, {
        "loss": float(total.detach()),
        "state_loss": float(state.detach()),
        "cell_onset_loss": float(cell_onset.detach()),
        "micro_onset_loss": float(micro_onset.detach()),
        "density_loss": float(density.detach()),
        "cell_occupancy": float(occupied_cells.detach()),
        "micro_occupancy": float(occupied_micro.detach()),
    }
