from __future__ import annotations

import torch

from orbit4k.losses_v1 import orbit4k_v1_loss


def test_v1_loss_accepts_cell_and_micro_masks():
    batch, cells = 2, 5
    logits = torch.zeros(batch, cells, 3, 4, 4)
    micro_onset_logits = torch.zeros(batch, cells, 3)
    cell_onset_logits = torch.zeros(batch, cells)
    target = torch.zeros(batch, cells, 3, 4, dtype=torch.long)
    target[0, 1, 0, 2] = 1
    target[0, 2, 2, 1] = 2
    target[1, 0, 0, 0] = 1
    cell_mask = torch.ones(batch, cells)
    micro_mask = torch.ones(batch, cells, 3)
    # Simulate a partial final cell in sample 1.
    micro_mask[1, -1, 1:] = 0

    loss, metrics = orbit4k_v1_loss(
        {
            "logits": logits,
            "micro_onset_logits": micro_onset_logits,
            "cell_onset_logits": cell_onset_logits,
        },
        target,
        cell_mask,
        micro_mask,
        class_weights=torch.tensor([0.2, 1.0, 1.2, 1.2]),
    )

    assert torch.isfinite(loss)
    assert metrics["loss"] > 0
    assert 0 < metrics["cell_occupancy"] < 1
    assert 0 < metrics["micro_occupancy"] < 1


def test_cell_occupancy_is_less_sparse_than_micro_occupancy():
    batch, cells = 1, 4
    target = torch.zeros(batch, cells, 3, 4, dtype=torch.long)
    # One onset in each of two cells, but only one micro slot per occupied cell.
    target[0, 0, 0, 0] = 1
    target[0, 2, 2, 3] = 1
    outputs = {
        "logits": torch.zeros(batch, cells, 3, 4, 4),
        "micro_onset_logits": torch.zeros(batch, cells, 3),
        "cell_onset_logits": torch.zeros(batch, cells),
    }
    _, metrics = orbit4k_v1_loss(
        outputs,
        target,
        torch.ones(batch, cells),
        torch.ones(batch, cells, 3),
        class_weights=torch.ones(4),
    )

    assert metrics["cell_occupancy"] == 0.5
    assert abs(metrics["micro_occupancy"] - (2 / 12)) < 1e-6
    assert metrics["cell_occupancy"] > metrics["micro_occupancy"]
