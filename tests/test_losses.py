from __future__ import annotations

import torch

from orbit4k.losses import LN_END, LN_START, TAP, note_onset_mask, note_probability, orbit4k_loss


def test_ln_end_is_not_an_onset():
    target = torch.tensor([[[LN_END, 0, 0, 0]]], dtype=torch.long)
    assert not note_onset_mask(target).any()


def test_tap_and_ln_start_are_onsets():
    target = torch.tensor([[[TAP, LN_START, 0, LN_END]]], dtype=torch.long)
    onset = note_onset_mask(target)
    assert onset.tolist() == [[[True, True, False, False]]]


def test_note_probability_excludes_ln_end():
    logits = torch.full((1, 1, 2, 4), -12.0)
    logits[0, 0, 0, LN_END] = 12.0
    logits[0, 0, 1, TAP] = 12.0

    probability = note_probability(logits)

    assert probability[0, 0, 0] < 1e-6
    assert probability[0, 0, 1] > 1.0 - 1e-6


def test_full_loss_treats_ln_end_as_release_for_onset_and_density():
    target = torch.tensor([[[LN_END, 0, 0, 0]]], dtype=torch.long)
    mask = torch.ones((1, 1), dtype=torch.float32)
    logits = torch.full((1, 1, 4, 4), -8.0)
    logits[..., 0] = 8.0
    logits[0, 0, 0, LN_END] = 9.0
    outputs = {
        "logits": logits,
        # A release-only tick should supervise the auxiliary onset head toward 0.
        "onset_logits": torch.tensor([[-8.0]]),
    }
    class_weights = torch.ones(4)

    _, metrics = orbit4k_loss(
        outputs,
        target,
        mask,
        class_weights=class_weights,
        onset_weight=1.0,
        density_weight=1.0,
    )

    assert metrics["onset_loss"] < 1e-3
    # The prediction places essentially no probability on TAP/LN_START, so a
    # release-only target should also have nearly zero density error.
    assert metrics["density_loss"] < 1e-6
