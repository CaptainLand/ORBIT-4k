from __future__ import annotations

import torch

from orbit4k.inference_v31 import constrained_tick_states_v31
from orbit4k.validator import EMPTY, LN_START, TAP


def _empty_state() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)


def test_moderate_onset_no_longer_forces_single_tap():
    logits = torch.full((4, 4), -6.0)
    logits[:, EMPTY] = 0.0
    logits[0, TAP] = 7.0
    logits[1, TAP] = 6.5
    logits[2, TAP] = -3.0
    logits[3, TAP] = -3.0
    active, age = _empty_state()

    # sigmoid(0.5 / 0.85) ~= 0.64: old V3 hard-forced a single because < 0.72.
    state, diag = constrained_tick_states_v31(
        logits,
        torch.tensor(0.5),
        active,
        age,
        stars=6.0,
        onset_threshold=0.50,
    )

    assert ((state == TAP) | (state == LN_START)).sum().item() == 2
    assert diag["selected_keydowns"] == 2


def test_six_star_can_emit_three_key_chord_when_model_is_certain():
    logits = torch.full((4, 4), -8.0)
    logits[:, EMPTY] = 0.0
    logits[0, TAP] = 9.0
    logits[1, TAP] = 8.5
    logits[2, TAP] = 8.0
    logits[3, TAP] = -5.0
    active, age = _empty_state()

    state, _ = constrained_tick_states_v31(
        logits,
        torch.tensor(6.0),
        active,
        age,
        stars=6.0,
        onset_threshold=0.40,
    )

    assert ((state == TAP) | (state == LN_START)).sum().item() == 3


def test_six_star_can_emit_quad_only_with_extreme_lane_evidence():
    logits = torch.full((4, 4), -10.0)
    logits[:, EMPTY] = 0.0
    logits[:, TAP] = 10.0
    active, age = _empty_state()

    state, _ = constrained_tick_states_v31(
        logits,
        torch.tensor(8.0),
        active,
        age,
        stars=6.0,
        onset_threshold=0.40,
    )

    assert ((state == TAP) | (state == LN_START)).sum().item() == 4


def test_mild_ln_advantage_is_preserved_in_v31():
    logits = torch.full((4, 4), -8.0)
    logits[:, EMPTY] = 0.0
    logits[0, TAP] = 5.0
    logits[0, LN_START] = 5.5
    active, age = _empty_state()

    state, _ = constrained_tick_states_v31(
        logits,
        torch.tensor(6.0),
        active,
        age,
        stars=6.0,
        onset_threshold=0.40,
        ln_start_margin=0.30,
    )

    assert int(state[0]) == LN_START


def test_active_ln_lane_still_cannot_restart_or_tap():
    logits = torch.zeros(4, 4)
    logits[0, TAP] = 20.0
    logits[0, LN_START] = 20.0

    state, _ = constrained_tick_states_v31(
        logits,
        torch.tensor(8.0),
        torch.tensor([1, 0, 0, 0]),
        torch.tensor([1, 0, 0, 0]),
        stars=6.0,
        onset_threshold=0.40,
    )

    assert int(state[0]) == EMPTY
