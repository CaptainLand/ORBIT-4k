from __future__ import annotations

import torch

from orbit4k.inference import chart_statistics, constrained_tick_states
from orbit4k.validator import EMPTY, LN_END, LN_START, TAP


def test_onset_gate_blocks_spurious_keydowns():
    # Per-lane head strongly wants LN_START everywhere, but the onset head says
    # there is no musical key-down at this tick. V2 must output EMPTY.
    logits = torch.zeros(4, 4)
    logits[:, LN_START] = 8.0
    state, diag = constrained_tick_states(
        logits,
        torch.tensor(-8.0),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        stars=6.0,
    )
    assert torch.equal(state, torch.full((4,), EMPTY, dtype=torch.long))
    assert not diag["onset_gate"]


def test_six_star_decoder_caps_chord_at_two_lanes():
    logits = torch.zeros(4, 4)
    logits[:, TAP] = 8.0
    state, _ = constrained_tick_states(
        logits,
        torch.tensor(8.0),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        stars=6.0,
    )
    keydowns = ((state == TAP) | (state == LN_START)).sum().item()
    assert keydowns == 2


def test_active_ln_lane_cannot_restart_or_tap():
    logits = torch.zeros(4, 4)
    logits[0, TAP] = 20.0
    logits[0, LN_START] = 20.0
    logits[0, LN_END] = -20.0
    state, _ = constrained_tick_states(
        logits,
        torch.tensor(-8.0),
        torch.tensor([1, 0, 0, 0]),
        torch.tensor([8, 0, 0, 0]),
        stars=6.0,
    )
    assert int(state[0]) == EMPTY


def test_ln_start_requires_margin_over_tap():
    logits = torch.zeros(4, 4)
    logits[0, TAP] = 5.0
    logits[0, LN_START] = 5.5
    logits[1:, TAP] = -10.0
    logits[1:, LN_START] = -10.0
    state, _ = constrained_tick_states(
        logits,
        torch.tensor(8.0),
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        stars=5.0,
        ln_start_margin=1.25,
    )
    assert int(state[0]) == TAP


def test_final_tick_closes_active_ln_and_starts_nothing_new():
    logits = torch.zeros(4, 4)
    logits[:, TAP] = 10.0
    state, _ = constrained_tick_states(
        logits,
        torch.tensor(10.0),
        torch.tensor([1, 0, 1, 0]),
        torch.tensor([10, 0, 10, 0]),
        stars=9.0,
        final_tick=True,
    )
    assert state.tolist() == [LN_END, EMPTY, LN_END, EMPTY]


def test_statistics_expose_ln_and_chord_collapse_metrics():
    chart = torch.tensor(
        [
            [TAP, EMPTY, EMPTY, EMPTY],
            [LN_START, TAP, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY, EMPTY],
            [LN_END, EMPTY, EMPTY, EMPTY],
        ],
        dtype=torch.uint8,
    ).numpy()
    stats = chart_statistics(chart)
    assert stats["keydowns"] == 3
    assert stats["ln"] == 1
    assert stats["onset_ticks"] == 2
    assert stats["chord_ticks"] == 1
    assert abs(stats["ln_ratio"] - 1 / 3) < 1e-6
    assert abs(stats["chord_ratio"] - 0.5) < 1e-6
