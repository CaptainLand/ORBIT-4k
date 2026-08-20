from __future__ import annotations

import numpy as np

from orbit4k.inference_full import (
    _state_before,
    _ticks_since_keydown_before,
    plan_full_song_windows,
)
from orbit4k.validator import EMPTY, LN_END, LN_START, TAP


def test_full_song_windows_cover_every_tick_without_gaps():
    total = 1000
    window = 384
    context = 96
    plans = plan_full_song_windows(total, window, context)

    filled = 0
    for index, plan in enumerate(plans):
        assert plan["new_start"] == filled
        assert plan["target_end"] > filled
        assert plan["start"] <= filled
        assert plan["memory_end"] > plan["target_end"] or plan["is_final"]
        if index > 0:
            assert plan["prefix_ticks"] == context
        filled = plan["target_end"]

    assert filled == total
    assert plans[-1]["is_final"] == 1


def test_internal_window_has_left_prompt_and_right_audio_lookahead():
    plans = plan_full_song_windows(total_ticks=1200, window_ticks=384, context_ticks=96)
    second = plans[1]

    assert second["prefix_ticks"] == 96
    assert second["memory_end"] - second["target_end"] == 96
    assert second["new_ticks"] == 384 - 2 * 96


def test_short_song_uses_single_final_window():
    plans = plan_full_song_windows(total_ticks=200, window_ticks=384, context_ticks=96)

    assert len(plans) == 1
    assert plans[0]["start"] == 0
    assert plans[0]["target_end"] == 200
    assert plans[0]["memory_end"] == 200
    assert plans[0]["is_final"] == 1


def test_invalid_context_that_consumes_window_is_rejected():
    try:
        plan_full_song_windows(total_ticks=1000, window_ticks=384, context_ticks=192)
    except ValueError as exc:
        assert "twice the context" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ln_state_survives_window_boundary():
    chart = np.full((12, 4), EMPTY, dtype=np.uint8)
    chart[3, 1] = LN_START
    # Tick 8 is an internal window boundary while lane 1 is still held.
    active, age = _state_before(chart, 8)

    assert active.tolist() == [0, 1, 0, 0]
    assert int(age[1]) == 5

    chart[9, 1] = LN_END
    active_after, age_after = _state_before(chart, 10)
    assert active_after.tolist() == [0, 0, 0, 0]
    assert int(age_after[1]) == 0


def test_ticks_since_keydown_preserves_silence_context():
    chart = np.full((16, 4), EMPTY, dtype=np.uint8)
    chart[4, 0] = TAP
    chart[9, 2] = LN_START

    assert _ticks_since_keydown_before(chart, 10) == 0
    assert _ticks_since_keydown_before(chart, 14) == 4
