from __future__ import annotations

import numpy as np

from orbit4k.inference_v3 import (
    _auto_base_threshold,
    _auto_floor_threshold,
    adaptive_onset_threshold,
    audio_activity_curve,
)


def test_relative_history_can_lower_v2_style_absolute_gate():
    history = [0.04, 0.05, 0.06, 0.08, 0.07, 0.09, 0.10, 0.30]
    threshold = adaptive_onset_threshold(
        history,
        stars=6.0,
        ticks_since_keydown=4,
        activity=0.80,
    )
    assert threshold < _auto_base_threshold(6.0)
    assert threshold >= _auto_floor_threshold(6.0)


def test_flat_probability_background_is_not_itself_a_peak():
    history = [0.20] * 24
    threshold = adaptive_onset_threshold(
        history,
        stars=6.0,
        ticks_since_keydown=4,
        activity=0.80,
    )
    assert threshold > 0.20


def test_active_silence_relaxes_threshold_progressively():
    history = [0.18, 0.20, 0.22, 0.24, 0.21, 0.23, 0.25, 0.20]
    early = adaptive_onset_threshold(
        history,
        stars=6.0,
        ticks_since_keydown=4,
        activity=0.85,
    )
    late = adaptive_onset_threshold(
        history,
        stars=6.0,
        ticks_since_keydown=36,
        activity=0.85,
    )
    assert late < early
    assert late >= _auto_floor_threshold(6.0)


def test_quiet_audio_is_not_relaxed_as_aggressively_as_active_audio():
    history = [0.15, 0.18, 0.20, 0.22, 0.19, 0.21, 0.23, 0.17]
    quiet = adaptive_onset_threshold(
        history,
        stars=6.0,
        ticks_since_keydown=36,
        activity=0.05,
    )
    active = adaptive_onset_threshold(
        history,
        stars=6.0,
        ticks_since_keydown=36,
        activity=0.95,
    )
    assert active < quiet


def test_audio_activity_curve_detects_relative_transient_peak():
    tokens = np.zeros((16, 520), dtype=np.float32)
    # Final energy feature layout: relative energy=-4, flux=-3, contrast=-2.
    tokens[:, -4] = -0.4
    tokens[8, -4] = 1.0
    tokens[8, -3] = 1.0
    tokens[8, -2] = 1.0

    activity, peaks = audio_activity_curve(tokens)

    assert activity.shape == (16,)
    assert peaks.shape == (16,)
    assert float(activity[8]) > float(activity[0])
    assert bool(peaks[8])
