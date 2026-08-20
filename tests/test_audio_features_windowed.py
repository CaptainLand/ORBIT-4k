from __future__ import annotations

import numpy as np

from orbit4k.preprocessing.audio_features import beat_synchronous_audio_tokens
from orbit4k.preprocessing.audio_features_windowed import (
    _frame_bounds,
    beat_synchronous_audio_tokens_windowed,
)


def _synthetic_audio(frames: int = 4000):
    rng = np.random.default_rng(20260820)
    log_mel = rng.normal(-2.0, 1.2, size=(frames, 128)).astype(np.float16)
    mean = log_mel.astype(np.float32).mean(axis=0)
    std = np.maximum(log_mel.astype(np.float32).std(axis=0), 1e-4)
    energy = rng.normal(-1.0, 0.7, size=(frames, 4)).astype(np.float16)
    return log_mel, mean, std, energy


def test_windowed_tokenizer_matches_full_v2_interior():
    log_mel, mean, std, energy = _synthetic_audio()
    kwargs = dict(
        energy_median=-1.0,
        energy_mad=0.8,
        duration_ms=19000.0,
        start_tick=240,
        length=192,
        offset_ms=25.0,
        beat_length_ms=333.333333,
        ticks_per_quarter=24,
        hop_length=220,
        sample_rate=44100,
    )
    full = beat_synchronous_audio_tokens(log_mel, mean, std, energy, **kwargs)
    bounded = beat_synchronous_audio_tokens_windowed(log_mel, mean, std, energy, **kwargs)

    assert full.shape == bounded.shape == (192, 520)
    assert np.allclose(full, bounded, rtol=2e-5, atol=2e-5)


def test_windowed_tokenizer_matches_song_start_padding():
    log_mel, mean, std, energy = _synthetic_audio(frames=1000)
    kwargs = dict(
        energy_median=-1.0,
        energy_mad=0.8,
        duration_ms=4500.0,
        start_tick=0,
        length=96,
        offset_ms=-40.0,
        beat_length_ms=500.0,
        ticks_per_quarter=24,
        hop_length=220,
        sample_rate=44100,
    )
    full = beat_synchronous_audio_tokens(log_mel, mean, std, energy, **kwargs)
    bounded = beat_synchronous_audio_tokens_windowed(log_mel, mean, std, energy, **kwargs)

    assert np.allclose(full, bounded, rtol=2e-5, atol=2e-5)


def test_frame_slice_is_bounded_for_marathon_song():
    # About one hour at ~5 ms/frame, matching the RAM failure that motivated
    # this path. A normal 16-measure crop should touch only a tiny fraction.
    total_frames = 719_142
    centers = 100_000.0 + np.arange(1536, dtype=np.float32) * 13.9
    start, end = _frame_bounds(
        centers,
        13.9,
        total_frames=total_frames,
        hop_length=220,
        sample_rate=44100,
    )

    assert 0 <= start < end <= total_frames
    assert end - start < 10_000
    assert end - start < total_frames // 50
