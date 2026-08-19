import numpy as np
import torch
import torchaudio

from orbit4k.preprocessing.audio_features import (
    AUDIO_TOKEN_DIM,
    AudioFeatureConfig,
    beat_synchronous_audio_tokens,
    chunked_mel_power,
)


def _make_tokens(level_shift: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.4, size=(220, 128)).astype(np.float32)
    log_mel = base + level_shift
    mel_mean = log_mel.mean(axis=0)
    mel_std = log_mel.std(axis=0) + 1e-3
    total = np.linspace(-2.0, 1.0, len(log_mel), dtype=np.float32) + level_shift
    energy = np.stack([total, total - 0.2, total - 0.4, total - 0.6], axis=1)
    return beat_synchronous_audio_tokens(
        log_mel,
        mel_mean,
        mel_std,
        energy,
        energy_median=float(np.median(total)),
        energy_mad=1.0,
        duration_ms=1100.0,
        start_tick=0,
        length=48,
        offset_ms=0.0,
        beat_length_ms=333.333,
        ticks_per_quarter=24,
        hop_length=220,
        sample_rate=44100,
    )


def test_beat_synchronous_token_shape_and_finiteness():
    tokens = _make_tokens(0.0)
    assert tokens.shape == (48, AUDIO_TOKEN_DIM)
    assert np.isfinite(tokens).all()


def test_normalized_view_is_level_invariant_but_raw_view_is_not():
    quiet = _make_tokens(-2.0)
    loud = _make_tokens(2.0)
    np.testing.assert_allclose(quiet[:, 128:256], loud[:, 128:256], atol=1e-4, rtol=1e-4)
    assert np.mean(np.abs(quiet[:, :128] - loud[:, :128])) > 0.1


def test_chunked_mel_matches_centered_whole_song_alignment():
    torch.manual_seed(3)
    config = AudioFeatureConfig(
        sample_rate=16000,
        n_fft=512,
        win_length=512,
        hop_length=160,
        n_mels=40,
        f_min=30.0,
        f_max=7600.0,
        chunk_seconds=0.25,
    )
    waveform = torch.randn(1, config.sample_rate * 2)
    whole = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max,
        power=2.0,
        center=True,
    )(waveform).squeeze(0).transpose(0, 1).numpy()
    chunked = chunked_mel_power(waveform, config)

    assert chunked.shape == whole.shape
    np.testing.assert_allclose(chunked, whole, atol=1e-3, rtol=1e-5)
