from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


AUDIO_FEATURE_VERSION = 2
SPECTRAL_VIEW_DIM = 128 * 4
ENERGY_FEATURE_DIM = 8
AUDIO_TOKEN_DIM = SPECTRAL_VIEW_DIM + ENERGY_FEATURE_DIM


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 44100
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 220
    n_mels: int = 128
    f_min: float = 30.0
    f_max: float = 18000.0
    chunk_seconds: float = 30.0


def load_mono(path: str | Path, target_sample_rate: int) -> torch.Tensor:
    """Decode to mono float audio while preserving the source recording level."""
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise ValueError(f"unexpected waveform shape: {tuple(waveform.shape)}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform.float()


def _mel_transform(config: AudioFeatureConfig, *, center: bool) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=min(config.f_max, config.sample_rate / 2),
        n_mels=config.n_mels,
        power=2.0,
        center=center,
    )


def chunked_mel_power(
    waveform: torch.Tensor,
    config: AudioFeatureConfig,
) -> np.ndarray:
    """Compute the same center-aligned Mel grid in bounded STFT chunks.

    A whole-song ``MelSpectrogram`` materializes a very large STFT tensor for
    marathon audio. We reproduce ``center=True`` frame alignment by reflect
    padding once, slicing only the samples needed by a bounded range of output
    frames, and running ``center=False`` on each slice.

    The returned array is [frames, n_mels] float32. Peak STFT memory is bounded
    by ``chunk_seconds`` instead of song duration.
    """
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(f"expected mono waveform [1, samples], got {tuple(waveform.shape)}")
    if config.win_length > config.n_fft:
        raise ValueError("win_length must not exceed n_fft")

    signal = waveform[0]
    sample_count = int(signal.numel())
    if sample_count == 0:
        return np.zeros((0, config.n_mels), dtype=np.float32)

    pad = config.n_fft // 2
    # torch.stft(center=True) uses reflect padding by default. Very short clips
    # cannot be reflected by n_fft//2, so use constant padding as a safe fallback.
    if sample_count > pad:
        padded = F.pad(signal[None, None, :], (pad, pad), mode="reflect")[0, 0]
    else:
        padded = F.pad(signal, (pad, pad))

    total_frames = 1 + sample_count // config.hop_length
    frames_per_chunk = max(
        1,
        int(float(config.chunk_seconds) * config.sample_rate / config.hop_length),
    )
    transform = _mel_transform(config, center=False)
    result = np.empty((total_frames, config.n_mels), dtype=np.float32)

    with torch.no_grad():
        for frame_start in range(0, total_frames, frames_per_chunk):
            frame_end = min(total_frames, frame_start + frames_per_chunk)
            sample_start = frame_start * config.hop_length
            sample_end = (frame_end - 1) * config.hop_length + config.n_fft
            segment = padded[sample_start:sample_end].unsqueeze(0)
            mel = transform(segment).squeeze(0).transpose(0, 1)
            expected = frame_end - frame_start
            if mel.shape[0] != expected:
                raise RuntimeError(
                    f"chunked Mel frame mismatch: {mel.shape[0]} != {expected}"
                )
            result[frame_start:frame_end] = mel.cpu().numpy().astype(np.float32, copy=False)

    return result


def extract_audio_cache(path: str | Path, config: AudioFeatureConfig) -> dict[str, np.ndarray | float | int]:
    """
    Build a song-level cache.

    Raw log-Mel values and song statistics are stored separately so the
    beat-synchronous tokenizer can expose both absolute/mastering information
    and song-relative structure. STFT work is chunked to keep RAM bounded for
    long songs.
    """
    waveform = load_mono(path, config.sample_rate)
    mel_power_np = chunked_mel_power(waveform, config)
    if len(mel_power_np) == 0:
        raise ValueError("decoded audio contains no analysis frames")

    mel_power = np.maximum(mel_power_np, 1e-8)
    log_mel = np.log(mel_power).astype(np.float32)

    # Statistics are computed in float64 accumulation for long songs, while the
    # stored frame cache stays float16 to control disk usage.
    mel_mean = log_mel.mean(axis=0, dtype=np.float64).astype(np.float32)
    mel_std = log_mel.std(axis=0, dtype=np.float64).astype(np.float32)
    mel_std = np.maximum(mel_std, 1e-4)

    one_third = config.n_mels // 3
    two_thirds = (config.n_mels * 2) // 3

    def log_band(start: int, end: int) -> np.ndarray:
        return np.log(np.maximum(mel_power[:, start:end].mean(axis=1), 1e-8)).astype(np.float32)

    log_energy = np.stack(
        [
            np.log(np.maximum(mel_power.mean(axis=1), 1e-8)).astype(np.float32),
            log_band(0, one_third),
            log_band(one_third, two_thirds),
            log_band(two_thirds, config.n_mels),
        ],
        axis=1,
    )
    total_energy = log_energy[:, 0]
    energy_median = float(np.median(total_energy))
    energy_mad = float(
        max(np.median(np.abs(total_energy - energy_median)) * 1.4826, 1e-3)
    )

    duration_ms = waveform.shape[-1] / config.sample_rate * 1000.0
    return {
        "log_mel": log_mel.astype(np.float16),
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "log_energy": log_energy.astype(np.float16),
        "energy_median": energy_median,
        "energy_mad": energy_mad,
        "duration_ms": float(duration_ms),
        "sample_rate": int(config.sample_rate),
        "hop_length": int(config.hop_length),
        "feature_version": int(AUDIO_FEATURE_VERSION),
    }


def extract_log_mel(path: str | Path, config: AudioFeatureConfig) -> tuple[np.ndarray, float]:
    """Compatibility helper for older callers; V0 training uses extract_audio_cache."""
    cache = extract_audio_cache(path, config)
    log_mel = np.asarray(cache["log_mel"], dtype=np.float32)
    mean = np.asarray(cache["mel_mean"], dtype=np.float32)
    std = np.asarray(cache["mel_std"], dtype=np.float32)
    normalized = (log_mel - mean[None]) / std[None]
    return normalized.astype(np.float16), float(cache["duration_ms"])


def _interpolate_frames(
    frames: np.ndarray,
    times_ms: np.ndarray,
    *,
    hop_length: int,
    sample_rate: int,
    duration_ms: float,
) -> np.ndarray:
    if len(frames) == 0:
        return np.zeros((*times_ms.shape, frames.shape[-1]), dtype=np.float32)
    positions = times_ms * sample_rate / (1000.0 * hop_length)
    left_unclipped = np.floor(positions).astype(np.int64)
    alpha = (positions - left_unclipped).astype(np.float32)
    left = np.clip(left_unclipped, 0, len(frames) - 1)
    right = np.clip(left + 1, 0, len(frames) - 1)
    values = (
        frames[left].astype(np.float32) * (1.0 - alpha[..., None])
        + frames[right].astype(np.float32) * alpha[..., None]
    )
    valid = (times_ms >= 0.0) & (times_ms <= duration_ms)
    values *= valid[..., None]
    return values


def _weighted_local_pool(
    frames: np.ndarray,
    centers_ms: np.ndarray,
    tick_ms: float,
    *,
    hop_length: int,
    sample_rate: int,
    duration_ms: float,
) -> np.ndarray:
    """Five-point pooling over the full +/-1/192 interval around a 1/96 tick."""
    offsets = np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=np.float32) * tick_ms
    weights = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    weights /= weights.sum()
    samples = _interpolate_frames(
        frames,
        centers_ms[:, None] + offsets[None],
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    return np.sum(samples * weights[None, :, None], axis=1)


def _context_pool(
    frames: np.ndarray,
    centers_ms: np.ndarray,
    tick_ms: float,
    *,
    hop_length: int,
    sample_rate: int,
    duration_ms: float,
    half_width_ticks: float = 12.0,
) -> np.ndarray:
    """Coarse +/-1/8-note context: 12 grid ticks each side at 24 TPQ."""
    offsets = np.linspace(-half_width_ticks, half_width_ticks, 9, dtype=np.float32) * tick_ms
    samples = _interpolate_frames(
        frames,
        centers_ms[:, None] + offsets[None],
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    return samples.mean(axis=1)


def beat_synchronous_audio_tokens(
    log_mel_frames: np.ndarray,
    mel_mean: np.ndarray,
    mel_std: np.ndarray,
    log_energy_frames: np.ndarray,
    *,
    energy_median: float,
    energy_mad: float,
    duration_ms: float,
    start_tick: int,
    length: int,
    offset_ms: float,
    beat_length_ms: float,
    ticks_per_quarter: int,
    hop_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Convert continuous audio into one 520-d feature token per chart-grid tick."""
    tick_ms = beat_length_ms / ticks_per_quarter
    tick_indices = start_tick + np.arange(length, dtype=np.float32)
    centers_ms = offset_ms + tick_indices * tick_ms

    raw = log_mel_frames.astype(np.float32)
    mean = mel_mean.astype(np.float32)
    std = np.maximum(mel_std.astype(np.float32), 1e-4)
    normalized = (raw - mean[None]) / std[None]

    flux_frames = np.zeros_like(normalized)
    if len(normalized) > 1:
        flux_frames[1:] = np.maximum(normalized[1:] - normalized[:-1], 0.0)

    raw_local = _weighted_local_pool(
        raw, centers_ms, tick_ms,
        hop_length=hop_length, sample_rate=sample_rate, duration_ms=duration_ms,
    )
    normalized_local = _weighted_local_pool(
        normalized, centers_ms, tick_ms,
        hop_length=hop_length, sample_rate=sample_rate, duration_ms=duration_ms,
    )
    flux_local = _weighted_local_pool(
        flux_frames, centers_ms, tick_ms,
        hop_length=hop_length, sample_rate=sample_rate, duration_ms=duration_ms,
    )
    context = _context_pool(
        normalized, centers_ms, tick_ms,
        hop_length=hop_length, sample_rate=sample_rate, duration_ms=duration_ms,
    )
    contrast = normalized_local - context

    local_energy = _weighted_local_pool(
        log_energy_frames.astype(np.float32), centers_ms, tick_ms,
        hop_length=hop_length, sample_rate=sample_rate, duration_ms=duration_ms,
    )
    relative_energy = (local_energy[:, 0] - float(energy_median)) / max(float(energy_mad), 1e-3)
    flux_strength = flux_local.mean(axis=1)
    contrast_strength = np.abs(contrast).mean(axis=1)

    raw_scaled = np.clip(raw_local / 6.0, -4.0, 4.0)
    energy_scaled = np.stack(
        [
            np.clip(local_energy[:, 0] / 6.0, -4.0, 4.0),
            np.clip(local_energy[:, 1] / 6.0, -4.0, 4.0),
            np.clip(local_energy[:, 2] / 6.0, -4.0, 4.0),
            np.clip(local_energy[:, 3] / 6.0, -4.0, 4.0),
            np.clip(relative_energy / 4.0, -4.0, 4.0),
            np.clip(flux_strength / 4.0, 0.0, 4.0),
            np.clip(contrast_strength / 4.0, 0.0, 4.0),
            np.full(length, np.clip(float(energy_median) / 6.0, -4.0, 4.0), dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    tokens = np.concatenate(
        [raw_scaled, normalized_local, flux_local, contrast, energy_scaled],
        axis=1,
    ).astype(np.float32)
    if tokens.shape[1] != AUDIO_TOKEN_DIM:
        raise RuntimeError(f"audio token width mismatch: {tokens.shape[1]} != {AUDIO_TOKEN_DIM}")

    valid = (centers_ms >= 0.0) & (centers_ms <= duration_ms)
    tokens[~valid] = 0.0
    return tokens


def align_mel_to_ticks(
    mel_frames: np.ndarray,
    *,
    start_tick: int,
    length: int,
    offset_ms: float,
    beat_length_ms: float,
    ticks_per_quarter: int,
    hop_length: int,
    sample_rate: int,
) -> np.ndarray:
    """Legacy point-sampling alignment retained only for compatibility/debugging."""
    tick_ms = beat_length_ms / ticks_per_quarter
    tick_indices = start_tick + np.arange(length, dtype=np.float32)
    times_ms = offset_ms + tick_indices * tick_ms
    duration_ms = max(0.0, (len(mel_frames) - 1) * hop_length / sample_rate * 1000.0)
    return _interpolate_frames(
        mel_frames,
        times_ms,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    ).astype(np.float32)
