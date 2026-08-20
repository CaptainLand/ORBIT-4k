from __future__ import annotations

import numpy as np

from .audio_features import AUDIO_TOKEN_DIM


def _interpolate_window(
    frames: np.ndarray,
    times_ms: np.ndarray,
    *,
    frame_offset: int,
    hop_length: int,
    sample_rate: int,
    duration_ms: float,
) -> np.ndarray:
    """Interpolate a bounded frame slice using absolute song timestamps."""
    if len(frames) == 0:
        return np.zeros((*times_ms.shape, frames.shape[-1]), dtype=np.float32)
    global_positions = times_ms * sample_rate / (1000.0 * hop_length)
    positions = global_positions - float(frame_offset)
    left_unclipped = np.floor(positions).astype(np.int64)
    alpha = (positions - left_unclipped).astype(np.float32)
    left = np.clip(left_unclipped, 0, len(frames) - 1)
    right = np.clip(left + 1, 0, len(frames) - 1)
    values = (
        frames[left].astype(np.float32, copy=False) * (1.0 - alpha[..., None])
        + frames[right].astype(np.float32, copy=False) * alpha[..., None]
    )
    valid = (times_ms >= 0.0) & (times_ms <= duration_ms)
    values *= valid[..., None]
    return values


def _weighted_local_pool(
    frames: np.ndarray,
    centers_ms: np.ndarray,
    tick_ms: float,
    *,
    frame_offset: int,
    hop_length: int,
    sample_rate: int,
    duration_ms: float,
) -> np.ndarray:
    offsets = np.asarray([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=np.float32) * tick_ms
    weights = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    weights /= weights.sum()
    samples = _interpolate_window(
        frames,
        centers_ms[:, None] + offsets[None],
        frame_offset=frame_offset,
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
    frame_offset: int,
    hop_length: int,
    sample_rate: int,
    duration_ms: float,
    half_width_ticks: float = 12.0,
) -> np.ndarray:
    offsets = np.linspace(-half_width_ticks, half_width_ticks, 9, dtype=np.float32) * tick_ms
    samples = _interpolate_window(
        frames,
        centers_ms[:, None] + offsets[None],
        frame_offset=frame_offset,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    return samples.mean(axis=1)


def _frame_bounds(
    centers_ms: np.ndarray,
    tick_ms: float,
    *,
    total_frames: int,
    hop_length: int,
    sample_rate: int,
) -> tuple[int, int]:
    """Return a small frame slice covering all local/context interpolation queries."""
    if total_frames <= 0:
        return 0, 0
    if len(centers_ms) == 0:
        return 0, 1

    # Context pooling is the widest consumer (+/- 12 ticks). Two extra frames
    # cover interpolation's right neighbour and the previous frame used by flux.
    min_ms = float(np.min(centers_ms)) - 12.0 * float(tick_ms)
    max_ms = float(np.max(centers_ms)) + 12.0 * float(tick_ms)
    scale = float(sample_rate) / (1000.0 * float(hop_length))
    raw_start = int(np.floor(min_ms * scale)) - 2
    raw_end = int(np.ceil(max_ms * scale)) + 3

    start = max(0, min(total_frames - 1, raw_start))
    end = max(start + 1, min(total_frames, raw_end))
    return start, end


def beat_synchronous_audio_tokens_windowed(
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
    """Memory-bounded equivalent of Audio Feature V2 tokenization.

    V0's tokenizer converted the *entire song* from float16 to float32 and then
    allocated another full-song normalized matrix for every training sample.
    A one-hour cache can therefore create multiple 300+ MiB temporaries inside
    each DataLoader worker. V1 only needs one training window, so this function
    slices the Mel/energy frames required by that window (+/- context), promotes
    only that slice to float32, and computes the same 520-d token semantics.
    """
    length = int(length)
    if length <= 0:
        return np.zeros((0, AUDIO_TOKEN_DIM), dtype=np.float32)

    tick_ms = float(beat_length_ms) / int(ticks_per_quarter)
    tick_indices = int(start_tick) + np.arange(length, dtype=np.float32)
    centers_ms = float(offset_ms) + tick_indices * tick_ms

    total_frames = int(len(log_mel_frames))
    frame_start, frame_end = _frame_bounds(
        centers_ms,
        tick_ms,
        total_frames=total_frames,
        hop_length=int(hop_length),
        sample_rate=int(sample_rate),
    )
    if frame_end <= frame_start:
        return np.zeros((length, AUDIO_TOKEN_DIM), dtype=np.float32)

    mean = np.asarray(mel_mean, dtype=np.float32)
    std = np.maximum(np.asarray(mel_std, dtype=np.float32), 1e-4)

    # Only this bounded slice is promoted from the float16 cache to float32.
    raw = np.asarray(log_mel_frames[frame_start:frame_end], dtype=np.float32)
    normalized = (raw - mean[None]) / std[None]

    flux_frames = np.zeros_like(normalized)
    if frame_start > 0:
        previous_raw = np.asarray(log_mel_frames[frame_start - 1], dtype=np.float32)
        previous_normalized = (previous_raw - mean) / std
        flux_frames[0] = np.maximum(normalized[0] - previous_normalized, 0.0)
    if len(normalized) > 1:
        flux_frames[1:] = np.maximum(normalized[1:] - normalized[:-1], 0.0)

    raw_local = _weighted_local_pool(
        raw,
        centers_ms,
        tick_ms,
        frame_offset=frame_start,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    normalized_local = _weighted_local_pool(
        normalized,
        centers_ms,
        tick_ms,
        frame_offset=frame_start,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    flux_local = _weighted_local_pool(
        flux_frames,
        centers_ms,
        tick_ms,
        frame_offset=frame_start,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    context = _context_pool(
        normalized,
        centers_ms,
        tick_ms,
        frame_offset=frame_start,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
    )
    contrast = normalized_local - context

    energy = np.asarray(log_energy_frames[frame_start:frame_end], dtype=np.float32)
    local_energy = _weighted_local_pool(
        energy,
        centers_ms,
        tick_ms,
        frame_offset=frame_start,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_ms=duration_ms,
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
            np.full(
                length,
                np.clip(float(energy_median) / 6.0, -4.0, 4.0),
                dtype=np.float32,
            ),
        ],
        axis=1,
    ).astype(np.float32)

    tokens = np.concatenate(
        [raw_scaled, normalized_local, flux_local, contrast, energy_scaled],
        axis=1,
    ).astype(np.float32)
    if tokens.shape[1] != AUDIO_TOKEN_DIM:
        raise RuntimeError(f"audio token width mismatch: {tokens.shape[1]} != {AUDIO_TOKEN_DIM}")

    valid = (centers_ms >= 0.0) & (centers_ms <= float(duration_ms))
    tokens[~valid] = 0.0
    return tokens
