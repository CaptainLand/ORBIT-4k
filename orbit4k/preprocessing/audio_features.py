from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 441
    n_mels: int = 128
    f_min: float = 30.0
    f_max: float = 18000.0


def load_mono(path: str | Path, target_sample_rate: int) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2:
        raise ValueError(f"unexpected waveform shape: {tuple(waveform.shape)}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    peak = waveform.abs().max().clamp_min(1e-6)
    return (waveform / peak).float()


def extract_log_mel(path: str | Path, config: AudioFeatureConfig) -> tuple[np.ndarray, float]:
    waveform = load_mono(path, config.sample_rate)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.f_min,
        f_max=min(config.f_max, config.sample_rate / 2),
        n_mels=config.n_mels,
        power=2.0,
        center=True,
    )
    with torch.no_grad():
        mel = transform(waveform).squeeze(0)
        log_mel = torch.log(mel.clamp_min(1e-5))
        # Per-bin normalization keeps the cache self-contained and numerically stable.
        mean = log_mel.mean(dim=1, keepdim=True)
        std = log_mel.std(dim=1, keepdim=True).clamp_min(1e-4)
        normalized = (log_mel - mean) / std
    duration_ms = waveform.shape[-1] / config.sample_rate * 1000.0
    return normalized.transpose(0, 1).cpu().numpy().astype(np.float16), float(duration_ms)


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
    tick_ms = beat_length_ms / ticks_per_quarter
    tick_indices = start_tick + np.arange(length, dtype=np.float32)
    times_ms = offset_ms + tick_indices * tick_ms
    frame_positions = times_ms * sample_rate / (1000.0 * hop_length)
    left = np.floor(frame_positions).astype(np.int64)
    alpha = (frame_positions - left).astype(np.float32)
    left = np.clip(left, 0, max(0, len(mel_frames) - 1))
    right = np.clip(left + 1, 0, max(0, len(mel_frames) - 1))
    aligned = mel_frames[left].astype(np.float32) * (1.0 - alpha[:, None]) + mel_frames[right].astype(np.float32) * alpha[:, None]
    # Negative times are padding before the audio starts.
    aligned[times_ms < 0] = 0.0
    return aligned.astype(np.float32)
