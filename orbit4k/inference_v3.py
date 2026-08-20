from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .inference import (
    _auto_max_chord,
    _model_from_checkpoint,
    _safe_stem,
    chart_statistics,
    constrained_tick_states,
    write_osu,
)
from .model import BOS_STATE, Orbit4KV0
from .preprocessing.audio_features import (
    AudioFeatureConfig,
    beat_synchronous_audio_tokens,
    extract_audio_cache,
)
from .validator import LN_END, LN_START, TAP, validate_and_repair

ProgressCallback = Callable[[dict], None]


def _emit(callback: ProgressCallback | None, **payload) -> None:
    if callback is not None:
        callback(payload)


def _auto_base_threshold(stars: float) -> float:
    """Absolute V3 ceiling before local/relative adaptation."""
    return float(max(0.30, min(0.46, 0.40 - 0.012 * (float(stars) - 6.0))))


def _auto_floor_threshold(stars: float) -> float:
    """Floor that keeps low-probability noise from becoming arbitrary notes."""
    return float(max(0.10, min(0.18, 0.14 - 0.006 * (float(stars) - 6.0))))


def audio_activity_curve(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build robust 0..1 activity and local-peak signals from Audio Feature V2."""
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 8:
        raise ValueError("expected beat-synchronous audio tokens [T,D]")
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

    # Final energy feature layout: total/low/mid/high, relative, flux, contrast, median.
    relative_energy = values[:, -4]
    flux = values[:, -3]
    contrast = values[:, -2]
    raw = 0.40 * relative_energy + 0.35 * flux + 0.25 * contrast

    if len(raw) >= 4:
        low, high = np.quantile(raw, [0.10, 0.90])
    else:
        low, high = float(raw.min()), float(raw.max())
    span = max(float(high - low), 1e-5)
    activity = np.clip((raw - float(low)) / span, 0.0, 1.0).astype(np.float32)

    peaks = np.zeros(len(activity), dtype=bool)
    for index in range(len(activity)):
        start = max(0, index - 3)
        end = min(len(activity), index + 4)
        local = activity[start:end]
        peaks[index] = bool(
            activity[index] >= 0.25
            and activity[index] >= float(local.max()) - 1e-6
        )
    return activity, peaks


def adaptive_onset_threshold(
    history: list[float],
    *,
    stars: float,
    ticks_since_keydown: int,
    activity: float,
    custom_threshold: float = 0.0,
    history_window: int = 24,
    relative_quantile: float = 0.85,
    relative_margin: float = 0.025,
) -> float:
    """Resolve one V3 onset gate from absolute, relative and silence signals.

    The positive margin is important: a flat sequence such as 0.20,0.20,...
    must not classify every tick as a relative peak merely because its q85 is
    also 0.20. Real peaks only need to stand a little above their local baseline.
    """
    base = float(custom_threshold) if custom_threshold > 0 else _auto_base_threshold(stars)
    floor = _auto_floor_threshold(stars)
    window = history[-max(4, int(history_window)) :]

    if len(window) >= 4:
        relative = float(np.quantile(np.asarray(window, dtype=np.float32), relative_quantile))
        threshold = min(base, max(floor, relative + float(relative_margin)))
    else:
        threshold = base

    activity_value = float(np.clip(activity, 0.0, 1.0))
    # Audio activity is a nudge, not the main gate: +/-0.02 maximum.
    threshold += (0.5 - activity_value) * 0.04

    # 24 TPQ: relaxation begins after half a beat and reaches full strength at 1.5 beats.
    gap_progress = float(np.clip((int(ticks_since_keydown) - 12) / 24.0, 0.0, 1.0))
    threshold -= 0.18 * gap_progress * (0.25 + 0.75 * activity_value)
    return float(np.clip(threshold, floor, 0.80))


@torch.inference_mode()
def adaptive_generate_window(
    model: Orbit4KV0,
    audio: torch.Tensor,
    tick: torch.Tensor,
    bpm: torch.Tensor,
    stars: torch.Tensor,
    *,
    audio_activity: np.ndarray,
    audio_peaks: np.ndarray,
    temperature: float = 0.85,
    onset_threshold: float = 0.0,
    lane_threshold: float = 0.32,
    ln_start_margin: float = 1.25,
    max_chord: int = 0,
    progress_callback=None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """V3 free-run: relative onset peaks + active-audio silence recovery."""
    model.eval()
    batch, length, _ = audio.shape
    if len(audio_activity) != length or len(audio_peaks) != length:
        raise ValueError("audio activity arrays must match generation length")

    result = torch.zeros((batch, length, 4), dtype=torch.long, device=audio.device)
    chart_input = torch.full_like(result, BOS_STATE)
    active_ln = torch.zeros_like(result)
    active_now = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)
    ln_age = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)
    mask = torch.ones((batch, length), dtype=torch.float32, device=audio.device)
    memory, memory_padding = model.encode_audio(audio, tick, mask)

    histories: list[list[float]] = [[] for _ in range(batch)]
    ticks_since = [0 for _ in range(batch)]
    max_silence = [0 for _ in range(batch)]
    onset_probability_sum = 0.0
    threshold_sum = 0.0
    threshold_min = 1.0
    threshold_max = 0.0
    gated_ticks = 0
    relative_gates = 0
    recovery_gates = 0
    fallback_count = 0
    releases = 0
    report_every = max(1, length // 100)

    for position in range(length):
        if position > 0:
            chart_input[:, position] = result[:, position - 1]
        active_ln[:, position] = active_now
        prefix = position + 1
        outputs = model.decode_from_memory(
            memory,
            chart_input[:, :prefix],
            active_ln[:, :prefix],
            tick[:, :prefix],
            bpm,
            stars,
            target_mask=mask[:, :prefix],
            memory_padding_bool=memory_padding,
        )
        lane_logits = outputs["logits"][:, -1]
        onset_logits = outputs["onset_logits"][:, -1]
        next_batch = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)

        for item in range(batch):
            temp = max(float(temperature), 1e-4)
            onset_probability = float(torch.sigmoid(onset_logits[item] / temp).item())
            histories[item].append(onset_probability)
            if len(histories[item]) > 48:
                del histories[item][:-48]

            activity = float(audio_activity[position])
            threshold = adaptive_onset_threshold(
                histories[item],
                stars=float(stars[item].item()),
                ticks_since_keydown=ticks_since[item],
                activity=activity,
                custom_threshold=float(onset_threshold),
            )
            base = (
                float(onset_threshold)
                if onset_threshold > 0
                else _auto_base_threshold(float(stars[item].item()))
            )
            floor = _auto_floor_threshold(float(stars[item].item()))
            recovery = False

            # After one beat without a key-down, an audio-local peak can break a
            # self-reinforcing EMPTY loop even when the onset head is under-calibrated.
            if (
                ticks_since[item] >= 24
                and bool(audio_peaks[position])
                and activity >= 0.45
                and onset_probability >= max(0.10, floor * 0.80)
                and onset_probability < threshold
            ):
                threshold = max(floor * 0.80, onset_probability - 1e-5)
                recovery = True

            # After two beats, permit a modest onset in active audio. Truly quiet
            # sections remain exempt, so this is anti-collapse rather than a metronome.
            if (
                ticks_since[item] >= 48
                and activity >= 0.35
                and onset_probability >= 0.08
                and onset_probability < threshold
            ):
                threshold = max(0.08, onset_probability - 1e-5)
                recovery = True

            next_state, diag = constrained_tick_states(
                lane_logits[item],
                onset_logits[item],
                active_now[item],
                ln_age[item],
                stars=float(stars[item].item()),
                temperature=temperature,
                onset_threshold=threshold,
                lane_threshold=lane_threshold,
                ln_start_margin=ln_start_margin,
                max_chord=max_chord,
                final_tick=position + 1 == length,
            )
            next_batch[item] = next_state

            has_keydown = bool(((next_state == TAP) | (next_state == LN_START)).any().item())
            if has_keydown:
                ticks_since[item] = 0
            else:
                ticks_since[item] += 1
                max_silence[item] = max(max_silence[item], ticks_since[item])

            onset_probability_sum += onset_probability
            threshold_sum += threshold
            threshold_min = min(threshold_min, threshold)
            threshold_max = max(threshold_max, threshold)
            gated = bool(diag["onset_gate"])
            gated_ticks += int(gated)
            relative_gates += int(gated and onset_probability < base and not recovery)
            recovery_gates += int(gated and recovery)
            fallback_count += int(bool(diag["lane_fallback"]))
            releases += int(diag["release_count"])

        result[:, position] = next_batch
        previous_active = active_now.bool()
        started = next_batch == LN_START
        ended = next_batch == LN_END
        continuing = previous_active & ~ended
        active_now = torch.where(started, 1, torch.where(ended, 0, active_now))
        ln_age = torch.where(
            started,
            torch.ones_like(ln_age),
            torch.where(continuing, ln_age + 1, torch.zeros_like(ln_age)),
        )

        if progress_callback is not None and (
            position == 0
            or position + 1 == length
            or (position + 1) % report_every == 0
        ):
            progress_callback(position + 1, length)

    denominator = max(1, batch * length)
    diagnostics: dict[str, float | int] = {
        "decoder_version": 3,
        "mean_onset_probability": onset_probability_sum / denominator,
        "mean_effective_threshold": threshold_sum / denominator,
        "min_effective_threshold": threshold_min,
        "max_effective_threshold": threshold_max,
        "gated_onset_ticks": gated_ticks,
        "relative_peak_gates": relative_gates,
        "silence_recovery_gates": recovery_gates,
        "max_silence_ticks": max(max_silence) if max_silence else 0,
        "lane_fallbacks": fallback_count,
        "releases": releases,
        "base_onset_threshold": (
            float(onset_threshold)
            if onset_threshold > 0
            else _auto_base_threshold(float(stars[0].item()))
        ),
        "floor_onset_threshold": _auto_floor_threshold(float(stars[0].item())),
        "relative_quantile": 0.85,
        "relative_margin": 0.025,
        "lane_threshold": float(lane_threshold),
        "ln_start_margin": float(ln_start_margin),
        "max_chord_limit": (
            _auto_max_chord(float(stars[0].item())) if max_chord <= 0 else int(max_chord)
        ),
    }
    return result, diagnostics


def _write_osu_v3(
    chart: np.ndarray,
    output_path: Path,
    *,
    audio_filename: str,
    bpm: float,
    offset_ms: float,
    stars: float,
    title: str,
) -> Path:
    path = write_osu(
        chart,
        output_path,
        audio_filename=audio_filename,
        bpm=bpm,
        offset_ms=offset_ms,
        stars=stars,
        title=title,
    )
    # Reuse the already-tested osu writer and only relabel the inference version.
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("Star Preview V2", "Star Preview V3")
    text = text.replace("constrained decoder v2", "adaptive decoder v3")
    path.write_text(text, encoding="utf-8-sig")
    return path


def generate_preview(
    *,
    checkpoint_path: str | Path,
    audio_path: str | Path,
    output_dir: str | Path,
    bpm: float,
    offset_ms: float,
    stars: float,
    temperature: float = 0.85,
    measures: int = 4,
    seed: int = 20260820,
    onset_threshold: float = 0.0,
    lane_threshold: float = 0.32,
    ln_start_margin: float = 1.25,
    max_chord: int = 0,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if bpm <= 0:
        raise ValueError("BPM must be positive")
    if not 1 <= int(measures) <= 16:
        raise ValueError("preview measures must be between 1 and 16")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _emit(progress_callback, stage="loading_checkpoint", device=str(device), decoder="adaptive_v3")
    model, config, checkpoint = _model_from_checkpoint(checkpoint_path, device)

    _emit(progress_callback, stage="audio_features")
    audio_config = AudioFeatureConfig(**config["audio"])
    cache = extract_audio_cache(audio_path, audio_config)
    beat_length_ms = 60000.0 / float(bpm)
    tick_ms = beat_length_ms / int(config["grid"]["ticks_per_quarter"])
    requested_ticks = int(measures) * int(config["grid"]["ticks_per_measure"])
    duration_ms = float(cache["duration_ms"])
    available_ticks = max(
        0,
        int(math.floor((duration_ms - float(offset_ms)) / tick_ms)) + 1,
    )
    length = min(requested_ticks, available_ticks)
    if length <= 0:
        raise ValueError(
            f"audio has no usable ticks after offset {offset_ms} ms "
            f"(duration={duration_ms:.1f} ms)"
        )

    tokens = beat_synchronous_audio_tokens(
        np.asarray(cache["log_mel"], dtype=np.float32),
        np.asarray(cache["mel_mean"], dtype=np.float32),
        np.asarray(cache["mel_std"], dtype=np.float32),
        np.asarray(cache["log_energy"], dtype=np.float32),
        energy_median=float(cache["energy_median"]),
        energy_mad=float(cache["energy_mad"]),
        duration_ms=duration_ms,
        start_tick=0,
        length=length,
        offset_ms=float(offset_ms),
        beat_length_ms=beat_length_ms,
        ticks_per_quarter=int(config["grid"]["ticks_per_quarter"]),
        hop_length=int(cache["hop_length"]),
        sample_rate=int(cache["sample_rate"]),
    )
    activity, peaks = audio_activity_curve(tokens)

    audio_tensor = torch.from_numpy(tokens).unsqueeze(0).to(device)
    tick_tensor = torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)
    bpm_tensor = torch.tensor([float(bpm)], device=device, dtype=torch.float32)
    stars_tensor = torch.tensor([float(stars)], device=device, dtype=torch.float32)

    def model_progress(current: int, total: int) -> None:
        _emit(
            progress_callback,
            stage="generating",
            current=current,
            total=total,
            percent=round(current / max(total, 1) * 100.0, 2),
        )

    _emit(progress_callback, stage="generating", current=0, total=length, percent=0.0)
    generated_tensor, decoder_diagnostics = adaptive_generate_window(
        model,
        audio_tensor,
        tick_tensor,
        bpm_tensor,
        stars_tensor,
        audio_activity=activity,
        audio_peaks=peaks,
        temperature=float(temperature),
        onset_threshold=float(onset_threshold),
        lane_threshold=float(lane_threshold),
        ln_start_margin=float(ln_start_margin),
        max_chord=int(max_chord),
        progress_callback=model_progress,
    )
    generated = generated_tensor[0].detach().cpu().numpy().astype(np.uint8)

    repaired, repairs = validate_and_repair(generated)
    stats = chart_statistics(repaired)
    stats["repairs"] = len(repairs)
    stats["keydowns_per_measure"] = float(
        stats["keydowns"]
        / max(1e-6, length / int(config["grid"]["ticks_per_measure"]))
    )
    stats["blank_tick_ratio"] = float(
        1.0 - stats["onset_ticks"] / max(1, length)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_target = output_dir / audio_path.name
    if audio_path.resolve() != audio_target.resolve():
        shutil.copy2(audio_path, audio_target)

    title = _safe_stem(audio_path.stem)
    filename = _safe_stem(
        f"{audio_path.stem} [ORBIT-4K {stars:.2f}sr {measures}m V3]"
    ) + ".osu"
    osu_path = _write_osu_v3(
        repaired,
        output_dir / filename,
        audio_filename=audio_target.name,
        bpm=float(bpm),
        offset_ms=float(offset_ms),
        stars=float(stars),
        title=title,
    )

    result = {
        "stage": "complete",
        "decoder": "adaptive_v3",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_score": float(checkpoint.get("score", float("nan"))),
        "device": str(device),
        "audio": str(audio_path.resolve()),
        "output": str(osu_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "bpm": float(bpm),
        "offset_ms": float(offset_ms),
        "stars": float(stars),
        "temperature": float(temperature),
        "requested_measures": int(measures),
        "generated_ticks": int(length),
        "stats": stats,
        "decoder_diagnostics": decoder_diagnostics,
        "repair_examples": repairs[:20],
    }
    _emit(progress_callback, **result)
    return result
