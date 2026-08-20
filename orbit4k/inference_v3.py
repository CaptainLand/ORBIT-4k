from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .inference import (
    _model_from_checkpoint,
    _safe_stem,
    _auto_max_chord,
    chart_statistics,
    constrained_tick_states,
)
from .model import BOS_STATE, Orbit4KV0
from .preprocessing.audio_features import (
    AudioFeatureConfig,
    beat_synchronous_audio_tokens,
    extract_audio_cache,
)
from .validator import LN_END, LN_START, TAP, validate_and_repair

ProgressCallback = Callable[[dict], None]
LANE_X = (64, 192, 320, 448)


def _emit(callback: ProgressCallback | None, **payload) -> None:
    if callback is not None:
        callback(payload)


def _auto_base_threshold(stars: float) -> float:
    """V3 absolute ceiling before local/relative adaptation.

    V2 used about 0.50 at 6★. That value is too strict for an onset head trained
    on a very sparse 1/96 grid. V3 keeps an absolute ceiling, but lets strong
    *relative* peaks through even when their raw probability is below 0.5.
    """
    return float(max(0.30, min(0.46, 0.40 - 0.012 * (float(stars) - 6.0))))


def _auto_floor_threshold(stars: float) -> float:
    """Never let adaptive recovery turn tiny onset noise into arbitrary notes."""
    return float(max(0.10, min(0.18, 0.14 - 0.006 * (float(stars) - 6.0))))


def audio_activity_curve(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return robust 0..1 activity and a local-peak mask from Audio Feature V2.

    The final eight token dimensions are energy/dynamics features. V3 uses only
    song-relative energy, spectral flux and local-vs-context contrast, then
    normalizes them within the requested preview. This means silence recovery is
    allowed to become aggressive in musically active regions without forcing
    notes into genuinely quiet passages.
    """
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 8:
        raise ValueError("expected beat-synchronous audio tokens [T,D]")
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

    relative_energy = values[:, -4]
    flux = values[:, -3]
    contrast = values[:, -2]
    raw = 0.40 * relative_energy + 0.35 * flux + 0.25 * contrast

    low, high = np.quantile(raw, [0.10, 0.90]) if len(raw) >= 4 else (float(raw.min()), float(raw.max()))
    span = max(float(high - low), 1e-5)
    activity = np.clip((raw - float(low)) / span, 0.0, 1.0).astype(np.float32)

    peaks = np.zeros(len(activity), dtype=bool)
    radius = 3
    for index in range(len(activity)):
        start = max(0, index - radius)
        end = min(len(activity), index + radius + 1)
        local = activity[start:end]
        peaks[index] = bool(activity[index] >= 0.25 and activity[index] >= float(local.max()) - 1e-6)
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
) -> float:
    """Resolve the V3 gate threshold for one tick.

    Three mechanisms cooperate:
    1. absolute ceiling: prevents V1-style over-generation;
    2. rolling high quantile: treats a 0.30 peak as meaningful if neighbors are
       around 0.05 instead of demanding a globally calibrated 0.50;
    3. silence relaxation: after half a beat without a key-down, the gate lowers
       gradually, but mostly when the audio features say the region is active.
    """
    base = float(custom_threshold) if custom_threshold > 0 else _auto_base_threshold(stars)
    floor = _auto_floor_threshold(stars)
    window = history[-max(4, int(history_window)) :]

    if len(window) >= 4:
        relative = float(np.quantile(np.asarray(window, dtype=np.float32), relative_quantile))
        threshold = min(base, max(floor, relative))
    else:
        threshold = base

    activity_value = float(np.clip(activity, 0.0, 1.0))
    # Active audio may lower the gate by up to 0.04; quiet audio raises it.
    threshold += (0.5 - activity_value) * 0.08

    # 24 TPQ: 12 ticks = half a beat, 36 ticks = one and a half beats.
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
    """V3 free-running decoder with relative onset gating and silence recovery."""
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
            base = float(onset_threshold) if onset_threshold > 0 else _auto_base_threshold(float(stars[item].item()))
            floor = _auto_floor_threshold(float(stars[item].item()))
            recovery = False

            # If a musically active local peak appears after a full beat of
            # silence, allow a lower-confidence but still non-trivial onset.
            if (
                ticks_since[item] >= 24
                and bool(audio_peaks[position])
                and activity >= 0.45
                and onset_probability >= max(0.10, floor * 0.80)
                and onset_probability < threshold
            ):
                threshold = max(floor * 0.80, onset_probability - 1e-5)
                recovery = True

            # Last-resort anti-collapse guard: after two beats of silence in an
            # active region, accept a modest onset instead of feeding EMPTY back
            # forever. Quiet audio is intentionally exempt.
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
            position == 0 or position + 1 == length or (position + 1) % report_every == 0
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
            float(onset_threshold) if onset_threshold > 0 else _auto_base_threshold(float(stars[0].item()))
        ),
        "floor_onset_threshold": _auto_floor_threshold(float(stars[0].item())),
        "lane_threshold": float(lane_threshold),
        "ln_start_margin": float(ln_start_margin),
        "max_chord_limit": _auto_max_chord(float(stars[0].item())) if max_chord <= 0 else int(max_chord),
    }
    return result, diagnostics


def write_osu_v3(
    chart: np.ndarray,
    output_path: str | Path,
    *,
    audio_filename: str,
    bpm: float,
    offset_ms: float,
    stars: float,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    beat_length_ms = 60000.0 / float(bpm)
    tick_ms = beat_length_ms / 24.0
    objects: list[tuple[int, str]] = []
    active_start: list[int | None] = [None] * 4

    def time_for_tick(tick_index: int) -> int:
        return max(0, int(round(float(offset_ms) + tick_index * tick_ms)))

    for tick_index in range(len(chart)):
        for lane in range(4):
            state = int(chart[tick_index, lane])
            x = LANE_X[lane]
            if state == TAP:
                time_ms = time_for_tick(tick_index)
                objects.append((time_ms, f"{x},192,{time_ms},1,0,0:0:0:0:"))
            elif state == LN_START:
                active_start[lane] = tick_index
            elif state == LN_END:
                start_tick = active_start[lane]
                if start_tick is not None:
                    start_ms = time_for_tick(start_tick)
                    end_ms = max(start_ms + 1, time_for_tick(tick_index))
                    objects.append((start_ms, f"{x},192,{start_ms},128,0,{end_ms}:0:0:0:0:"))
                    active_start[lane] = None

    objects.sort(key=lambda item: (item[0], item[1]))
    object_lines = "\n".join(line for _, line in objects)
    text = f"""osu file format v14

[General]
AudioFilename: {audio_filename}
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
WidescreenStoryboard: 1

[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1

[Metadata]
Title:{title}
TitleUnicode:{title}
Artist:ORBIT-4K Input
ArtistUnicode:ORBIT-4K Input
Creator:ORBIT-4K V0
Version:AI {stars:.2f} Star Preview V3
Source:
Tags:ORBIT-4K AI generated adaptive decoder v3
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:5
CircleSize:4
OverallDifficulty:8
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events
//Break Periods

[TimingPoints]
{float(offset_ms):.6f},{beat_length_ms:.12f},4,2,0,100,1,0

[HitObjects]
{object_lines}
"""
    output_path.write_text(text, encoding="utf-8-sig")
    return output_path


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
    available_ticks = max(0, int(math.floor((duration_ms - float(offset_ms)) / tick_ms)) + 1)
    length = min(requested_ticks, available_ticks)
    if length <= 0:
        raise ValueError(
            f"audio has no usable ticks after offset {offset_ms} ms (duration={duration_ms:.1f} ms)"
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
        stats["keydowns"] / max(1e-6, length / int(config["grid"]["ticks_per_measure"]))
    )
    stats["blank_tick_ratio"] = float(1.0 - stats["onset_ticks"] / max(1, length))

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_target = output_dir / audio_path.name
    if audio_path.resolve() != audio_target.resolve():
        shutil.copy2(audio_path, audio_target)

    title = _safe_stem(audio_path.stem)
    filename = _safe_stem(f"{audio_path.stem} [ORBIT-4K {stars:.2f}sr {measures}m V3]") + ".osu"
    osu_path = write_osu_v3(
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
