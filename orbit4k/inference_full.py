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
from .inference_v3 import (
    _auto_base_threshold,
    _auto_floor_threshold,
    adaptive_onset_threshold,
    audio_activity_curve,
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


def plan_full_song_windows(
    total_ticks: int,
    window_ticks: int,
    context_ticks: int,
) -> list[dict[str, int]]:
    """Plan left-context / new-chart / right-audio-lookahead windows.

    Internal windows reserve ``context_ticks`` on both sides. The left side is a
    fixed chart prompt from the already stitched result; the right side is only
    encoded as audio memory so the bidirectional Audio Encoder can see what is
    coming next. The final window consumes the remaining song with no right
    lookahead reservation.
    """
    total_ticks = int(total_ticks)
    window_ticks = int(window_ticks)
    context_ticks = int(context_ticks)
    if total_ticks <= 0:
        return []
    if window_ticks <= 0:
        raise ValueError("window_ticks must be positive")
    if context_ticks < 0:
        raise ValueError("context_ticks must be non-negative")
    if 2 * context_ticks >= window_ticks:
        raise ValueError("window must be larger than twice the context")

    windows: list[dict[str, int]] = []
    filled_until = 0
    while filled_until < total_ticks:
        start = 0 if filled_until == 0 else max(0, filled_until - context_ticks)
        memory_end = min(total_ticks, start + window_ticks)
        is_final = memory_end >= total_ticks
        target_end = memory_end if is_final else memory_end - context_ticks
        if target_end <= filled_until:
            raise RuntimeError("full-song window planner made no forward progress")
        prefix_ticks = filled_until - start
        windows.append(
            {
                "start": start,
                "memory_end": memory_end,
                "target_end": target_end,
                "prefix_ticks": prefix_ticks,
                "new_start": filled_until,
                "new_ticks": target_end - filled_until,
                "is_final": int(is_final),
            }
        )
        filled_until = target_end
    return windows


def _state_before(chart: np.ndarray, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """Return active-LN flags and approximate LN ages immediately before tick."""
    active = np.zeros(4, dtype=np.int64)
    age = np.zeros(4, dtype=np.int64)
    for row in chart[: max(0, int(tick))]:
        for lane, state in enumerate(row):
            value = int(state)
            if value == LN_START:
                active[lane] = 1
                age[lane] = 1
            elif value == LN_END:
                active[lane] = 0
                age[lane] = 0
            elif active[lane]:
                age[lane] += 1
    return active, age


def _ticks_since_keydown_before(chart: np.ndarray, tick: int) -> int:
    count = 0
    for index in range(int(tick) - 1, -1, -1):
        row = chart[index]
        if np.any((row == TAP) | (row == LN_START)):
            break
        count += 1
    return count


@torch.inference_mode()
def adaptive_generate_prompted_window(
    model: Orbit4KV0,
    audio: torch.Tensor,
    tick: torch.Tensor,
    bpm: torch.Tensor,
    stars: torch.Tensor,
    *,
    target_length: int,
    prefix_chart: np.ndarray,
    initial_active: np.ndarray,
    initial_ln_age: np.ndarray,
    initial_ticks_since_keydown: int,
    audio_activity: np.ndarray,
    audio_peaks: np.ndarray,
    temperature: float = 0.85,
    onset_threshold: float = 0.0,
    lane_threshold: float = 0.32,
    ln_start_margin: float = 1.25,
    max_chord: int = 0,
    close_at_end: bool = False,
    progress_callback=None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Generate one full-song window while teacher-forcing its left overlap.

    Audio memory may be longer than ``target_length``. Those extra frames are
    right-side lookahead: the chart decoder never emits notes there in this
    window, but cross-attention can still use the future audio context.
    """
    model.eval()
    if audio.shape[0] != 1:
        raise ValueError("full-song V3 currently expects batch size 1")
    memory_length = int(audio.shape[1])
    target_length = int(target_length)
    if not 1 <= target_length <= memory_length:
        raise ValueError("target_length must be within the encoded audio window")

    prefix = np.asarray(prefix_chart, dtype=np.int64)
    if prefix.ndim != 2 or prefix.shape[1] != 4:
        raise ValueError("prefix_chart must have shape [T,4]")
    prefix_length = int(len(prefix))
    if prefix_length >= target_length:
        raise ValueError("prompt must leave at least one tick to generate")

    device = audio.device
    result = torch.zeros((1, target_length, 4), dtype=torch.long, device=device)
    if prefix_length:
        result[0, :prefix_length] = torch.from_numpy(prefix).to(device=device, dtype=torch.long)
    chart_input = torch.full_like(result, BOS_STATE)
    active_ln = torch.zeros_like(result)
    active_now = torch.from_numpy(np.asarray(initial_active, dtype=np.int64)).to(device).view(1, 4)
    ln_age = torch.from_numpy(np.asarray(initial_ln_age, dtype=np.int64)).to(device).view(1, 4)

    memory_mask = torch.ones((1, memory_length), dtype=torch.float32, device=device)
    memory, memory_padding = model.encode_audio(audio, tick, memory_mask)
    target_mask = torch.ones((1, target_length), dtype=torch.float32, device=device)

    history: list[float] = []
    ticks_since = max(0, int(initial_ticks_since_keydown))
    onset_probability_sum = 0.0
    threshold_sum = 0.0
    threshold_min = 1.0
    threshold_max = 0.0
    relative_gates = 0
    recovery_gates = 0
    fallback_count = 0
    releases = 0
    generated_count = 0
    max_silence = ticks_since
    report_every = max(1, target_length // 80)

    for position in range(target_length):
        if position > 0:
            chart_input[:, position] = result[:, position - 1]
        active_ln[:, position] = active_now
        prefix_len = position + 1
        outputs = model.decode_from_memory(
            memory,
            chart_input[:, :prefix_len],
            active_ln[:, :prefix_len],
            tick[:, :prefix_len],
            bpm,
            stars,
            target_mask=target_mask[:, :prefix_len],
            memory_padding_bool=memory_padding,
        )
        onset_logit = outputs["onset_logits"][0, -1]
        temp = max(float(temperature), 1e-4)
        onset_probability = float(torch.sigmoid(onset_logit / temp).item())
        history.append(onset_probability)
        if len(history) > 48:
            del history[:-48]

        if position < prefix_length:
            next_state = result[0, position]
        else:
            activity = float(audio_activity[position])
            threshold = adaptive_onset_threshold(
                history,
                stars=float(stars[0].item()),
                ticks_since_keydown=ticks_since,
                activity=activity,
                custom_threshold=float(onset_threshold),
            )
            base = (
                float(onset_threshold)
                if onset_threshold > 0
                else _auto_base_threshold(float(stars[0].item()))
            )
            floor = _auto_floor_threshold(float(stars[0].item()))
            recovery = False

            if (
                ticks_since >= 24
                and bool(audio_peaks[position])
                and activity >= 0.45
                and onset_probability >= max(0.10, floor * 0.80)
                and onset_probability < threshold
            ):
                threshold = max(floor * 0.80, onset_probability - 1e-5)
                recovery = True
            if (
                ticks_since >= 48
                and activity >= 0.35
                and onset_probability >= 0.08
                and onset_probability < threshold
            ):
                threshold = max(0.08, onset_probability - 1e-5)
                recovery = True

            next_state, diag = constrained_tick_states(
                outputs["logits"][0, -1],
                onset_logit,
                active_now[0],
                ln_age[0],
                stars=float(stars[0].item()),
                temperature=temperature,
                onset_threshold=threshold,
                lane_threshold=lane_threshold,
                ln_start_margin=ln_start_margin,
                max_chord=max_chord,
                final_tick=bool(close_at_end and position + 1 == target_length),
            )
            result[0, position] = next_state
            generated_count += 1
            onset_probability_sum += onset_probability
            threshold_sum += threshold
            threshold_min = min(threshold_min, threshold)
            threshold_max = max(threshold_max, threshold)
            gated = bool(diag["onset_gate"])
            relative_gates += int(gated and onset_probability < base and not recovery)
            recovery_gates += int(gated and recovery)
            fallback_count += int(bool(diag["lane_fallback"]))
            releases += int(diag["release_count"])

        next_state = result[0, position]
        has_keydown = bool(((next_state == TAP) | (next_state == LN_START)).any().item())
        if has_keydown:
            ticks_since = 0
        else:
            ticks_since += 1
            max_silence = max(max_silence, ticks_since)

        previous_active = active_now.bool()
        started = next_state.view(1, 4) == LN_START
        ended = next_state.view(1, 4) == LN_END
        continuing = previous_active & ~ended
        active_now = torch.where(started, 1, torch.where(ended, 0, active_now))
        ln_age = torch.where(
            started,
            torch.ones_like(ln_age),
            torch.where(continuing, ln_age + 1, torch.zeros_like(ln_age)),
        )

        if progress_callback is not None and (
            position == prefix_length
            or position + 1 == target_length
            or (position + 1) % report_every == 0
        ):
            progress_callback(position + 1, target_length)

    denominator = max(1, generated_count)
    diagnostics: dict[str, float | int] = {
        "decoder_version": 3,
        "prefix_ticks": prefix_length,
        "memory_ticks": memory_length,
        "target_ticks": target_length,
        "generated_ticks": generated_count,
        "mean_onset_probability": onset_probability_sum / denominator,
        "mean_effective_threshold": threshold_sum / denominator,
        "min_effective_threshold": threshold_min if generated_count else 0.0,
        "max_effective_threshold": threshold_max if generated_count else 0.0,
        "relative_peak_gates": relative_gates,
        "silence_recovery_gates": recovery_gates,
        "max_silence_ticks": max_silence,
        "lane_fallbacks": fallback_count,
        "releases": releases,
    }
    return result, diagnostics


def _write_full_song_osu(
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
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("Star Preview V2", "Star Full Song V3")
    text = text.replace("constrained decoder v2", "adaptive full-song decoder v3")
    path.write_text(text, encoding="utf-8-sig")
    return path


def generate_full_song(
    *,
    checkpoint_path: str | Path,
    audio_path: str | Path,
    output_dir: str | Path,
    bpm: float,
    offset_ms: float,
    stars: float,
    temperature: float = 0.85,
    window_measures: int = 4,
    context_measures: int = 1,
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
    if not 3 <= int(window_measures) <= 16:
        raise ValueError("full-song window_measures must be between 3 and 16")
    if not 1 <= int(context_measures) < int(window_measures) / 2:
        raise ValueError("context_measures must be at least 1 and less than half the window")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _emit(
        progress_callback,
        stage="loading_checkpoint",
        mode="full_song",
        device=str(device),
        decoder="adaptive_v3_full",
    )
    model, config, checkpoint = _model_from_checkpoint(checkpoint_path, device)
    ticks_per_measure = int(config["grid"]["ticks_per_measure"])
    ticks_per_quarter = int(config["grid"]["ticks_per_quarter"])
    trained_window_measures = int(config["window"]["measures"])
    if int(window_measures) > trained_window_measures:
        raise ValueError(
            f"full-song window ({window_measures} measures) exceeds trained window "
            f"({trained_window_measures} measures)"
        )

    _emit(progress_callback, stage="audio_features", mode="full_song")
    audio_config = AudioFeatureConfig(**config["audio"])
    cache = extract_audio_cache(audio_path, audio_config)
    beat_length_ms = 60000.0 / float(bpm)
    tick_ms = beat_length_ms / ticks_per_quarter
    duration_ms = float(cache["duration_ms"])
    total_ticks = max(
        0,
        int(math.floor((duration_ms - float(offset_ms)) / tick_ms)) + 1,
    )
    if total_ticks <= 0:
        raise ValueError(
            f"audio has no usable ticks after offset {offset_ms} ms "
            f"(duration={duration_ms:.1f} ms)"
        )

    # Beat tokens are lightweight enough to hold for a normal song and avoid
    # recomputing Mel-to-grid alignment for every overlapping chart window.
    full_tokens = beat_synchronous_audio_tokens(
        np.asarray(cache["log_mel"], dtype=np.float32),
        np.asarray(cache["mel_mean"], dtype=np.float32),
        np.asarray(cache["mel_std"], dtype=np.float32),
        np.asarray(cache["log_energy"], dtype=np.float32),
        energy_median=float(cache["energy_median"]),
        energy_mad=float(cache["energy_mad"]),
        duration_ms=duration_ms,
        start_tick=0,
        length=total_ticks,
        offset_ms=float(offset_ms),
        beat_length_ms=beat_length_ms,
        ticks_per_quarter=ticks_per_quarter,
        hop_length=int(cache["hop_length"]),
        sample_rate=int(cache["sample_rate"]),
    )
    full_activity, full_peaks = audio_activity_curve(full_tokens)

    window_ticks = int(window_measures) * ticks_per_measure
    context_ticks = int(context_measures) * ticks_per_measure
    windows = plan_full_song_windows(total_ticks, window_ticks, context_ticks)
    chart = np.zeros((total_ticks, 4), dtype=np.uint8)
    window_diagnostics: list[dict[str, float | int]] = []
    filled_until = 0

    _emit(
        progress_callback,
        stage="full_song_generating",
        mode="full_song",
        current=0,
        total=total_ticks,
        percent=0.0,
        window=0,
        total_windows=len(windows),
    )

    for window_index, plan in enumerate(windows, 1):
        start = plan["start"]
        memory_end = plan["memory_end"]
        target_end = plan["target_end"]
        prefix_ticks = plan["prefix_ticks"]
        target_length = target_end - start
        initial_active, initial_age = _state_before(chart, start)
        initial_ticks_since = _ticks_since_keydown_before(chart, start)
        prefix_chart = chart[start:filled_until].copy()

        audio_tensor = torch.from_numpy(full_tokens[start:memory_end]).unsqueeze(0).to(device)
        tick_tensor = torch.arange(
            start,
            memory_end,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        bpm_tensor = torch.tensor([float(bpm)], device=device, dtype=torch.float32)
        stars_tensor = torch.tensor([float(stars)], device=device, dtype=torch.float32)

        previous_filled = filled_until

        def window_progress(current: int, total: int) -> None:
            generated_in_window = max(0, int(current) - prefix_ticks)
            global_current = min(
                target_end,
                previous_filled + generated_in_window,
            )
            _emit(
                progress_callback,
                stage="full_song_generating",
                mode="full_song",
                current=global_current,
                total=total_ticks,
                percent=round(global_current / max(total_ticks, 1) * 100.0, 2),
                window=window_index,
                total_windows=len(windows),
                window_current=current,
                window_total=total,
            )

        generated_tensor, diagnostics = adaptive_generate_prompted_window(
            model,
            audio_tensor,
            tick_tensor,
            bpm_tensor,
            stars_tensor,
            target_length=target_length,
            prefix_chart=prefix_chart,
            initial_active=initial_active,
            initial_ln_age=initial_age,
            initial_ticks_since_keydown=initial_ticks_since,
            audio_activity=full_activity[start:memory_end],
            audio_peaks=full_peaks[start:memory_end],
            temperature=float(temperature),
            onset_threshold=float(onset_threshold),
            lane_threshold=float(lane_threshold),
            ln_start_margin=float(ln_start_margin),
            max_chord=int(max_chord),
            close_at_end=bool(plan["is_final"]),
            progress_callback=window_progress,
        )
        generated = generated_tensor[0].detach().cpu().numpy().astype(np.uint8)
        new_local_start = prefix_ticks
        chart[filled_until:target_end] = generated[new_local_start:target_length]
        filled_until = target_end
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "window": window_index,
                "start_tick": start,
                "target_end_tick": target_end,
                "memory_end_tick": memory_end,
            }
        )
        window_diagnostics.append(diagnostics)
        _emit(
            progress_callback,
            stage="full_song_generating",
            mode="full_song",
            current=filled_until,
            total=total_ticks,
            percent=round(filled_until / max(total_ticks, 1) * 100.0, 2),
            window=window_index,
            total_windows=len(windows),
        )

    repaired, repairs = validate_and_repair(chart)
    stats = chart_statistics(repaired)
    stats["repairs"] = len(repairs)
    stats["keydowns_per_measure"] = float(
        stats["keydowns"] / max(1e-6, total_ticks / ticks_per_measure)
    )
    stats["blank_tick_ratio"] = float(
        1.0 - stats["onset_ticks"] / max(1, total_ticks)
    )
    stats["generated_measures"] = float(total_ticks / ticks_per_measure)
    stats["duration_seconds"] = float(duration_ms / 1000.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_target = output_dir / audio_path.name
    if audio_path.resolve() != audio_target.resolve():
        shutil.copy2(audio_path, audio_target)

    title = _safe_stem(audio_path.stem)
    filename = _safe_stem(
        f"{audio_path.stem} [ORBIT-4K {stars:.2f}sr FULL V3]"
    ) + ".osu"
    osu_path = _write_full_song_osu(
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
        "mode": "full_song",
        "decoder": "adaptive_v3_full",
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
        "generated_ticks": int(total_ticks),
        "window_measures": int(window_measures),
        "context_measures": int(context_measures),
        "windows": len(windows),
        "stats": stats,
        "window_diagnostics": window_diagnostics,
        "repair_examples": repairs[:20],
    }
    _emit(progress_callback, **result)
    return result
