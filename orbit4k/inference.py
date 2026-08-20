from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .model import BOS_STATE, Orbit4KV0
from .preprocessing.audio_features import (
    AudioFeatureConfig,
    beat_synchronous_audio_tokens,
    extract_audio_cache,
)
from .validator import EMPTY, LN_END, LN_START, TAP, validate_and_repair

ProgressCallback = Callable[[dict], None]
LANE_X = (64, 192, 320, 448)


def _emit(callback: ProgressCallback | None, **payload) -> None:
    if callback is not None:
        callback(payload)


def _model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Orbit4KV0, dict, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = Orbit4KV0(
        audio_token_dim=config["model"]["audio_token_dim"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        audio_layers=config["model"]["audio_layers"],
        chart_layers=config["model"]["chart_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        local_audio_scale_ticks=config["model"]["local_audio_scale_ticks"],
        max_cross_attention_bias=config["model"]["max_cross_attention_bias"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, config, checkpoint


def _safe_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_ ." else "_" for ch in value).strip()
    return cleaned[:120] or "orbit4k-preview"


def _auto_onset_threshold(stars: float) -> float:
    """A conservative key-down gate that relaxes gradually with target SR."""
    return float(max(0.42, min(0.58, 0.50 - 0.015 * (float(stars) - 6.0))))


def _auto_max_chord(stars: float) -> int:
    """Prevent low/mid-SR free-running collapse into constant hands/quads."""
    value = float(stars)
    if value < 4.5:
        return 1
    if value < 7.0:
        return 2
    if value < 9.0:
        return 3
    return 4


def constrained_tick_states(
    lane_logits: torch.Tensor,
    onset_logit: torch.Tensor,
    active_now: torch.Tensor,
    ln_age: torch.Tensor,
    *,
    stars: float,
    temperature: float = 0.85,
    onset_threshold: float = 0.0,
    lane_threshold: float = 0.32,
    ln_start_margin: float = 1.25,
    release_threshold: float = 0.50,
    min_ln_ticks: int = 2,
    max_ln_ticks: int = 192,
    max_chord: int = 0,
    final_tick: bool = False,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Decode one 4K tick with legality and anti-collapse constraints.

    The training model has two complementary heads: ``onset_head`` predicts
    whether a key-down should happen at this tick, while ``state_head`` predicts
    per-lane EMPTY/TAP/LN_START/LN_END. Older inference ignored the onset head
    and sampled all four lanes independently, which can amplify tiny rare-state
    probabilities into constant chords/LNs. This function makes the two heads
    cooperate and keeps the free-running state inside combinations seen during
    training.
    """
    if lane_logits.shape != (4, 4):
        raise ValueError(f"expected lane logits [4,4], got {tuple(lane_logits.shape)}")
    if active_now.shape != (4,) or ln_age.shape != (4,):
        raise ValueError("active_now and ln_age must be [4]")

    temp = max(float(temperature), 1e-4)
    gate_threshold = _auto_onset_threshold(stars) if onset_threshold <= 0 else float(onset_threshold)
    chord_cap = _auto_max_chord(stars) if max_chord <= 0 else max(1, min(4, int(max_chord)))
    next_state = torch.zeros(4, dtype=torch.long, device=lane_logits.device)

    # First resolve lanes that are already holding an LN. They may only remain
    # held (EMPTY interior state) or release (LN_END). TAP/LN_START are illegal.
    release_count = 0
    for lane in range(4):
        if int(active_now[lane].item()) == 0:
            continue
        age = int(ln_age[lane].item())
        force_end = final_tick or age >= int(max_ln_ticks)
        if force_end:
            next_state[lane] = LN_END
            release_count += 1
            continue
        if age < int(min_ln_ticks):
            continue
        pair = torch.stack([lane_logits[lane, EMPTY], lane_logits[lane, LN_END]]) / temp
        end_probability = torch.softmax(pair, dim=0)[1]
        if float(end_probability.item()) >= float(release_threshold):
            next_state[lane] = LN_END
            release_count += 1

    onset_probability = float(torch.sigmoid(onset_logit / temp).item())
    fallback = False
    selected: list[int] = []

    # Never start a new object on the final tick: it would be immediately
    # repaired/closed and is almost always a generation artifact.
    if not final_tick and onset_probability >= gate_threshold:
        inactive = [lane for lane in range(4) if int(active_now[lane].item()) == 0]
        if inactive:
            probabilities = torch.softmax(lane_logits / temp, dim=-1)
            key_probability = probabilities[:, TAP] + probabilities[:, LN_START]
            ranked = sorted(inactive, key=lambda lane: float(key_probability[lane]), reverse=True)
            top_lane = ranked[0]
            top_probability = float(key_probability[top_lane].item())

            # The onset head says a key-down belongs here. Always choose the
            # strongest free lane, even if the per-lane head is under-confident.
            selected = [top_lane]
            if top_probability < float(lane_threshold):
                fallback = True

            # Additional chord lanes require substantially stronger evidence
            # than the first lane. At modest onset confidence we keep singles;
            # strong onset confidence can use the SR-dependent chord cap.
            effective_cap = 1 if onset_probability < 0.72 else chord_cap
            extra_threshold = max(float(lane_threshold) + 0.14, top_probability * 0.72)
            for lane in ranked[1:]:
                if len(selected) >= effective_cap:
                    break
                if float(key_probability[lane].item()) >= extra_threshold:
                    selected.append(lane)

            for lane in selected:
                # LN_START must beat TAP by a real margin. This counteracts the
                # rare-class bias of the old independent sampler without banning
                # LNs when the model is genuinely confident.
                tap_logit = lane_logits[lane, TAP]
                start_logit = lane_logits[lane, LN_START] - float(ln_start_margin)
                next_state[lane] = LN_START if start_logit > tap_logit else TAP

    return next_state, {
        "onset_probability": onset_probability,
        "onset_gate": onset_probability >= gate_threshold,
        "onset_threshold": gate_threshold,
        "selected_keydowns": len(selected),
        "release_count": release_count,
        "lane_fallback": fallback,
        "max_chord": chord_cap,
    }


@torch.inference_mode()
def constrained_generate_window(
    model: Orbit4KV0,
    audio: torch.Tensor,
    tick: torch.Tensor,
    bpm: torch.Tensor,
    stars: torch.Tensor,
    *,
    temperature: float = 0.85,
    onset_threshold: float = 0.0,
    lane_threshold: float = 0.32,
    ln_start_margin: float = 1.25,
    max_chord: int = 0,
    progress_callback=None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Free-run with onset gating, lane legality, chord control, and LN state."""
    model.eval()
    batch, length, _ = audio.shape
    result = torch.zeros((batch, length, 4), dtype=torch.long, device=audio.device)
    chart_input = torch.full_like(result, BOS_STATE)
    active_ln = torch.zeros_like(result)
    active_now = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)
    ln_age = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)
    mask = torch.ones((batch, length), dtype=torch.float32, device=audio.device)
    memory, memory_padding = model.encode_audio(audio, tick, mask)

    onset_probability_sum = 0.0
    gated_ticks = 0
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
            next_state, diag = constrained_tick_states(
                lane_logits[item],
                onset_logits[item],
                active_now[item],
                ln_age[item],
                stars=float(stars[item].item()),
                temperature=temperature,
                onset_threshold=onset_threshold,
                lane_threshold=lane_threshold,
                ln_start_margin=ln_start_margin,
                max_chord=max_chord,
                final_tick=position + 1 == length,
            )
            next_batch[item] = next_state
            onset_probability_sum += float(diag["onset_probability"])
            gated_ticks += int(bool(diag["onset_gate"]))
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
    resolved_threshold = (
        _auto_onset_threshold(float(stars[0].item())) if onset_threshold <= 0 else float(onset_threshold)
    )
    resolved_chord = _auto_max_chord(float(stars[0].item())) if max_chord <= 0 else int(max_chord)
    diagnostics: dict[str, float | int] = {
        "decoder_version": 2,
        "mean_onset_probability": onset_probability_sum / denominator,
        "gated_onset_ticks": gated_ticks,
        "lane_fallbacks": fallback_count,
        "releases": releases,
        "onset_threshold": resolved_threshold,
        "lane_threshold": float(lane_threshold),
        "ln_start_margin": float(ln_start_margin),
        "max_chord_limit": resolved_chord,
    }
    return result, diagnostics


def chart_statistics(chart: np.ndarray) -> dict[str, int | float]:
    onset = (chart == TAP) | (chart == LN_START)
    per_tick = onset.sum(axis=1)
    taps = int((chart == TAP).sum())
    ln = int((chart == LN_START).sum())
    keydowns = taps + ln
    onset_ticks = int((per_tick > 0).sum())
    chord_ticks = int((per_tick >= 2).sum())
    return {
        "ticks": int(len(chart)),
        "taps": taps,
        "ln": ln,
        "keydowns": keydowns,
        "onset_ticks": onset_ticks,
        "ln_ratio": float(ln / max(1, keydowns)),
        "chord_ticks": chord_ticks,
        "chord_ratio": float(chord_ticks / max(1, onset_ticks)),
        "three_plus_ticks": int((per_tick >= 3).sum()),
        "four_chord_ticks": int((per_tick >= 4).sum()),
        "max_chord": int(per_tick.max()) if len(per_tick) else 0,
    }


def write_osu(
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

    def time_for_tick(tick: int) -> int:
        return max(0, int(round(float(offset_ms) + tick * tick_ms)))

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
                    objects.append(
                        (start_ms, f"{x},192,{start_ms},128,0,{end_ms}:0:0:0:0:")
                    )
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
Version:AI {stars:.2f} Star Preview V2
Source:
Tags:ORBIT-4K AI generated constrained decoder v2
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
    if not 0.0 <= float(onset_threshold) <= 1.0:
        raise ValueError("onset_threshold must be 0 (auto) or between 0 and 1")
    if not 0.0 < float(lane_threshold) < 1.0:
        raise ValueError("lane_threshold must be between 0 and 1")
    if float(ln_start_margin) < 0:
        raise ValueError("ln_start_margin must be non-negative")
    if not 0 <= int(max_chord) <= 4:
        raise ValueError("max_chord must be 0 (auto) or 1..4")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _emit(progress_callback, stage="loading_checkpoint", device=str(device), decoder="constrained_v2")
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
    generated_tensor, decoder_diagnostics = constrained_generate_window(
        model,
        audio_tensor,
        tick_tensor,
        bpm_tensor,
        stars_tensor,
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

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_target = output_dir / audio_path.name
    if audio_path.resolve() != audio_target.resolve():
        shutil.copy2(audio_path, audio_target)

    title = _safe_stem(audio_path.stem)
    filename = _safe_stem(
        f"{audio_path.stem} [ORBIT-4K {stars:.2f}sr {measures}m V2]"
    ) + ".osu"
    osu_path = write_osu(
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
        "decoder": "constrained_v2",
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
