from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .model import Orbit4KV0
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


def chart_statistics(chart: np.ndarray) -> dict[str, int | float]:
    onset = (chart == TAP) | (chart == LN_START)
    per_tick = onset.sum(axis=1)
    taps = int((chart == TAP).sum())
    ln = int((chart == LN_START).sum())
    return {
        "ticks": int(len(chart)),
        "taps": taps,
        "ln": ln,
        "keydowns": taps + ln,
        "chord_ticks": int((per_tick >= 2).sum()),
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

    for tick in range(len(chart)):
        for lane in range(4):
            state = int(chart[tick, lane])
            x = LANE_X[lane]
            if state == TAP:
                time_ms = time_for_tick(tick)
                objects.append((time_ms, f"{x},192,{time_ms},1,0,0:0:0:0:"))
            elif state == LN_START:
                active_start[lane] = tick
            elif state == LN_END:
                start_tick = active_start[lane]
                if start_tick is not None:
                    start_ms = time_for_tick(start_tick)
                    end_ms = max(start_ms + 1, time_for_tick(tick))
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
Version:AI {stars:.2f} Star Preview
Source:
Tags:ORBIT-4K AI generated preview
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

    _emit(progress_callback, stage="loading_checkpoint", device=str(device))
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
    generated = model.generate_window(
        audio_tensor,
        tick_tensor,
        bpm_tensor,
        stars_tensor,
        temperature=float(temperature),
        progress_callback=model_progress,
    )[0].detach().cpu().numpy().astype(np.uint8)

    repaired, repairs = validate_and_repair(generated)
    stats = chart_statistics(repaired)
    stats["repairs"] = len(repairs)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_target = output_dir / audio_path.name
    if audio_path.resolve() != audio_target.resolve():
        shutil.copy2(audio_path, audio_target)

    title = _safe_stem(audio_path.stem)
    filename = _safe_stem(
        f"{audio_path.stem} [ORBIT-4K {stars:.2f}sr {measures}m]"
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
        "repair_examples": repairs[:20],
    }
    _emit(progress_callback, **result)
    return result
