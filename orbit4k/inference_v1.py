from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .inference import _safe_stem, chart_statistics, write_osu
from .inference_v31 import constrained_tick_states_v31
from .model import BOS_STATE
from .model_v1 import MICRO_SLOTS, Orbit4KV1
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


def _model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[Orbit4KV1, dict, dict]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if checkpoint.get("architecture") not in {None, "v1_32x3"}:
        raise ValueError(f"checkpoint is not V1 32x3: {checkpoint.get('architecture')}")
    config = checkpoint["config"]
    model = Orbit4KV1(
        audio_token_dim=config["model"]["audio_token_dim"],
        audio_micro_dim=config["model"]["audio_micro_dim"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        audio_layers=config["model"]["audio_layers"],
        chart_layers=config["model"]["chart_layers"],
        micro_layers=config["model"]["micro_layers"],
        micro_dim_feedforward=config["model"]["micro_dim_feedforward"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        local_audio_scale_cells=config["model"]["local_audio_scale_cells"],
        max_cross_attention_bias=config["model"]["max_cross_attention_bias"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, config, checkpoint


def _cell_onset_threshold(stars: float) -> float:
    return float(max(0.34, min(0.52, 0.44 - 0.012 * (float(stars) - 6.0))))


def _micro_onset_threshold(stars: float) -> float:
    return float(max(0.30, min(0.50, 0.40 - 0.010 * (float(stars) - 6.0))))


def _advance_ln_state(
    state: torch.Tensor,
    active_now: torch.Tensor,
    ln_age: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    previous_active = active_now.bool()
    started = state.view(1, 4) == LN_START
    ended = state.view(1, 4) == LN_END
    continuing = previous_active & ~ended
    active_now = torch.where(started, 1, torch.where(ended, 0, active_now))
    ln_age = torch.where(
        started,
        torch.ones_like(ln_age),
        torch.where(continuing, ln_age + 1, torch.zeros_like(ln_age)),
    )
    return active_now, ln_age


def decode_v1_cell(
    lane_logits: torch.Tensor,
    micro_onset_logits: torch.Tensor,
    cell_onset_logit: torch.Tensor,
    active_now: torch.Tensor,
    ln_age: torch.Tensor,
    *,
    stars: float,
    temperature: float = 0.85,
    lane_threshold: float = 0.32,
    ln_start_margin: float = 0.30,
    final_cell: bool = False,
    cells_since_keydown: int = 0,
    audio_activity: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Resolve one 1/32 cell into three legal 1/96 micro-slot states."""
    if lane_logits.shape != (MICRO_SLOTS, 4, 4):
        raise ValueError("lane_logits must be [3,4,4]")
    if micro_onset_logits.shape != (MICRO_SLOTS,):
        raise ValueError("micro_onset_logits must be [3]")

    temp = max(float(temperature), 1e-4)
    cell_probability = float(torch.sigmoid(cell_onset_logit / temp).item())
    micro_probabilities = torch.sigmoid(micro_onset_logits / temp)
    cell_threshold = _cell_onset_threshold(stars)
    micro_threshold = _micro_onset_threshold(stars)

    # V1's cell head is much less sparse than V0's per-1/96 onset head. Silence
    # recovery is therefore deliberately weaker: only active audio can lower the
    # cell gate after half a beat (4 cells at 8 cells/quarter).
    activity = float(np.clip(audio_activity, 0.0, 1.0))
    if cells_since_keydown >= 4:
        relax = min(0.12, 0.03 * (cells_since_keydown - 3)) * activity
        cell_threshold = max(0.20, cell_threshold - relax)

    strong_micro = float(micro_probabilities.max().item())
    cell_gate = cell_probability >= cell_threshold or strong_micro >= 0.72
    selected_micro = [
        index
        for index in range(MICRO_SLOTS)
        if float(micro_probabilities[index].item()) >= micro_threshold
    ]
    fallback_micro = False
    if cell_gate and not selected_micro and strong_micro >= 0.12:
        selected_micro = [int(torch.argmax(micro_probabilities).item())]
        fallback_micro = True

    result = torch.zeros((MICRO_SLOTS, 4), dtype=torch.long, device=lane_logits.device)
    release_count = 0
    keydown_count = 0
    for micro in range(MICRO_SLOTS):
        probability = float(micro_probabilities[micro].item())
        if micro in selected_micro:
            effective_threshold = min(micro_threshold, max(0.01, probability - 1e-5))
        else:
            # >1 disables new onsets but still lets the legality code close LNs.
            effective_threshold = 1.01
        state, diag = constrained_tick_states_v31(
            lane_logits[micro],
            micro_onset_logits[micro],
            active_now[0],
            ln_age[0],
            stars=float(stars),
            temperature=temperature,
            onset_threshold=effective_threshold,
            lane_threshold=lane_threshold,
            ln_start_margin=ln_start_margin,
            max_chord=0,
            final_tick=bool(final_cell and micro == MICRO_SLOTS - 1),
        )
        result[micro] = state
        release_count += int(diag["release_count"])
        keydown_count += int(((state == TAP) | (state == LN_START)).sum().item())
        active_now, ln_age = _advance_ln_state(state, active_now, ln_age)

    return result, active_now, ln_age, {
        "cell_onset_probability": cell_probability,
        "cell_onset_threshold": cell_threshold,
        "micro_onset_max": strong_micro,
        "selected_micro_slots": len(selected_micro),
        "fallback_micro": int(fallback_micro),
        "keydowns": keydown_count,
        "releases": release_count,
    }


@torch.inference_mode()
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
    lane_threshold: float = 0.32,
    ln_start_margin: float = 0.30,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    if not 1 <= int(measures) <= 32:
        raise ValueError("V1 preview measures must be between 1 and 32")
    if bpm <= 0 or temperature <= 0:
        raise ValueError("BPM and temperature must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _emit(progress_callback, stage="loading_checkpoint", architecture="v1_32x3", device=str(device))
    model, config, checkpoint = _model_from_checkpoint(checkpoint_path, device)
    audio_config = AudioFeatureConfig(**config["audio"])
    _emit(progress_callback, stage="audio_features", architecture="v1_32x3")
    cache = extract_audio_cache(audio_path, audio_config)

    ticks_per_quarter = int(config["grid"]["ticks_per_quarter"])
    ticks_per_measure = int(config["grid"]["ticks_per_measure"])
    beat_length_ms = 60000.0 / float(bpm)
    tick_ms = beat_length_ms / ticks_per_quarter
    requested_ticks = int(measures) * ticks_per_measure
    duration_ms = float(cache["duration_ms"])
    available_ticks = max(0, int(math.floor((duration_ms - float(offset_ms)) / tick_ms)) + 1)
    length_ticks = min(requested_ticks, available_ticks)
    if length_ticks <= 0:
        raise ValueError("audio has no usable ticks after offset")

    cells = int(math.ceil(length_ticks / MICRO_SLOTS))
    padded_ticks = cells * MICRO_SLOTS
    tokens = beat_synchronous_audio_tokens(
        np.asarray(cache["log_mel"], dtype=np.float32),
        np.asarray(cache["mel_mean"], dtype=np.float32),
        np.asarray(cache["mel_std"], dtype=np.float32),
        np.asarray(cache["log_energy"], dtype=np.float32),
        energy_median=float(cache["energy_median"]),
        energy_mad=float(cache["energy_mad"]),
        duration_ms=duration_ms,
        start_tick=0,
        length=padded_ticks,
        offset_ms=float(offset_ms),
        beat_length_ms=beat_length_ms,
        ticks_per_quarter=ticks_per_quarter,
        hop_length=int(cache["hop_length"]),
        sample_rate=int(cache["sample_rate"]),
    )
    if length_ticks < padded_ticks:
        tokens[length_ticks:] = 0.0
    audio_cells = tokens.reshape(cells, MICRO_SLOTS, tokens.shape[-1])

    # Lightweight activity proxy from V0 token dynamics: relative energy, flux,
    # contrast are the final feature block's -4/-3/-2 dimensions.
    raw_activity = (
        0.40 * tokens[:, -4]
        + 0.35 * tokens[:, -3]
        + 0.25 * tokens[:, -2]
    )
    low, high = np.quantile(raw_activity, [0.10, 0.90]) if len(raw_activity) >= 4 else (raw_activity.min(), raw_activity.max())
    span = max(float(high - low), 1e-5)
    micro_activity = np.clip((raw_activity - float(low)) / span, 0.0, 1.0)
    cell_activity = micro_activity.reshape(cells, MICRO_SLOTS).max(axis=1)

    audio_tensor = torch.from_numpy(audio_cells.astype(np.float32)).unsqueeze(0).to(device)
    cell_tick = torch.arange(cells, device=device, dtype=torch.long).unsqueeze(0)
    bpm_tensor = torch.tensor([float(bpm)], device=device)
    stars_tensor = torch.tensor([float(stars)], device=device)
    mask = torch.ones((1, cells), device=device)
    memory, memory_padding = model.encode_audio(audio_tensor, cell_tick, mask)

    result = torch.zeros((1, cells, MICRO_SLOTS, 4), dtype=torch.long, device=device)
    chart_input = torch.full_like(result, BOS_STATE)
    active_ln = torch.zeros((1, cells, 4), dtype=torch.long, device=device)
    active_now = torch.zeros((1, 4), dtype=torch.long, device=device)
    ln_age = torch.zeros((1, 4), dtype=torch.long, device=device)
    cells_since_keydown = 0
    cell_probability_sum = 0.0
    fallback_micro_count = 0
    report_every = max(1, cells // 100)

    for position in range(cells):
        if position > 0:
            chart_input[:, position] = result[:, position - 1]
        active_ln[:, position] = active_now
        prefix = position + 1
        outputs = model.decode_from_memory(
            memory,
            chart_input[:, :prefix],
            active_ln[:, :prefix],
            cell_tick[:, :prefix],
            bpm_tensor,
            stars_tensor,
            target_mask=mask[:, :prefix],
            memory_padding_bool=memory_padding,
        )
        cell_state, active_now, ln_age, diag = decode_v1_cell(
            outputs["logits"][0, -1],
            outputs["micro_onset_logits"][0, -1],
            outputs["cell_onset_logits"][0, -1],
            active_now,
            ln_age,
            stars=float(stars),
            temperature=float(temperature),
            lane_threshold=float(lane_threshold),
            ln_start_margin=float(ln_start_margin),
            final_cell=position + 1 == cells,
            cells_since_keydown=cells_since_keydown,
            audio_activity=float(cell_activity[position]),
        )
        result[0, position] = cell_state
        has_keydown = bool(((cell_state == TAP) | (cell_state == LN_START)).any().item())
        cells_since_keydown = 0 if has_keydown else cells_since_keydown + 1
        cell_probability_sum += float(diag["cell_onset_probability"])
        fallback_micro_count += int(diag["fallback_micro"])

        if progress_callback is not None and (
            position == 0 or position + 1 == cells or (position + 1) % report_every == 0
        ):
            _emit(
                progress_callback,
                stage="generating_v1",
                architecture="v1_32x3",
                current=position + 1,
                total=cells,
                percent=round((position + 1) / max(1, cells) * 100.0, 2),
            )

    chart = result[0].reshape(cells * MICRO_SLOTS, 4)[:length_ticks].cpu().numpy().astype(np.uint8)
    repaired, repairs = validate_and_repair(chart)
    stats = chart_statistics(repaired)
    stats["repairs"] = len(repairs)
    stats["cells"] = cells
    stats["transformer_steps"] = cells
    stats["original_tick_steps_equivalent"] = length_ticks
    stats["step_reduction"] = float(length_ticks / max(1, cells))

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_target = output_dir / audio_path.name
    if audio_path.resolve() != audio_target.resolve():
        shutil.copy2(audio_path, audio_target)
    title = _safe_stem(audio_path.stem)
    filename = _safe_stem(
        f"{audio_path.stem} [ORBIT-4K V1 32x3 {stars:.2f}sr {measures}m]"
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
    text = osu_path.read_text(encoding="utf-8-sig")
    text = text.replace("Star Preview V2", "Star Preview V1 32x3")
    text = text.replace("constrained decoder v2", "hierarchical 32x3 decoder v1")
    osu_path.write_text(text, encoding="utf-8-sig")

    result_payload = {
        "stage": "complete",
        "architecture": "v1_32x3",
        "decoder": "hierarchical_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_score": float(checkpoint.get("score", float("nan"))),
        "device": str(device),
        "audio": str(audio_path.resolve()),
        "output": str(osu_path.resolve()),
        "bpm": float(bpm),
        "offset_ms": float(offset_ms),
        "stars": float(stars),
        "temperature": float(temperature),
        "generated_ticks": int(length_ticks),
        "generated_cells": int(cells),
        "stats": stats,
        "decoder_diagnostics": {
            "mean_cell_onset_probability": cell_probability_sum / max(1, cells),
            "fallback_micro_cells": fallback_micro_count,
            "cell_onset_threshold": _cell_onset_threshold(stars),
            "micro_onset_threshold": _micro_onset_threshold(stars),
        },
        "repair_examples": repairs[:20],
    }
    _emit(progress_callback, **result_payload)
    return result_payload
