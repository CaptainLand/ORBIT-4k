from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import BOS_STATE
from .model_v1 import MICRO_SLOTS
from .preprocessing.audio_features import AUDIO_FEATURE_VERSION, beat_synchronous_audio_tokens


class Orbit4KCellDataset(Dataset):
    """Reuse V0 1/96 caches while presenting 32 cells x 3 micro slots to V1."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        ticks_per_quarter: int = 24,
        ticks_per_measure: int = 96,
        micro_ticks_per_cell: int = 3,
        window_measures: int = 16,
        stride_measures: int = 8,
        random_crop: bool = True,
        random_samples_per_chart: int = 4,
        audio_cache_size: int = 4,
    ) -> None:
        self.root = Path(root)
        if micro_ticks_per_cell != MICRO_SLOTS:
            raise ValueError(f"V1 requires exactly {MICRO_SLOTS} 1/96 ticks per cell")
        if ticks_per_quarter % MICRO_SLOTS or ticks_per_measure % MICRO_SLOTS:
            raise ValueError("grid must divide cleanly into 3-slot cells")

        index_path = self.root / "index.jsonl"
        self.rows = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] == split:
                self.rows.append(row)

        self.ticks_per_quarter = int(ticks_per_quarter)
        self.ticks_per_measure = int(ticks_per_measure)
        self.cells_per_measure = self.ticks_per_measure // MICRO_SLOTS
        self.window_ticks = int(window_measures) * self.ticks_per_measure
        self.window_cells = self.window_ticks // MICRO_SLOTS
        self.stride_ticks = int(stride_measures) * self.ticks_per_measure
        self.random_crop = bool(random_crop and split == "train")
        self.audio_cache_size = max(1, int(audio_cache_size))
        self.audio_cache: OrderedDict[str, dict[str, np.ndarray | float | int]] = OrderedDict()

        self.samples: list[tuple[int, int]] = []
        for row_index, row in enumerate(self.rows):
            total = int(row["total_ticks"])
            if self.random_crop:
                self.samples.extend((row_index, -1) for _ in range(max(1, int(random_samples_per_chart))))
            else:
                starts = list(range(0, max(1, total - self.window_ticks + 1), self.stride_ticks))
                final = max(0, total - self.window_ticks)
                final = (final // self.ticks_per_measure) * self.ticks_per_measure
                if final not in starts:
                    starts.append(final)
                self.samples.extend((row_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.samples)

    def _audio(self, row: dict) -> dict[str, np.ndarray | float | int]:
        key = row["audio_path"]
        if key in self.audio_cache:
            value = self.audio_cache.pop(key)
            self.audio_cache[key] = value
            return value
        with np.load(self.root / key) as data:
            version = int(data.get("feature_version", 1))
            if version != AUDIO_FEATURE_VERSION:
                raise RuntimeError(
                    f"audio cache version {version} is incompatible with feature version "
                    f"{AUDIO_FEATURE_VERSION}; rebuild with scripts/prepare_dataset.py"
                )
            value = {
                "log_mel": data["log_mel"].astype(np.float32),
                "mel_mean": data["mel_mean"].astype(np.float32),
                "mel_std": data["mel_std"].astype(np.float32),
                "log_energy": data["log_energy"].astype(np.float32),
                "energy_median": float(data["energy_median"]),
                "energy_mad": float(data["energy_mad"]),
                "duration_ms": float(data["duration_ms"]),
                "sample_rate": int(data["sample_rate"]),
                "hop_length": int(data["hop_length"]),
                "feature_version": version,
            }
        self.audio_cache[key] = value
        while len(self.audio_cache) > self.audio_cache_size:
            self.audio_cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row_index, fixed_start = self.samples[index]
        row = self.rows[row_index]
        with np.load(self.root / row["chart_path"]) as data:
            chart = data["chart"].astype(np.int64)

        max_start = max(0, len(chart) - self.window_ticks)
        if fixed_start < 0:
            measure_max = max_start // self.ticks_per_measure
            start = random.randint(0, measure_max) * self.ticks_per_measure if measure_max else 0
        else:
            start = int(fixed_start)
        if start % self.ticks_per_measure:
            raise RuntimeError("V1 crops must start on a measure boundary")

        length_ticks = min(self.window_ticks, max(0, len(chart) - start))
        valid_cells = int(math.ceil(length_ticks / MICRO_SLOTS)) if length_ticks else 0

        target_ticks = np.zeros((self.window_ticks, 4), dtype=np.int64)
        if length_ticks:
            target_ticks[:length_ticks] = chart[start : start + length_ticks]
        target = target_ticks.reshape(self.window_cells, MICRO_SLOTS, 4)

        # Cell-level teacher forcing: current cell receives the complete previous
        # cell. This is what reduces autoregressive generation from 96 to 32
        # Transformer steps per measure without losing 1/96 outputs.
        chart_input = np.full((self.window_cells, MICRO_SLOTS, 4), BOS_STATE, dtype=np.int64)
        if valid_cells > 1:
            chart_input[1:valid_cells] = target[: valid_cells - 1]
        if valid_cells < self.window_cells:
            chart_input[valid_cells:] = 0

        cell_mask = np.zeros(self.window_cells, dtype=np.float32)
        cell_mask[:valid_cells] = 1.0
        micro_mask = np.zeros((self.window_cells, MICRO_SLOTS), dtype=np.float32)
        if length_ticks:
            micro_mask.reshape(-1)[:length_ticks] = 1.0

        # Only expose LN state at the *start* of a cell. Supplying active state
        # for each micro slot would leak whether earlier target slots in the same
        # cell started/ended a hold.
        active = np.zeros(4, dtype=np.uint8)
        for state_row in chart[:start]:
            for lane, state in enumerate(state_row):
                if state == 2:
                    active[lane] = 1
                elif state == 3:
                    active[lane] = 0
        active_ln = np.zeros((self.window_cells, 4), dtype=np.int64)
        for cell in range(valid_cells):
            active_ln[cell] = active
            for micro in range(MICRO_SLOTS):
                local_tick = cell * MICRO_SLOTS + micro
                if local_tick >= length_ticks:
                    break
                for lane, state in enumerate(target[cell, micro]):
                    if state == 2:
                        active[lane] = 1
                    elif state == 3:
                        active[lane] = 0

        cache = self._audio(row)
        audio_ticks = beat_synchronous_audio_tokens(
            cache["log_mel"],
            cache["mel_mean"],
            cache["mel_std"],
            cache["log_energy"],
            energy_median=float(cache["energy_median"]),
            energy_mad=float(cache["energy_mad"]),
            duration_ms=float(cache["duration_ms"]),
            start_tick=start,
            length=self.window_ticks,
            offset_ms=float(row["offset_ms"]),
            beat_length_ms=float(row["beat_length_ms"]),
            ticks_per_quarter=self.ticks_per_quarter,
            hop_length=int(cache["hop_length"]),
            sample_rate=int(cache["sample_rate"]),
        )
        if length_ticks < self.window_ticks:
            audio_ticks[length_ticks:] = 0.0
        audio = audio_ticks.reshape(self.window_cells, MICRO_SLOTS, audio_ticks.shape[-1])

        cell_start = start // MICRO_SLOTS
        cell_tick = cell_start + np.arange(self.window_cells, dtype=np.int64)
        stars = row.get("star_rating")
        if stars is None:
            raise RuntimeError(
                f"missing star rating for {row['chart_id']}; install rosu-pp-py and rebuild the dataset"
            )

        return {
            "audio": torch.from_numpy(audio),
            "chart_input": torch.from_numpy(chart_input),
            "target": torch.from_numpy(target),
            "active_ln": torch.from_numpy(active_ln),
            "mask": torch.from_numpy(cell_mask),
            "micro_mask": torch.from_numpy(micro_mask),
            "cell_tick": torch.from_numpy(cell_tick),
            "bpm": torch.tensor(float(row["bpm"]), dtype=torch.float32),
            "stars": torch.tensor(float(stars), dtype=torch.float32),
            "chart_id": row["chart_id"],
        }
