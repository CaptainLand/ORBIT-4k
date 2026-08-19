from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing.audio_features import (
    AUDIO_FEATURE_VERSION,
    beat_synchronous_audio_tokens,
)

BOS = 4


class Orbit4KDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        ticks_per_quarter: int = 24,
        ticks_per_measure: int = 96,
        window_measures: int = 16,
        stride_measures: int = 8,
        random_crop: bool = True,
        random_samples_per_chart: int = 4,
        audio_cache_size: int = 12,
    ) -> None:
        self.root = Path(root)
        self.rows = [
            json.loads(line)
            for line in (self.root / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["split"] == split
        ]
        self.ticks_per_quarter = ticks_per_quarter
        self.ticks_per_measure = ticks_per_measure
        self.window_ticks = window_measures * ticks_per_measure
        self.stride_ticks = stride_measures * ticks_per_measure
        self.random_crop = random_crop and split == "train"
        self.audio_cache_size = audio_cache_size
        self.audio_cache: OrderedDict[str, dict[str, np.ndarray | float | int]] = OrderedDict()
        self.samples: list[tuple[int, int]] = []
        for row_index, row in enumerate(self.rows):
            total = int(row["total_ticks"])
            if self.random_crop:
                self.samples.extend((row_index, -1) for _ in range(max(1, random_samples_per_chart)))
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
                    f"audio cache version {version} is incompatible with V0 feature version "
                    f"{AUDIO_FEATURE_VERSION}; rebuild the dataset with scripts/prepare_dataset.py"
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
            start = fixed_start
        length = min(self.window_ticks, len(chart) - start)
        target = np.zeros((self.window_ticks, 4), dtype=np.int64)
        target[:length] = chart[start : start + length]
        chart_input = np.full((self.window_ticks, 4), BOS, dtype=np.int64)
        if length > 1:
            chart_input[1:length] = target[: length - 1]
        if length < self.window_ticks:
            chart_input[length:] = 0
        mask = np.zeros(self.window_ticks, dtype=np.float32)
        mask[:length] = 1.0

        active = np.zeros(4, dtype=np.uint8)
        for state_row in chart[:start]:
            for lane, state in enumerate(state_row):
                if state == 2:
                    active[lane] = 1
                elif state == 3:
                    active[lane] = 0
        active_ln = np.zeros((self.window_ticks, 4), dtype=np.int64)
        for local_tick in range(length):
            active_ln[local_tick] = active
            for lane, state in enumerate(target[local_tick]):
                if state == 2:
                    active[lane] = 1
                elif state == 3:
                    active[lane] = 0

        cache = self._audio(row)
        audio = beat_synchronous_audio_tokens(
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
        if length < self.window_ticks:
            audio[length:] = 0.0
        ticks = start + np.arange(self.window_ticks, dtype=np.int64)
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
            "mask": torch.from_numpy(mask),
            "tick": torch.from_numpy(ticks),
            "bpm": torch.tensor(float(row["bpm"]), dtype=torch.float32),
            "stars": torch.tensor(float(stars), dtype=torch.float32),
            "chart_id": row["chart_id"],
        }
