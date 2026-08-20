from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from orbit4k.data_v1 import Orbit4KCellDataset
from orbit4k.preprocessing.audio_features import AUDIO_FEATURE_VERSION


def test_v1_dataset_groups_three_ticks_and_only_exposes_cell_start_ln_state():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "audio").mkdir()
        (root / "charts").mkdir()

        chart = np.zeros((7, 4), dtype=np.uint8)
        chart[2, 0] = 2  # LN_START at final micro of cell 0.
        chart[4, 0] = 3  # LN_END in cell 1.
        chart[5, 2] = 1
        np.savez_compressed(root / "charts" / "map.npz", chart=chart)

        frames = 1200
        np.savez_compressed(
            root / "audio" / "song.npz",
            log_mel=np.zeros((frames, 128), dtype=np.float16),
            mel_mean=np.zeros(128, dtype=np.float32),
            mel_std=np.ones(128, dtype=np.float32),
            log_energy=np.zeros((frames, 4), dtype=np.float16),
            energy_median=np.float32(0.0),
            energy_mad=np.float32(1.0),
            duration_ms=np.float32(6000.0),
            sample_rate=np.int32(44100),
            hop_length=np.int32(220),
            feature_version=np.int32(AUDIO_FEATURE_VERSION),
        )
        row = {
            "chart_id": "map",
            "split": "train",
            "chart_path": "charts/map.npz",
            "audio_path": "audio/song.npz",
            "total_ticks": len(chart),
            "offset_ms": 0.0,
            "beat_length_ms": 60000.0 / 180.0,
            "bpm": 180.0,
            "star_rating": 5.0,
        }
        (root / "index.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

        ds = Orbit4KCellDataset(
            root,
            "train",
            window_measures=1,
            random_crop=False,
            audio_cache_size=1,
        )
        sample = ds[0]

    assert sample["audio"].shape == (32, 3, 520)
    assert sample["target"].shape == (32, 3, 4)
    assert sample["chart_input"].shape == (32, 3, 4)
    assert sample["active_ln"].shape == (32, 4)
    assert sample["mask"].sum().item() == 3  # ceil(7/3)
    assert sample["micro_mask"].sum().item() == 7

    # Cell 0 begins unheld. LN starts at its last micro, so cell 1 begins held.
    assert int(sample["active_ln"][0, 0]) == 0
    assert int(sample["active_ln"][1, 0]) == 1
    # The end in cell 1 means cell 2 begins released.
    assert int(sample["active_ln"][2, 0]) == 0

    # Cell-level teacher forcing: cell 1 receives the whole target cell 0.
    assert np.array_equal(sample["chart_input"][1].numpy(), sample["target"][0].numpy())
