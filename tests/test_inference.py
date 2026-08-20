from __future__ import annotations

import numpy as np

from orbit4k.inference import chart_statistics, write_osu


def test_chart_statistics_count_keydowns_and_chords():
    chart = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [3, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    stats = chart_statistics(chart)
    assert stats["taps"] == 2
    assert stats["ln"] == 1
    assert stats["keydowns"] == 3
    assert stats["chord_ticks"] == 1
    assert stats["max_chord"] == 2


def test_write_osu_contains_tap_and_ln(tmp_path):
    chart = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 3, 0, 0],
        ],
        dtype=np.uint8,
    )
    path = write_osu(
        chart,
        tmp_path / "preview.osu",
        audio_filename="song.mp3",
        bpm=180.0,
        offset_ms=0.0,
        stars=5.0,
        title="Preview",
    )
    text = path.read_text(encoding="utf-8-sig")
    assert "Mode: 3" in text
    assert "CircleSize:4" in text
    assert "AudioFilename: song.mp3" in text
    assert ",1,0,0:0:0:0:" in text
    assert ",128,0," in text
