from __future__ import annotations

import json

from orbit4k.preprocessing.build_dataset import _progress_line


def test_progress_line_is_ascii_safe_and_round_trips_unicode():
    payload = {
        "stage": "processing",
        "current": 286,
        "total": 516,
        "title": "初音ミク — déjà vu 🟣",
        "version": "Äwesome / 1.25x",
    }

    line = _progress_line(payload)

    # The subprocess transport must survive cp936/GBK or any ASCII-compatible
    # Windows console even when beatmap metadata contains arbitrary Unicode.
    line.encode("ascii")
    assert line.startswith("ORBIT4K_PROGRESS ")

    decoded = json.loads(line[len("ORBIT4K_PROGRESS "):])
    assert decoded == payload
