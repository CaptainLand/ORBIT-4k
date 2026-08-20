from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orbit4k.inference import generate_preview


def _progress(payload: dict) -> None:
    # Machine-readable progress must stay ASCII-safe on Windows consoles.
    print("ORBIT4K_PROGRESS " + json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ORBIT-4K V0 preview beatmap")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--offset-ms", type=float, required=True)
    parser.add_argument("--stars", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--measures", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.0,
        help="0 = auto from target SR; otherwise onset-head gate in [0,1]",
    )
    parser.add_argument("--lane-threshold", type=float, default=0.32)
    parser.add_argument("--ln-start-margin", type=float, default=1.25)
    parser.add_argument(
        "--max-chord",
        type=int,
        default=0,
        help="0 = auto from target SR; otherwise 1..4",
    )
    args = parser.parse_args()

    # Keep stdout robust even when launched from a GBK/CP936 Windows shell.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    result = generate_preview(
        checkpoint_path=args.checkpoint,
        audio_path=args.audio,
        output_dir=args.output_dir,
        bpm=args.bpm,
        offset_ms=args.offset_ms,
        stars=args.stars,
        temperature=args.temperature,
        measures=args.measures,
        seed=args.seed,
        onset_threshold=args.onset_threshold,
        lane_threshold=args.lane_threshold,
        ln_start_margin=args.ln_start_margin,
        max_chord=args.max_chord,
        progress_callback=_progress,
    )
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
