from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orbit4k.inference_v31 import generate_full_song, generate_preview


def _progress(payload: dict) -> None:
    # Machine-readable progress must stay ASCII-safe on Windows consoles.
    print("ORBIT4K_PROGRESS " + json.dumps(payload, ensure_ascii=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ORBIT-4K V0 preview or full-song beatmaps with adaptive V3.1 decoding"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--offset-ms", type=float, required=True)
    parser.add_argument("--stars", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--measures", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--full-song", action="store_true")
    parser.add_argument(
        "--window-measures",
        type=int,
        default=4,
        help="full-song audio window size in measures (3..16)",
    )
    parser.add_argument(
        "--context-measures",
        type=int,
        default=1,
        help="full-song left chart context and right audio lookahead in measures",
    )
    parser.add_argument(
        "--onset-threshold",
        type=float,
        default=0.0,
        help="0 = V3 adaptive threshold; otherwise absolute onset-head ceiling in [0,1]",
    )
    parser.add_argument("--lane-threshold", type=float, default=0.32)
    parser.add_argument(
        "--ln-start-margin",
        type=float,
        default=0.30,
        help="mild TAP-over-LN safety bias; V3.1 default 0.30 (V3 used 1.25)",
    )
    parser.add_argument(
        "--max-chord",
        type=int,
        default=0,
        help="0 = V3.1 soft evidence with physical 4-key cap; otherwise explicit 1..4 cap",
    )
    args = parser.parse_args()

    # Keep stdout robust even when launched from a GBK/CP936 Windows shell.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    common = dict(
        checkpoint_path=args.checkpoint,
        audio_path=args.audio,
        output_dir=args.output_dir,
        bpm=args.bpm,
        offset_ms=args.offset_ms,
        stars=args.stars,
        temperature=args.temperature,
        seed=args.seed,
        onset_threshold=args.onset_threshold,
        lane_threshold=args.lane_threshold,
        ln_start_margin=args.ln_start_margin,
        max_chord=args.max_chord,
        progress_callback=_progress,
    )
    if args.full_song:
        result = generate_full_song(
            **common,
            window_measures=args.window_measures,
            context_measures=args.context_measures,
        )
    else:
        result = generate_preview(
            **common,
            measures=args.measures,
        )
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
