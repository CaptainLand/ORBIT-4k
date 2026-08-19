from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import yaml

from .audio_features import AUDIO_FEATURE_VERSION, AudioFeatureConfig, extract_audio_cache
from .osu_parser import scan_4k_beatmaps, stable_audio_id


def _load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _split_for_audio(audio_id: str, ratios: dict[str, float], seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{audio_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    train_end = float(ratios["train"])
    val_end = train_end + float(ratios["validation"])
    if value < train_end:
        return "train"
    if value < val_end:
        return "validation"
    return "test"


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"unsafe zip member: {member.filename}")
    archive.extractall(destination)


def _materialize_input(source: Path, temp: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            _safe_extract(archive, temp)
        return temp
    raise ValueError("input must be a Songs directory, beatmap-set directory, or .zip")


def build_dataset(source: str | Path, output: str | Path, config_path: str | Path) -> dict:
    source = Path(source)
    output = Path(output)
    config = _load_config(config_path)
    grid = config["grid"]
    audio_cfg = AudioFeatureConfig(**config["audio"])
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    chart_dir = output / "charts"
    audio_dir.mkdir(exist_ok=True)
    chart_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="orbit4k_") as tmp:
        scan_root = _materialize_input(source, Path(tmp))
        accepted, rejected = scan_4k_beatmaps(
            scan_root,
            ticks_per_quarter=int(grid["ticks_per_quarter"]),
            constant_bpm_only=bool(grid["constant_bpm_only"]),
            max_quantization_error_ms=float(grid["max_quantization_error_ms"]),
            calculate_stars=True,
        )

        audio_cache: dict[str, dict] = {}
        audio_ids_by_path: dict[Path, str] = {}
        index_rows: list[dict] = []
        for beatmap in accepted:
            resolved_audio = beatmap.audio_path.resolve()
            audio_id = audio_ids_by_path.get(resolved_audio)
            if audio_id is None:
                audio_id = stable_audio_id(beatmap.audio_path)
                audio_ids_by_path[resolved_audio] = audio_id

            if audio_id not in audio_cache:
                feature_path = audio_dir / f"{audio_id}.npz"
                rebuild_audio = not feature_path.exists()
                if feature_path.exists():
                    with np.load(feature_path) as data:
                        existing_version = int(data.get("feature_version", 1))
                    rebuild_audio = existing_version != AUDIO_FEATURE_VERSION

                if rebuild_audio:
                    cache = extract_audio_cache(beatmap.audio_path, audio_cfg)
                    np.savez_compressed(feature_path, **cache)
                    duration_ms = float(cache["duration_ms"])
                    feature_version = int(cache["feature_version"])
                else:
                    with np.load(feature_path) as data:
                        duration_ms = float(data["duration_ms"])
                        feature_version = int(data["feature_version"])
                audio_cache[audio_id] = {
                    "path": str(feature_path.relative_to(output)),
                    "duration_ms": duration_ms,
                    "feature_version": feature_version,
                }

            chart_id = hashlib.sha1(beatmap.path.read_bytes()).hexdigest()[:16]
            chart_path = chart_dir / f"{chart_id}.npz"
            np.savez_compressed(chart_path, chart=beatmap.chart)
            split = _split_for_audio(audio_id, config["splits"], int(config["seed"]))
            row = {
                "chart_id": chart_id,
                "audio_id": audio_id,
                "audio_path": audio_cache[audio_id]["path"],
                "audio_feature_version": audio_cache[audio_id]["feature_version"],
                "chart_path": str(chart_path.relative_to(output)),
                "split": split,
                "title": beatmap.title,
                "artist": beatmap.artist,
                "version": beatmap.version,
                "creator": beatmap.creator,
                "bpm": beatmap.bpm,
                "offset_ms": beatmap.offset_ms,
                "beat_length_ms": beatmap.beat_length_ms,
                "star_rating": beatmap.star_rating,
                "object_count": beatmap.object_count,
                "total_ticks": beatmap.total_ticks,
                "p95_quantization_error_ms": beatmap.p95_quantization_error_ms,
            }
            index_rows.append(row)

    index_path = output / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    (output / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_versions = sorted({int(row["audio_feature_version"]) for row in index_rows})
    summary = {
        "accepted_charts": len(index_rows),
        "rejected_charts": len(rejected),
        "unique_audio": len({row["audio_id"] for row in index_rows}),
        "audio_feature_versions": feature_versions,
        "missing_star_ratings": sum(row["star_rating"] is None for row in index_rows),
        "splits": {
            name: sum(row["split"] == name for row in index_rows)
            for name in ("train", "validation", "test")
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ORBIT-4K V0 training data")
    parser.add_argument(
        "input",
        type=Path,
        help="osu! Songs directory, beatmap-set directory, or .zip",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/v0"))
    parser.add_argument("--config", type=Path, default=Path("configs/v0.yaml"))
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.input, args.output, args.config), indent=2))


if __name__ == "__main__":
    main()
