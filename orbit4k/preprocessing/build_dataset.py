from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .audio_features import AUDIO_FEATURE_VERSION, AudioFeatureConfig, extract_audio_cache
from .osu_parser import scan_4k_beatmaps, stable_audio_id

ProgressCallback = Callable[[dict], None]


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


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict | list) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _existing_audio_metadata(feature_path: Path) -> dict | None:
    if not feature_path.exists():
        return None
    try:
        with np.load(feature_path) as data:
            version = int(data.get("feature_version", 1))
            if version != AUDIO_FEATURE_VERSION:
                return None
            required = {"log_mel", "mel_mean", "mel_std", "log_energy", "duration_ms", "hop_length"}
            if not required.issubset(set(data.files)):
                return None
            return {
                "duration_ms": float(data["duration_ms"]),
                "feature_version": version,
            }
    except Exception:
        return None


def _reject_row(beatmap, reason: str, error: str | None = None) -> dict:
    row = {
        "path": str(beatmap.path),
        "title": beatmap.title,
        "version": beatmap.version,
        "reason": reason,
    }
    if error:
        row["error"] = error[:1200]
    return row


def build_dataset(
    source: str | Path,
    output: str | Path,
    config_path: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    source = Path(source)
    output = Path(output)
    config = _load_config(config_path)
    grid = config["grid"]
    audio_cfg = AudioFeatureConfig(**config["audio"])

    def emit(**payload) -> None:
        if progress_callback is not None:
            progress_callback(payload)

    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "audio"
    chart_dir = output / "charts"
    audio_dir.mkdir(exist_ok=True)
    chart_dir.mkdir(exist_ok=True)

    state_path = output / "dataset_state.json"
    partial_index_path = output / "index.partial.jsonl"
    partial_rejected_path = output / "rejected.partial.json"
    index_rows: list[dict] = []
    rejected_rows: list[dict] = []
    total = 0
    current = 0
    unique_audio = 0
    failed_audio_paths: dict[Path, str] = {}

    def checkpoint(*, stage: str, status: str = "building", last_error: str | None = None) -> None:
        _atomic_write_text(
            partial_index_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows),
        )
        _atomic_write_json(partial_rejected_path, rejected_rows)
        state = {
            "status": status,
            "stage": stage,
            "source": str(source),
            "output": str(output),
            "total_scanned_accepted": total,
            "processed": current,
            "accepted_processed": len(index_rows),
            "rejected_total": len(rejected_rows),
            "unique_audio": unique_audio,
            "failed_unique_audio": len(failed_audio_paths),
            "audio_feature_version": AUDIO_FEATURE_VERSION,
        }
        if last_error:
            state["last_error"] = last_error[:2000]
        _atomic_write_json(state_path, state)

    emit(stage="starting", source=str(source), output=str(output))
    _atomic_write_json(
        state_path,
        {
            "status": "building",
            "stage": "starting",
            "source": str(source),
            "output": str(output),
            "audio_feature_version": AUDIO_FEATURE_VERSION,
        },
    )

    try:
        with tempfile.TemporaryDirectory(prefix="orbit4k_") as tmp:
            scan_root = _materialize_input(source, Path(tmp))
            emit(stage="scanning", root=str(scan_root))
            accepted, rejected = scan_4k_beatmaps(
                scan_root,
                ticks_per_quarter=int(grid["ticks_per_quarter"]),
                constant_bpm_only=bool(grid["constant_bpm_only"]),
                max_quantization_error_ms=float(grid["max_quantization_error_ms"]),
                calculate_stars=True,
            )
            rejected_rows = list(rejected)
            total = len(accepted)
            emit(
                stage="scan_complete",
                accepted=total,
                rejected=len(rejected_rows),
                total=total,
            )
            checkpoint(stage="scan_complete")

            audio_cache: dict[str, dict] = {}
            audio_ids_by_path: dict[Path, str] = {}
            report_every = max(1, total // 200) if total else 1

            for current, beatmap in enumerate(accepted, 1):
                resolved_audio = beatmap.audio_path.resolve()

                if resolved_audio in failed_audio_paths:
                    rejected_rows.append(
                        _reject_row(
                            beatmap,
                            "audio_feature_error",
                            failed_audio_paths[resolved_audio],
                        )
                    )
                    if current == total or current % report_every == 0:
                        checkpoint(stage="processing", last_error=failed_audio_paths[resolved_audio])
                        emit(
                            stage="processing",
                            current=current,
                            total=total,
                            percent=round(current / max(total, 1) * 100.0, 2),
                            unique_audio=unique_audio,
                            accepted=len(index_rows),
                            rejected=len(rejected_rows),
                            failed_unique_audio=len(failed_audio_paths),
                            title=beatmap.title,
                            version=beatmap.version,
                            last_error=failed_audio_paths[resolved_audio],
                        )
                    continue

                try:
                    audio_id = audio_ids_by_path.get(resolved_audio)
                    if audio_id is None:
                        audio_id = stable_audio_id(beatmap.audio_path)
                        audio_ids_by_path[resolved_audio] = audio_id

                    if audio_id not in audio_cache:
                        feature_path = audio_dir / f"{audio_id}.npz"
                        metadata = _existing_audio_metadata(feature_path)
                        if metadata is None:
                            cache = extract_audio_cache(beatmap.audio_path, audio_cfg)
                            _atomic_savez(feature_path, **cache)
                            metadata = {
                                "duration_ms": float(cache["duration_ms"]),
                                "feature_version": int(cache["feature_version"]),
                            }
                        audio_cache[audio_id] = {
                            "path": str(feature_path.relative_to(output)),
                            **metadata,
                        }
                        unique_audio = len(audio_cache)

                    chart_id = hashlib.sha1(beatmap.path.read_bytes()).hexdigest()[:16]
                    chart_path = chart_dir / f"{chart_id}.npz"
                    if not chart_path.exists():
                        _atomic_savez(chart_path, chart=beatmap.chart)

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
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    failed_audio_paths[resolved_audio] = message
                    rejected_rows.append(_reject_row(beatmap, "audio_feature_error", message))
                    # Release large temporary tensors/arrays after a failed decode or STFT
                    # before moving on to the next independent audio file.
                    gc.collect()

                if current == 1 or current == total or current % report_every == 0:
                    last_error = failed_audio_paths.get(resolved_audio)
                    checkpoint(stage="processing", last_error=last_error)
                    emit(
                        stage="processing",
                        current=current,
                        total=total,
                        percent=round(current / max(total, 1) * 100.0, 2),
                        unique_audio=unique_audio,
                        accepted=len(index_rows),
                        rejected=len(rejected_rows),
                        failed_unique_audio=len(failed_audio_paths),
                        title=beatmap.title,
                        version=beatmap.version,
                        last_error=last_error,
                    )

        index_path = output / "index.jsonl"
        _atomic_write_text(
            index_path,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows),
        )
        _atomic_write_json(output / "rejected.json", rejected_rows)
        feature_versions = sorted({int(row["audio_feature_version"]) for row in index_rows})
        summary = {
            "accepted_charts": len(index_rows),
            "rejected_charts": len(rejected_rows),
            "unique_audio": len({row["audio_id"] for row in index_rows}),
            "failed_unique_audio": len(failed_audio_paths),
            "audio_feature_versions": feature_versions,
            "missing_star_ratings": sum(row["star_rating"] is None for row in index_rows),
            "splits": {
                name: sum(row["split"] == name for row in index_rows)
                for name in ("train", "validation", "test")
            },
        }
        _atomic_write_json(output / "summary.json", summary)
        _atomic_write_json(
            state_path,
            {
                "status": "complete",
                "stage": "complete",
                "source": str(source),
                "output": str(output),
                "processed": total,
                "accepted_processed": len(index_rows),
                "rejected_total": len(rejected_rows),
                "unique_audio": summary["unique_audio"],
                "failed_unique_audio": len(failed_audio_paths),
                "audio_feature_version": AUDIO_FEATURE_VERSION,
            },
        )
        partial_index_path.unlink(missing_ok=True)
        partial_rejected_path.unlink(missing_ok=True)
        emit(
            stage="complete",
            current=total,
            total=total,
            percent=100.0,
            unique_audio=summary["unique_audio"],
            accepted=summary["accepted_charts"],
            rejected=summary["rejected_charts"],
            failed_unique_audio=summary["failed_unique_audio"],
            summary=summary,
        )
        return summary
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            checkpoint(stage="failed", status="failed", last_error=error)
        except Exception:
            pass
        emit(
            stage="failed",
            current=current,
            total=total,
            unique_audio=unique_audio,
            accepted=len(index_rows),
            rejected=len(rejected_rows),
            last_error=error,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ORBIT-4K V0 training data")
    parser.add_argument(
        "input",
        type=Path,
        help="osu! Songs directory, beatmap-set directory, or .zip",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/v0"))
    parser.add_argument("--config", type=Path, default=Path("configs/v0.yaml"))
    parser.add_argument(
        "--progress",
        action="store_true",
        help="emit machine-readable progress lines for the local Lab UI",
    )
    args = parser.parse_args()

    callback = None
    if args.progress:
        def callback(payload: dict) -> None:
            print(
                "ORBIT4K_PROGRESS " + json.dumps(payload, ensure_ascii=False),
                flush=True,
            )

    summary = build_dataset(
        args.input,
        args.output,
        args.config,
        progress_callback=callback,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
