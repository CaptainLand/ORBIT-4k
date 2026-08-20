from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orbit4k.model import Orbit4KV0, parameter_count
from orbit4k.preprocessing.audio_features import AUDIO_FEATURE_VERSION, AUDIO_TOKEN_DIM
from orbit4k.preprocessing.build_dataset import _safe_extract
from orbit4k.preprocessing.osu_parser import scan_4k_beatmaps

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
CONFIG_PATH = ROOT / "configs" / "v0.yaml"

app = FastAPI(title="ORBIT-4K V0 Lab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    if not value or not value.strip():
        raise HTTPException(400, "path must not be empty")
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


class JobRequest(BaseModel):
    config_path: str = "configs/v0.yaml"


class PrepareRequest(JobRequest):
    source_path: str
    output_path: str


class TrainRequest(JobRequest):
    data_path: str
    run_dir: str


class GenerateRequest(BaseModel):
    mode: str = "preview"
    checkpoint_path: str = "runs/v0/best.pt"
    audio_path: str
    output_dir: str = "runs/v0/generated"
    bpm: float
    offset_ms: float
    stars: float
    temperature: float = 0.85
    measures: int = 4
    window_measures: int = 4
    context_measures: int = 1
    seed: int = 20260820


class ManagedJob:
    def __init__(self, name: str) -> None:
        self.name = name
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.state = "idle"
        self.command: list[str] = []
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.returncode: int | None = None
        self.logs: deque[str] = deque(maxlen=300)
        self.progress: dict[str, Any] = {}
        self.records: deque[dict[str, Any]] = deque(maxlen=200)
        self.paths: dict[str, str] = {}

    def running(self) -> bool:
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def start(self, command: list[str], *, paths: dict[str, str]) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError(f"{self.name} job is already running")
            self.state = "running"
            self.command = command
            self.started_at = time.time()
            self.ended_at = None
            self.returncode = None
            self.logs.clear()
            self.progress = {}
            self.records.clear()
            self.paths = dict(paths)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            process = self.process
        threading.Thread(target=self._read_output, args=(process,), daemon=True).start()

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            with self.lock:
                self.logs.append(line)
                if line.startswith("ORBIT4K_PROGRESS "):
                    try:
                        payload = json.loads(line[len("ORBIT4K_PROGRESS "):])
                        self.progress.update(payload)
                    except json.JSONDecodeError:
                        pass
                else:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        record = None
                    if isinstance(record, dict) and "epoch" in record:
                        self.records.append(record)
                        self.progress = {
                            "stage": "training",
                            "epoch": record.get("epoch"),
                            "train": record.get("train", {}),
                            "validation": record.get("validation", {}),
                            "learning_rate": record.get("learning_rate"),
                        }
                    elif isinstance(record, dict) and record.get("stage") == "complete":
                        self.progress.update(record)
        returncode = process.wait()
        with self.lock:
            self.returncode = returncode
            self.ended_at = time.time()
            if self.state == "stopping":
                self.state = "stopped"
            else:
                self.state = "completed" if returncode == 0 else "failed"
            if self.process is process:
                self.process = None

    def stop(self) -> bool:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return False
            self.state = "stopping"
        process.terminate()
        return True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            end = self.ended_at or now
            elapsed = None if self.started_at is None else max(0.0, end - self.started_at)
            failure_reason = None
            if self.state == "failed":
                failure_reason = self.progress.get("last_error")
                if not failure_reason:
                    for line in reversed(self.logs):
                        if line.strip() and not line.startswith("ORBIT4K_PROGRESS "):
                            failure_reason = line.strip()
                            break
            return {
                "name": self.name,
                "state": self.state,
                "running": self.process is not None and self.process.poll() is None,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "elapsed_seconds": elapsed,
                "returncode": self.returncode,
                "failure_reason": failure_reason,
                "progress": dict(self.progress),
                "records": list(self.records),
                "logs": list(self.logs),
                "paths": dict(self.paths),
            }


prepare_job = ManagedJob("prepare")
train_job = ManagedJob("train")
generate_job = ManagedJob("generate")


def ensure_idle() -> None:
    if prepare_job.running() or train_job.running() or generate_job.running():
        raise HTTPException(409, "another ORBIT-4K job is already running")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def dataset_summary(path: Path) -> dict | None:
    return _read_json(path / "summary.json")


def dataset_status(path: Path) -> dict[str, Any]:
    index_path = path / "index.jsonl"
    summary = dataset_summary(path)
    state = _read_json(path / "dataset_state.json")
    index_exists = index_path.is_file()
    summary_exists = summary is not None

    if state is None:
        complete = index_exists and summary_exists
        status_name = "complete" if complete else "missing"
    else:
        complete = state.get("status") == "complete"
        status_name = str(state.get("status") or "unknown")

    ready = bool(complete and index_exists and summary_exists)
    return {
        "path": str(path),
        "ready": ready,
        "status": status_name,
        "index_exists": index_exists,
        "summary_exists": summary_exists,
        "state": state,
        "summary": summary,
        "partial_index_exists": (path / "index.partial.jsonl").is_file(),
        "partial_rejected_exists": (path / "rejected.partial.json").is_file(),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    cfg = config()
    model = Orbit4KV0(
        audio_token_dim=cfg["model"]["audio_token_dim"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        audio_layers=cfg["model"]["audio_layers"],
        chart_layers=cfg["model"]["chart_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        local_audio_scale_ticks=cfg["model"]["local_audio_scale_ticks"],
        max_cross_attention_bias=cfg["model"]["max_cross_attention_bias"],
    )
    default_data = ROOT / "data" / "processed" / "v0"
    return {
        "version": "V0",
        "parameters": parameter_count(model),
        "grid": "1/96 whole note (24 ticks / quarter)",
        "model": "Beat-synchronous Audio Encoder + Causal Chart Decoder",
        "audio_token_dim": AUDIO_TOKEN_DIM,
        "audio_feature_version": AUDIO_FEATURE_VERSION,
        "dataset": dataset_summary(default_data),
        "checkpoint_ready": (ROOT / "runs" / "v0" / "best.pt").exists(),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "root": str(ROOT),
    }


@app.get("/api/jobs")
def jobs():
    return {
        "prepare": prepare_job.snapshot(),
        "train": train_job.snapshot(),
        "generate": generate_job.snapshot(),
    }


@app.post("/api/prepare/start")
def start_prepare(request: PrepareRequest):
    ensure_idle()
    source = resolve_path(request.source_path)
    output = resolve_path(request.output_path)
    cfg = resolve_path(request.config_path)
    if not source.exists():
        raise HTTPException(400, f"source path does not exist: {source}")
    if not source.is_dir() and source.suffix.lower() != ".zip":
        raise HTTPException(400, "source must be an osu! Songs directory, beatmap-set directory, or .zip")
    if not cfg.is_file():
        raise HTTPException(400, f"config file does not exist: {cfg}")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "prepare_dataset.py"),
        str(source),
        "--output",
        str(output),
        "--config",
        str(cfg),
        "--progress",
    ]
    prepare_job.start(
        command,
        paths={"source": str(source), "output": str(output), "config": str(cfg)},
    )
    return {"ok": True, "source": str(source), "output": str(output)}


@app.post("/api/prepare/stop")
def stop_prepare():
    return {"ok": prepare_job.stop()}


@app.post("/api/train/start")
def start_train(request: TrainRequest):
    ensure_idle()
    data = resolve_path(request.data_path)
    run_dir = resolve_path(request.run_dir)
    cfg = resolve_path(request.config_path)
    readiness = dataset_status(data)
    if not readiness["ready"]:
        detail = (
            f"dataset is not ready (state={readiness['status']}). "
            "Finish Dataset Builder first; partial caches/checkpoints are not a trainable dataset."
        )
        raise HTTPException(400, detail)
    if not cfg.is_file():
        raise HTTPException(400, f"config file does not exist: {cfg}")
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "train_v0.py"),
        "--data",
        str(data),
        "--run-dir",
        str(run_dir),
        "--config",
        str(cfg),
    ]
    train_job.start(
        command,
        paths={"data": str(data), "run_dir": str(run_dir), "config": str(cfg)},
    )
    return {"ok": True, "data": str(data), "run_dir": str(run_dir)}


@app.post("/api/train/stop")
def stop_train():
    return {"ok": train_job.stop()}


@app.post("/api/generate/start")
def start_generate(request: GenerateRequest):
    ensure_idle()
    checkpoint = resolve_path(request.checkpoint_path)
    audio = resolve_path(request.audio_path)
    output_dir = resolve_path(request.output_dir)
    mode = request.mode.strip().lower()
    if mode not in {"preview", "full_song"}:
        raise HTTPException(400, "generation mode must be preview or full_song")
    if not checkpoint.is_file():
        raise HTTPException(400, f"checkpoint not found: {checkpoint}")
    if not audio.is_file():
        raise HTTPException(400, f"audio file not found: {audio}")
    if audio.suffix.lower() not in {".mp3", ".wav", ".ogg", ".flac", ".m4a"}:
        raise HTTPException(400, "audio must be MP3/WAV/OGG/FLAC/M4A supported by torchaudio")
    if not 20.0 <= request.bpm <= 500.0:
        raise HTTPException(400, "BPM must be between 20 and 500")
    if not 0.1 <= request.stars <= 15.0:
        raise HTTPException(400, "target SR must be between 0.1 and 15")
    if not 0.05 <= request.temperature <= 2.0:
        raise HTTPException(400, "temperature must be between 0.05 and 2.0")
    if mode == "preview" and not 1 <= request.measures <= 16:
        raise HTTPException(400, "preview measures must be between 1 and 16")
    if mode == "full_song":
        if not 3 <= request.window_measures <= 16:
            raise HTTPException(400, "full-song window measures must be between 3 and 16")
        if not 1 <= request.context_measures or 2 * request.context_measures >= request.window_measures:
            raise HTTPException(400, "context must be at least 1 measure and less than half the window")

    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "generate_v0.py"),
        "--checkpoint",
        str(checkpoint),
        "--audio",
        str(audio),
        "--output-dir",
        str(output_dir),
        "--bpm",
        str(request.bpm),
        "--offset-ms",
        str(request.offset_ms),
        "--stars",
        str(request.stars),
        "--temperature",
        str(request.temperature),
        "--seed",
        str(request.seed),
    ]
    if mode == "full_song":
        command.extend(
            [
                "--full-song",
                "--window-measures",
                str(request.window_measures),
                "--context-measures",
                str(request.context_measures),
            ]
        )
    else:
        command.extend(["--measures", str(request.measures)])

    generate_job.start(
        command,
        paths={
            "mode": mode,
            "checkpoint": str(checkpoint),
            "audio": str(audio),
            "output_dir": str(output_dir),
        },
    )
    return {
        "ok": True,
        "mode": mode,
        "checkpoint": str(checkpoint),
        "audio": str(audio),
        "output_dir": str(output_dir),
    }


@app.post("/api/generate/stop")
def stop_generate():
    return {"ok": generate_job.stop()}


@app.get("/api/dataset-summary")
def get_dataset_summary(path: str):
    resolved = resolve_path(path)
    summary = dataset_summary(resolved)
    if summary is None:
        raise HTTPException(404, f"summary.json not found under {resolved}")
    return {"path": str(resolved), "summary": summary}


@app.get("/api/dataset-status")
def get_dataset_status(path: str):
    resolved = resolve_path(path)
    return dataset_status(resolved)


@app.post("/api/inspect-zip")
async def inspect_zip(file: UploadFile = File(...)):
    suffix = Path(file.filename or "beatmap.zip").suffix.lower()
    if suffix != ".zip":
        return {"ok": False, "error": "Please upload an osu! beatmap-set .zip"}
    with tempfile.TemporaryDirectory(prefix="orbit4k_web_") as tmp:
        archive_path = Path(tmp) / "upload.zip"
        archive_path.write_bytes(await file.read())
        extract_root = Path(tmp) / "set"
        extract_root.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _safe_extract(archive, extract_root)
        except zipfile.BadZipFile:
            return {"ok": False, "error": "Invalid zip archive"}
        cfg = config()
        accepted, rejected = scan_4k_beatmaps(
            extract_root,
            ticks_per_quarter=cfg["grid"]["ticks_per_quarter"],
            constant_bpm_only=cfg["grid"]["constant_bpm_only"],
            max_quantization_error_ms=cfg["grid"]["max_quantization_error_ms"],
            calculate_stars=True,
        )
        return {
            "ok": True,
            "accepted": [
                {
                    "title": item.title,
                    "version": item.version,
                    "bpm": round(item.bpm, 4),
                    "offset_ms": round(item.offset_ms, 3),
                    "stars": None if item.star_rating is None else round(item.star_rating, 3),
                    "objects": item.object_count,
                    "p95_error_ms": round(item.p95_quantization_error_ms, 3),
                }
                for item in accepted
            ],
            "rejected": rejected[:100],
        }
