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
                        self.progress = json.loads(line[len("ORBIT4K_PROGRESS "):])
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
            return {
                "name": self.name,
                "state": self.state,
                "running": self.process is not None and self.process.poll() is None,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "elapsed_seconds": elapsed,
                "returncode": self.returncode,
                "progress": dict(self.progress),
                "records": list(self.records),
                "logs": list(self.logs),
                "paths": dict(self.paths),
            }


prepare_job = ManagedJob("prepare")
train_job = ManagedJob("train")


def ensure_idle() -> None:
    if prepare_job.running() or train_job.running():
        raise HTTPException(409, "another ORBIT-4K job is already running")


def dataset_summary(path: Path) -> dict | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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
    if not (data / "index.jsonl").is_file():
        raise HTTPException(400, f"dataset index not found: {data / 'index.jsonl'}")
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


@app.get("/api/dataset-summary")
def get_dataset_summary(path: str):
    resolved = resolve_path(path)
    summary = dataset_summary(resolved)
    if summary is None:
        raise HTTPException(404, f"summary.json not found under {resolved}")
    return {"path": str(resolved), "summary": summary}


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
