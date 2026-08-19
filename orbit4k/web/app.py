from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from orbit4k.model import Orbit4KV0, parameter_count
from orbit4k.preprocessing.audio_features import AUDIO_FEATURE_VERSION, AUDIO_TOKEN_DIM
from orbit4k.preprocessing.build_dataset import _safe_extract
from orbit4k.preprocessing.osu_parser import scan_4k_beatmaps

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
CONFIG_PATH = ROOT / "configs" / "v0.yaml"

app = FastAPI(title="ORBIT-4K V0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


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
    summary_path = ROOT / "data" / "processed" / "v0" / "summary.json"
    dataset = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {
        "version": "V0",
        "parameters": parameter_count(model),
        "grid": "1/96 whole note (24 ticks / quarter)",
        "model": "Beat-synchronous Audio Encoder + Causal Chart Decoder",
        "audio_token_dim": AUDIO_TOKEN_DIM,
        "audio_feature_version": AUDIO_FEATURE_VERSION,
        "dataset": dataset,
        "checkpoint_ready": (ROOT / "runs" / "v0" / "best.pt").exists(),
    }


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
