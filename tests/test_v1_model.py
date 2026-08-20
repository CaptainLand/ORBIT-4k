from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from orbit4k.model import BOS_STATE, Orbit4KV0
from orbit4k.model_v1 import Orbit4KV1, warm_start_from_v0


def _small_v1() -> Orbit4KV1:
    return Orbit4KV1(
        audio_micro_dim=32,
        d_model=96,
        n_heads=6,
        audio_layers=1,
        chart_layers=1,
        micro_layers=1,
        micro_dim_feedforward=192,
        dim_feedforward=192,
        dropout=0.0,
        local_audio_scale_cells=8.0,
    )


def test_v1_forward_preserves_three_micro_slots_per_cell():
    model = _small_v1().eval()
    batch, cells = 2, 7
    audio = torch.randn(batch, cells, 3, 520)
    chart_input = torch.full((batch, cells, 3, 4), BOS_STATE, dtype=torch.long)
    active_ln = torch.zeros(batch, cells, 4, dtype=torch.long)
    cell_tick = torch.arange(cells).repeat(batch, 1)
    bpm = torch.tensor([170.0, 190.0])
    stars = torch.tensor([5.0, 7.0])
    mask = torch.ones(batch, cells)

    with torch.no_grad():
        out = model(audio, chart_input, active_ln, cell_tick, bpm, stars, mask)

    assert out["logits"].shape == (batch, cells, 3, 4, 4)
    assert out["micro_onset_logits"].shape == (batch, cells, 3)
    assert out["cell_onset_logits"].shape == (batch, cells)
    # 7 Transformer cells still describe 21 original 1/96 ticks.
    assert out["logits"].shape[1] * out["logits"].shape[2] == 21


def test_v1_warm_start_reuses_v0_transformer_backbone():
    v0 = Orbit4KV0(
        d_model=96,
        n_heads=6,
        audio_layers=1,
        chart_layers=1,
        dim_feedforward=192,
        dropout=0.0,
        local_audio_scale_ticks=24.0,
    )
    v1 = _small_v1()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v0.pt"
        torch.save({"model": v0.state_dict(), "config": {}, "epoch": 3}, path)
        report = warm_start_from_v0(v1, path)

    assert report["source_epoch"] == 3
    assert report["copied_tensors"] > 10
    assert torch.equal(
        v1.audio_encoder.layers[0].self_attn.in_proj_weight,
        v0.audio_encoder.layers[0].self_attn.in_proj_weight,
    )
    assert torch.equal(v1.beat_phase.weight, v0.beat_phase.weight[::3][:8])
    assert torch.equal(v1.measure_phase.weight, v0.measure_phase.weight[::3][:32])
