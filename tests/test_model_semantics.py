from __future__ import annotations

import torch
from torch import nn

from orbit4k.model import Orbit4KV0


def _tiny_model() -> Orbit4KV0:
    model = Orbit4KV0(
        d_model=32,
        n_heads=4,
        audio_layers=1,
        chart_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    # Expose the concatenated lane slices directly for deterministic regression tests.
    model.chart_projection = nn.Identity()
    return model


def _common_inputs():
    tick = torch.tensor([[0]], dtype=torch.long)
    bpm = torch.tensor([180.0])
    stars = torch.tensor([5.0])
    return tick, bpm, stars


def test_chart_state_lane_permutation_is_distinguishable():
    model = _tiny_model()
    tick, bpm, stars = _common_inputs()
    active_ln = torch.zeros((1, 1, 4), dtype=torch.long)

    left_tap = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.long)
    right_tap = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.long)

    left = model._chart_embeddings(left_tap, active_ln, tick, bpm, stars)
    right = model._chart_embeddings(right_tap, active_ln, tick, bpm, stars)

    assert not torch.allclose(left, right)


def test_active_ln_lane_permutation_is_distinguishable():
    model = _tiny_model()
    tick, bpm, stars = _common_inputs()
    chart_input = torch.zeros((1, 1, 4), dtype=torch.long)

    left_hold = torch.tensor([[[1, 0, 0, 0]]], dtype=torch.long)
    right_hold = torch.tensor([[[0, 0, 0, 1]]], dtype=torch.long)

    left = model._chart_embeddings(chart_input, left_hold, tick, bpm, stars)
    right = model._chart_embeddings(chart_input, right_hold, tick, bpm, stars)

    assert not torch.allclose(left, right)


def test_forward_keeps_expected_output_shape():
    model = _tiny_model()
    batch, length = 2, 8
    audio = torch.randn(batch, length, 520)
    chart_input = torch.zeros((batch, length, 4), dtype=torch.long)
    active_ln = torch.zeros_like(chart_input)
    tick = torch.arange(length).repeat(batch, 1)
    bpm = torch.full((batch,), 180.0)
    stars = torch.full((batch,), 5.0)
    mask = torch.ones((batch, length))

    outputs = model(audio, chart_input, active_ln, tick, bpm, stars, mask)

    assert outputs["logits"].shape == (batch, length, 4, 4)
    assert outputs["onset_logits"].shape == (batch, length)
