from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .model import BOS_STATE, ContinuousCondition, sinusoidal_tick_encoding
from .preprocessing.audio_features import AUDIO_TOKEN_DIM, ENERGY_FEATURE_DIM, SPECTRAL_VIEW_DIM

MICRO_SLOTS = 3
LANES = 4
OUTPUT_STATES = 4


class Orbit4KV1(nn.Module):
    """Hierarchical 32-cell / 3-micro-slot ORBIT-4K model.

    One Transformer timestep represents one 1/32-note cell. Each cell retains
    three ordered 1/96 micro slots, so the final chart has exactly the same
    timing resolution as V0 while the expensive Transformer sequence is 3x
    shorter.
    """

    def __init__(
        self,
        *,
        audio_token_dim: int = AUDIO_TOKEN_DIM,
        audio_micro_dim: int = 128,
        d_model: int = 384,
        n_heads: int = 6,
        audio_layers: int = 6,
        chart_layers: int = 8,
        micro_layers: int = 1,
        micro_dim_feedforward: int = 768,
        dim_feedforward: int = 1536,
        dropout: float = 0.10,
        local_audio_scale_cells: float = 32.0,
        max_cross_attention_bias: float = 4.0,
    ) -> None:
        super().__init__()
        if audio_token_dim != SPECTRAL_VIEW_DIM + ENERGY_FEATURE_DIM:
            raise ValueError(
                f"V1 expects {SPECTRAL_VIEW_DIM} spectral + {ENERGY_FEATURE_DIM} energy features"
            )
        if d_model % (MICRO_SLOTS * LANES) != 0:
            raise ValueError("d_model must be divisible by 12 for 3 micro slots x 4 lanes")
        if audio_micro_dim * MICRO_SLOTS != d_model:
            raise ValueError("audio_micro_dim * 3 must equal d_model")

        self.d_model = int(d_model)
        self.chart_atom_dim = d_model // (MICRO_SLOTS * LANES)
        self.audio_micro_dim = int(audio_micro_dim)
        self.local_audio_scale_cells = float(local_audio_scale_cells)
        self.max_cross_attention_bias = float(max_cross_attention_bias)

        # Preserve where within the 1/32 cell the sound occurred. The three
        # 520-d V0 audio tokens are projected independently, receive explicit
        # micro-position embeddings, then concatenate into one 384-d cell token.
        self.micro_spectral_projection = nn.Linear(SPECTRAL_VIEW_DIM, audio_micro_dim)
        self.micro_energy_projection = nn.Sequential(
            nn.Linear(ENERGY_FEATURE_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, audio_micro_dim),
        )
        self.audio_micro_embedding = nn.Embedding(MICRO_SLOTS, audio_micro_dim)
        self.audio_cell_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.audio_input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=audio_layers,
            norm=nn.LayerNorm(d_model),
        )

        # A chart cell contains 3 micro slots x 4 lanes. Every atom gets its own
        # state/lane/micro identity; flattening 12 x 32-d atoms yields 384-d.
        self.state_embedding = nn.Embedding(5, self.chart_atom_dim)
        self.lane_embedding = nn.Embedding(LANES, self.chart_atom_dim)
        self.micro_chart_embedding = nn.Embedding(MICRO_SLOTS, self.chart_atom_dim)
        self.active_ln_embedding = nn.Embedding(2, self.chart_atom_dim)
        self.chart_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.beat_phase = nn.Embedding(8, d_model)
        self.measure_phase = nn.Embedding(32, d_model)
        self.condition = ContinuousCondition(d_model)
        self.same_cell_audio_gate = nn.Parameter(torch.zeros(d_model))
        self.same_cell_audio_norm = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chart_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=chart_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Cheap local mixer: the expensive temporal Transformer operates on 32
        # cells/measure, while a length-3 mixer lets the three 1/96 slots inside
        # each cell coordinate before emitting lane/state logits.
        self.output_micro_embedding = nn.Embedding(MICRO_SLOTS, d_model)
        micro_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=micro_dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.micro_mixer = nn.TransformerEncoder(
            micro_layer,
            num_layers=micro_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.state_head = nn.Linear(d_model, LANES * OUTPUT_STATES)
        self.micro_onset_head = nn.Linear(d_model, 1)
        self.cell_onset_head = nn.Linear(d_model, 1)

    @staticmethod
    def _causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros((length, length), device=device, dtype=dtype)
        upper = torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask.masked_fill(upper, float("-inf"))

    def _memory_bias_rect(
        self,
        target_length: int,
        memory_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target_positions = torch.arange(target_length, device=device, dtype=torch.float32)
        memory_positions = torch.arange(memory_length, device=device, dtype=torch.float32)
        distance = (target_positions[:, None] - memory_positions[None, :]).abs()
        bias = -(distance / max(self.local_audio_scale_cells, 1.0))
        return bias.clamp(min=-self.max_cross_attention_bias, max=0.0).to(dtype=dtype)

    def _audio_embeddings(self, audio: torch.Tensor, cell_tick: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 4 or audio.shape[2] != MICRO_SLOTS or audio.shape[3] != AUDIO_TOKEN_DIM:
            raise ValueError("V1 audio must have shape [B, cells, 3, 520]")
        spectral = audio[..., :SPECTRAL_VIEW_DIM]
        energy = audio[..., SPECTRAL_VIEW_DIM:]
        hidden = self.micro_spectral_projection(spectral) + self.micro_energy_projection(energy)
        micro_ids = torch.arange(MICRO_SLOTS, device=audio.device)
        hidden = hidden + self.audio_micro_embedding(micro_ids).view(1, 1, MICRO_SLOTS, -1)
        hidden = hidden.flatten(start_dim=2)
        hidden = self.audio_cell_projection(hidden)
        hidden = hidden + sinusoidal_tick_encoding(cell_tick, self.d_model).to(hidden.dtype)
        return self.audio_input_norm(hidden)

    def _chart_embeddings(
        self,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        cell_tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
    ) -> torch.Tensor:
        if chart_input.ndim != 4 or chart_input.shape[2:] != (MICRO_SLOTS, LANES):
            raise ValueError("V1 chart_input must have shape [B, cells, 3, 4]")
        if active_ln.ndim != 3 or active_ln.shape[-1] != LANES:
            raise ValueError("V1 active_ln must have shape [B, cells, 4]")

        lane_ids = torch.arange(LANES, device=chart_input.device)
        micro_ids = torch.arange(MICRO_SLOTS, device=chart_input.device)
        lane_bias = self.lane_embedding(lane_ids).view(1, 1, 1, LANES, self.chart_atom_dim)
        micro_bias = self.micro_chart_embedding(micro_ids).view(
            1, 1, MICRO_SLOTS, 1, self.chart_atom_dim
        )
        active_bias = self.active_ln_embedding(active_ln).unsqueeze(2)
        atoms = (
            self.state_embedding(chart_input)
            + lane_bias
            + micro_bias
            + active_bias
        )
        chart = self.chart_projection(atoms.flatten(start_dim=2))
        chart = (
            chart
            + self.beat_phase(cell_tick % 8)
            + self.measure_phase(cell_tick % 32)
            + sinusoidal_tick_encoding(cell_tick, self.d_model).to(chart.dtype)
        )
        return chart + self.condition(bpm, stars)[:, None, :]

    def encode_audio(
        self,
        audio: torch.Tensor,
        cell_tick: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self._audio_embeddings(audio, cell_tick)
        padding_bool = None if mask is None else mask <= 0.5
        memory = self.audio_encoder(hidden, src_key_padding_mask=padding_bool)
        return memory, padding_bool

    def decode_from_memory(
        self,
        memory: torch.Tensor,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        cell_tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        memory_padding_bool: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_length = chart_input.shape[1]
        memory_length = memory.shape[1]
        if target_length > memory_length:
            raise ValueError("chart prefix cannot be longer than audio memory")

        chart_hidden = self._chart_embeddings(chart_input, active_ln, cell_tick, bpm, stars)
        gate = torch.sigmoid(self.same_cell_audio_gate).view(1, 1, -1)
        chart_hidden = chart_hidden + gate * self.same_cell_audio_norm(memory[:, :target_length])
        target_padding_bool = None if target_mask is None else target_mask <= 0.5
        decoded = self.chart_decoder(
            chart_hidden,
            memory,
            tgt_mask=self._causal_mask(target_length, chart_hidden.device, chart_hidden.dtype),
            memory_mask=self._memory_bias_rect(
                target_length,
                memory_length,
                chart_hidden.device,
                chart_hidden.dtype,
            ),
            tgt_key_padding_mask=target_padding_bool,
            memory_key_padding_mask=memory_padding_bool,
        )

        micro_ids = torch.arange(MICRO_SLOTS, device=decoded.device)
        micro_hidden = decoded.unsqueeze(2) + self.output_micro_embedding(micro_ids).view(
            1, 1, MICRO_SLOTS, self.d_model
        )
        batch, cells = decoded.shape[:2]
        micro_hidden = self.micro_mixer(micro_hidden.reshape(batch * cells, MICRO_SLOTS, self.d_model))
        micro_hidden = micro_hidden.reshape(batch, cells, MICRO_SLOTS, self.d_model)
        logits = self.state_head(micro_hidden).view(
            batch, cells, MICRO_SLOTS, LANES, OUTPUT_STATES
        )
        micro_onset_logits = self.micro_onset_head(micro_hidden).squeeze(-1)
        cell_onset_logits = self.cell_onset_head(decoded).squeeze(-1)
        return {
            "logits": logits,
            "micro_onset_logits": micro_onset_logits,
            "cell_onset_logits": cell_onset_logits,
            "hidden": decoded,
            "micro_hidden": micro_hidden,
        }

    def forward(
        self,
        audio: torch.Tensor,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        cell_tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, padding_bool = self.encode_audio(audio, cell_tick, mask)
        return self.decode_from_memory(
            memory,
            chart_input,
            active_ln,
            cell_tick,
            bpm,
            stars,
            target_mask=mask,
            memory_padding_bool=padding_bool,
        )


def warm_start_from_v0(model: Orbit4KV1, checkpoint_path: str | Path) -> dict[str, int | list[str]]:
    """Reuse the V0 Transformer backbone while keeping V1-specific heads fresh."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    old = checkpoint["model"]
    current = model.state_dict()
    copied: list[str] = []

    exact_prefixes = (
        "audio_encoder.",
        "chart_decoder.",
        "condition.",
        "audio_input_norm.",
    )
    exact_names = {"same_tick_audio_gate", "same_tick_audio_norm.weight", "same_tick_audio_norm.bias"}
    # V1 renamed same-tick -> same-cell; map those tensors explicitly.
    name_map = {
        "same_cell_audio_gate": "same_tick_audio_gate",
        "same_cell_audio_norm.weight": "same_tick_audio_norm.weight",
        "same_cell_audio_norm.bias": "same_tick_audio_norm.bias",
    }

    for name, tensor in list(current.items()):
        source_name = name_map.get(name, name)
        allowed = name.startswith(exact_prefixes) or source_name in exact_names
        if allowed and source_name in old and old[source_name].shape == tensor.shape:
            current[name] = old[source_name].detach().clone()
            copied.append(name)

    # V0 lane/state embeddings are 96-d; V1 atoms are 32-d. Average the three
    # equally-sized V0 thirds to obtain a stable warm-start rather than slicing.
    for name in ("state_embedding.weight", "lane_embedding.weight", "active_ln_embedding.weight"):
        if name in old and name in current and old[name].shape[-1] == current[name].shape[-1] * 3:
            reduced = old[name].reshape(old[name].shape[0], 3, current[name].shape[-1]).mean(dim=1)
            current[name] = reduced
            copied.append(name)

    if "beat_phase.weight" in old and old["beat_phase.weight"].shape[0] >= 24:
        current["beat_phase.weight"] = old["beat_phase.weight"][::3][:8].detach().clone()
        copied.append("beat_phase.weight")
    if "measure_phase.weight" in old and old["measure_phase.weight"].shape[0] >= 96:
        current["measure_phase.weight"] = old["measure_phase.weight"][::3][:32].detach().clone()
        copied.append("measure_phase.weight")

    model.load_state_dict(current, strict=True)
    return {
        "copied_tensors": len(copied),
        "copied": copied,
        "source_epoch": int(checkpoint.get("epoch", 0)),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
