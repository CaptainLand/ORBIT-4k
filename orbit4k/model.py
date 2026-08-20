from __future__ import annotations

import math

import torch
from torch import nn

from .preprocessing.audio_features import (
    AUDIO_TOKEN_DIM,
    ENERGY_FEATURE_DIM,
    SPECTRAL_VIEW_DIM,
)

BOS_STATE = 4


class ContinuousCondition(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, bpm: torch.Tensor, stars: torch.Tensor) -> torch.Tensor:
        values = torch.stack([(bpm - 180.0) / 80.0, (stars - 5.0) / 3.0], dim=-1)
        return self.net(values)


def sinusoidal_tick_encoding(tick: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        torch.arange(half, device=tick.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = tick.float().unsqueeze(-1) * frequencies
    encoded = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if encoded.shape[-1] < dimension:
        encoded = torch.nn.functional.pad(encoded, (0, dimension - encoded.shape[-1]))
    return encoded


class Orbit4KV0(nn.Module):
    """Audio-conditioned encoder/decoder Transformer for direct 4K chart generation."""

    def __init__(
        self,
        *,
        audio_token_dim: int = AUDIO_TOKEN_DIM,
        d_model: int = 384,
        n_heads: int = 6,
        audio_layers: int = 6,
        chart_layers: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.10,
        local_audio_scale_ticks: float = 96.0,
        max_cross_attention_bias: float = 4.0,
    ) -> None:
        super().__init__()
        if audio_token_dim != SPECTRAL_VIEW_DIM + ENERGY_FEATURE_DIM:
            raise ValueError(
                f"V0 expects {SPECTRAL_VIEW_DIM} spectral + {ENERGY_FEATURE_DIM} energy features"
            )
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 so every 4K lane keeps its own embedding slice")

        self.d_model = d_model
        self.lane_dim = d_model // 4
        self.local_audio_scale_ticks = float(local_audio_scale_ticks)
        self.max_cross_attention_bias = float(max_cross_attention_bias)

        self.spectral_projection = nn.Linear(SPECTRAL_VIEW_DIM, d_model)
        self.energy_projection = nn.Sequential(
            nn.Linear(ENERGY_FEATURE_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
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

        # Preserve lane identity explicitly. Each lane owns a fixed slice of the
        # chart token, so [TAP, EMPTY, EMPTY, EMPTY] and
        # [EMPTY, EMPTY, EMPTY, TAP] cannot collapse to the same representation.
        self.state_embedding = nn.Embedding(5, self.lane_dim)
        self.lane_embedding = nn.Embedding(4, self.lane_dim)
        self.active_ln_embedding = nn.Embedding(2, self.lane_dim)
        self.chart_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.beat_phase = nn.Embedding(24, d_model)
        self.measure_phase = nn.Embedding(96, d_model)
        self.condition = ContinuousCondition(d_model)
        self.same_tick_audio_gate = nn.Parameter(torch.zeros(d_model))
        self.same_tick_audio_norm = nn.LayerNorm(d_model)

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
        self.state_head = nn.Linear(d_model, 4 * 4)
        self.onset_head = nn.Linear(d_model, 1)

    def _audio_embeddings(self, audio: torch.Tensor, tick: torch.Tensor) -> torch.Tensor:
        spectral = audio[..., :SPECTRAL_VIEW_DIM]
        energy = audio[..., SPECTRAL_VIEW_DIM:]
        hidden = self.spectral_projection(spectral) + self.energy_projection(energy)
        hidden = hidden + sinusoidal_tick_encoding(tick, self.d_model).to(hidden.dtype)
        return self.audio_input_norm(hidden)

    def _chart_embeddings(
        self,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
    ) -> torch.Tensor:
        if chart_input.shape[-1] != 4 or active_ln.shape[-1] != 4:
            raise ValueError("ORBIT-4K V0 expects exactly four chart lanes")

        lane_ids = torch.arange(4, device=chart_input.device)
        lane_bias = self.lane_embedding(lane_ids).view(1, 1, 4, self.lane_dim)
        lane_features = (
            self.state_embedding(chart_input)
            + self.active_ln_embedding(active_ln)
            + lane_bias
        )
        chart = self.chart_projection(lane_features.flatten(start_dim=2))
        chart = (
            chart
            + self.beat_phase(tick % 24)
            + self.measure_phase(tick % 96)
            + sinusoidal_tick_encoding(tick, self.d_model).to(chart.dtype)
        )
        chart = chart + self.condition(bpm, stars)[:, None, :]
        return chart

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
        bias = -(distance / max(self.local_audio_scale_ticks, 1.0))
        bias = bias.clamp(min=-self.max_cross_attention_bias, max=0.0)
        return bias.to(dtype=dtype)

    def _memory_bias(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._memory_bias_rect(length, length, device, dtype)

    def encode_audio(
        self,
        audio: torch.Tensor,
        tick: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode audio once so autoregressive generation can reuse the memory."""
        audio_hidden = self._audio_embeddings(audio, tick)
        padding_bool = None if mask is None else mask <= 0.5
        memory = self.audio_encoder(audio_hidden, src_key_padding_mask=padding_bool)
        return memory, padding_bool

    def decode_from_memory(
        self,
        memory: torch.Tensor,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
        *,
        target_mask: torch.Tensor | None = None,
        memory_padding_bool: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_length = chart_input.shape[1]
        memory_length = memory.shape[1]
        if target_length > memory_length:
            raise ValueError("chart prefix cannot be longer than encoded audio memory")

        chart_hidden = self._chart_embeddings(chart_input, active_ln, tick, bpm, stars)
        gate = torch.sigmoid(self.same_tick_audio_gate).view(1, 1, -1)
        chart_hidden = chart_hidden + gate * self.same_tick_audio_norm(memory[:, :target_length])

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
        logits = self.state_head(decoded).view(decoded.shape[0], decoded.shape[1], 4, 4)
        onset_logits = self.onset_head(decoded).squeeze(-1)
        return {"logits": logits, "onset_logits": onset_logits, "hidden": decoded}

    def forward(
        self,
        audio: torch.Tensor,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        memory, padding_bool = self.encode_audio(audio, tick, mask)
        return self.decode_from_memory(
            memory,
            chart_input,
            active_ln,
            tick,
            bpm,
            stars,
            target_mask=mask,
            memory_padding_bool=padding_bool,
        )

    @torch.inference_mode()
    def generate_window(
        self,
        audio: torch.Tensor,
        tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
        *,
        temperature: float = 0.9,
        progress_callback=None,
    ) -> torch.Tensor:
        """Autoregressively generate a window while caching Audio Encoder memory.

        V0 still recomputes the causal Chart Decoder prefix at every tick because
        ``nn.TransformerDecoder`` does not expose a KV cache. Caching audio removes
        the largest avoidable repeated encoder cost and keeps the implementation
        identical to training semantics.
        """
        self.eval()
        batch, length, _ = audio.shape
        result = torch.zeros((batch, length, 4), dtype=torch.long, device=audio.device)
        chart_input = torch.full_like(result, BOS_STATE)
        active_ln = torch.zeros_like(result)
        active_now = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)
        mask = torch.ones((batch, length), dtype=torch.float32, device=audio.device)
        memory, memory_padding = self.encode_audio(audio, tick, mask)

        report_every = max(1, length // 100)
        for position in range(length):
            if position > 0:
                chart_input[:, position] = result[:, position - 1]
            active_ln[:, position] = active_now
            prefix = position + 1
            outputs = self.decode_from_memory(
                memory,
                chart_input[:, :prefix],
                active_ln[:, :prefix],
                tick[:, :prefix],
                bpm,
                stars,
                target_mask=mask[:, :prefix],
                memory_padding_bool=memory_padding,
            )
            logits = outputs["logits"][:, -1] / max(temperature, 1e-4)
            result[:, position] = torch.distributions.Categorical(logits=logits).sample()
            active_now = torch.where(result[:, position] == 2, 1, active_now)
            active_now = torch.where(result[:, position] == 3, 0, active_now)
            if progress_callback is not None and (
                position == 0 or position + 1 == length or (position + 1) % report_every == 0
            ):
                progress_callback(position + 1, length)
        return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
