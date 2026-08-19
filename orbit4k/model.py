from __future__ import annotations

import torch
from torch import nn

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


class Orbit4KV0(nn.Module):
    """Audio-conditioned encoder/decoder Transformer for direct 4K chart generation."""

    def __init__(
        self,
        *,
        n_mels: int = 128,
        d_model: int = 384,
        n_heads: int = 6,
        audio_layers: int = 6,
        chart_layers: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.audio_projection = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.LayerNorm(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=audio_layers, norm=nn.LayerNorm(d_model))

        self.state_embedding = nn.Embedding(5, d_model)
        self.lane_embedding = nn.Embedding(4, d_model)
        self.active_ln_embedding = nn.Embedding(2, d_model)
        self.beat_phase = nn.Embedding(24, d_model)
        self.measure_phase = nn.Embedding(96, d_model)
        self.condition = ContinuousCondition(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.chart_decoder = nn.TransformerDecoder(decoder_layer, num_layers=chart_layers, norm=nn.LayerNorm(d_model))
        self.state_head = nn.Linear(d_model, 4 * 4)
        self.onset_head = nn.Linear(d_model, 1)

    def _chart_embeddings(
        self,
        chart_input: torch.Tensor,
        active_ln: torch.Tensor,
        tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, lanes = chart_input.shape
        lane_ids = torch.arange(4, device=chart_input.device)
        states = self.state_embedding(chart_input)
        lane_bias = self.lane_embedding(lane_ids).view(1, 1, 4, self.d_model)
        chart = (states + lane_bias).mean(dim=2)
        active = (self.active_ln_embedding(active_ln) + lane_bias).mean(dim=2)
        chart = chart + active + self.beat_phase(tick % 24) + self.measure_phase(tick % 96)
        chart = chart + self.condition(bpm, stars)[:, None, :]
        return chart

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

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
        audio_hidden = self.audio_projection(audio)
        padding = None if mask is None else mask <= 0.5
        memory = self.audio_encoder(audio_hidden, src_key_padding_mask=padding)
        chart_hidden = self._chart_embeddings(chart_input, active_ln, tick, bpm, stars)
        decoded = self.chart_decoder(
            chart_hidden,
            memory,
            tgt_mask=self._causal_mask(chart_hidden.shape[1], chart_hidden.device),
            tgt_key_padding_mask=padding,
            memory_key_padding_mask=padding,
        )
        logits = self.state_head(decoded).view(decoded.shape[0], decoded.shape[1], 4, 4)
        onset_logits = self.onset_head(decoded).squeeze(-1)
        return {"logits": logits, "onset_logits": onset_logits, "hidden": decoded}

    @torch.inference_mode()
    def generate_window(
        self,
        audio: torch.Tensor,
        tick: torch.Tensor,
        bpm: torch.Tensor,
        stars: torch.Tensor,
        *,
        temperature: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        batch, length, _ = audio.shape
        result = torch.zeros((batch, length, 4), dtype=torch.long, device=audio.device)
        chart_input = torch.full_like(result, BOS_STATE)
        active_ln = torch.zeros_like(result)
        active_now = torch.zeros((batch, 4), dtype=torch.long, device=audio.device)
        mask = torch.ones((batch, length), dtype=torch.float32, device=audio.device)
        # V0 reference implementation. It is intentionally simple; KV caching can be added after the baseline is proven.
        for position in range(length):
            if position > 0:
                chart_input[:, position] = result[:, position - 1]
            active_ln[:, position] = active_now
            outputs = self(audio, chart_input, active_ln, tick, bpm, stars, mask)
            logits = outputs["logits"][:, position] / max(temperature, 1e-4)
            result[:, position] = torch.distributions.Categorical(logits=logits).sample()
            active_now = torch.where(result[:, position] == 2, 1, active_now)
            active_now = torch.where(result[:, position] == 3, 0, active_now)
        return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
