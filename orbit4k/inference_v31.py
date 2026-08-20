from __future__ import annotations

from pathlib import Path

import torch

from . import inference_full as _full
from . import inference_v3 as _v3
from .validator import EMPTY, LN_END, LN_START, TAP

DEFAULT_LN_START_MARGIN = 0.30
DEFAULT_AUTO_MAX_CHORD = 4


def _soft_chord_threshold(
    rank: int,
    *,
    top_probability: float,
    lane_threshold: float,
    stars: float,
) -> float:
    """Probability threshold for an additional chord lane.

    V3.1 deliberately uses *soft evidence* rather than SR-dependent hard bans.
    At 6★ the second/third/fourth lanes need progressively stronger evidence,
    but even 3/4-key chords remain possible when the model is genuinely sure.
    Higher SR relaxes these thresholds slightly; lower SR raises them slightly.
    """
    if rank <= 1:
        return float(lane_threshold)

    # rank is 2/3/4 for the 2nd/3rd/4th selected key.
    absolute = {2: 0.42, 3: 0.64, 4: 0.82}[rank]
    relative = {2: 0.72, 3: 0.84, 4: 0.94}[rank]
    difficulty_adjust = max(-0.06, min(0.06, (6.0 - float(stars)) * 0.015))
    return float(
        max(
            float(lane_threshold) + 0.04 * (rank - 1),
            absolute + difficulty_adjust,
            top_probability * relative,
        )
    )


def constrained_tick_states_v31(
    lane_logits: torch.Tensor,
    onset_logit: torch.Tensor,
    active_now: torch.Tensor,
    ln_age: torch.Tensor,
    *,
    stars: float,
    temperature: float = 0.85,
    onset_threshold: float = 0.0,
    lane_threshold: float = 0.32,
    ln_start_margin: float = DEFAULT_LN_START_MARGIN,
    release_threshold: float = 0.50,
    min_ln_ticks: int = 2,
    max_ln_ticks: int = 192,
    max_chord: int = 0,
    final_tick: bool = False,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """V3.1 pattern policy: legality constraints without style straitjackets.

    The V3 adaptive onset system still decides *when* a musical event belongs.
    This function only resolves lanes and TAP/LN state. Unlike V2/V3 it does not
    force low-confidence onsets to single taps and does not impose an SR-based
    hard chord ceiling. Additional chord lanes are admitted by progressively
    stronger per-lane evidence. LN_START keeps only a mild anti-collapse margin.
    """
    if lane_logits.shape != (4, 4):
        raise ValueError(f"expected lane logits [4,4], got {tuple(lane_logits.shape)}")
    if active_now.shape != (4,) or ln_age.shape != (4,):
        raise ValueError("active_now and ln_age must be [4]")

    temp = max(float(temperature), 1e-4)
    gate_threshold = float(onset_threshold)
    if gate_threshold <= 0:
        # V3/V3.1 callers normally pass an adaptive threshold explicitly. This
        # fallback keeps direct calls sensible without importing V2's 0.50 gate.
        gate_threshold = 0.40
    chord_cap = DEFAULT_AUTO_MAX_CHORD if int(max_chord) <= 0 else max(1, min(4, int(max_chord)))
    next_state = torch.zeros(4, dtype=torch.long, device=lane_logits.device)

    # Pure legality: an already-held lane can only stay held or end. It cannot
    # TAP or LN_START again before the previous hold ends.
    release_count = 0
    for lane in range(4):
        if int(active_now[lane].item()) == 0:
            continue
        age = int(ln_age[lane].item())
        force_end = final_tick or age >= int(max_ln_ticks)
        if force_end:
            next_state[lane] = LN_END
            release_count += 1
            continue
        if age < int(min_ln_ticks):
            continue
        pair = torch.stack([lane_logits[lane, EMPTY], lane_logits[lane, LN_END]]) / temp
        end_probability = torch.softmax(pair, dim=0)[1]
        if float(end_probability.item()) >= float(release_threshold):
            next_state[lane] = LN_END
            release_count += 1

    onset_probability = float(torch.sigmoid(onset_logit / temp).item())
    selected: list[int] = []
    fallback = False

    if not final_tick and onset_probability >= gate_threshold:
        inactive = [lane for lane in range(4) if int(active_now[lane].item()) == 0]
        if inactive:
            probabilities = torch.softmax(lane_logits / temp, dim=-1)
            key_probability = probabilities[:, TAP] + probabilities[:, LN_START]
            ranked = sorted(
                inactive,
                key=lambda lane: float(key_probability[lane].item()),
                reverse=True,
            )
            top_lane = ranked[0]
            top_probability = float(key_probability[top_lane].item())

            # If the V3 onset head says there is a musical event, always give it
            # the strongest legal lane. This preserves V3's anti-silence behavior.
            selected = [top_lane]
            if top_probability < float(lane_threshold):
                fallback = True

            # V3.1 removes the old `onset < 0.72 => single only` rule. Every
            # additional lane is decided by its own evidence. SR only nudges the
            # required probability; it never makes 3/4-key chords impossible.
            for rank, lane in enumerate(ranked[1:], start=2):
                if len(selected) >= chord_cap:
                    break
                threshold = _soft_chord_threshold(
                    rank,
                    top_probability=top_probability,
                    lane_threshold=float(lane_threshold),
                    stars=float(stars),
                )
                if float(key_probability[lane].item()) >= threshold:
                    selected.append(lane)

            for lane in selected:
                tap_logit = lane_logits[lane, TAP]
                # Keep only a mild bias against starting an LN. The previous
                # +1.25-logit requirement suppressed almost every learned hold.
                start_logit = lane_logits[lane, LN_START] - float(ln_start_margin)
                next_state[lane] = LN_START if start_logit > tap_logit else TAP

    return next_state, {
        "policy_version": 31,
        "onset_probability": onset_probability,
        "onset_gate": onset_probability >= gate_threshold,
        "onset_threshold": gate_threshold,
        "selected_keydowns": len(selected),
        "release_count": release_count,
        "lane_fallback": fallback,
        "max_chord": chord_cap,
        "ln_start_margin": float(ln_start_margin),
    }


def _relabel_output(result: dict, *, full_song: bool) -> dict:
    """Relabel V3 output so A/B comparisons cannot mix policy versions."""
    result = dict(result)
    result["decoder"] = "adaptive_v3_1_full" if full_song else "adaptive_v3_1"
    output_value = result.get("output")
    if not output_value:
        return result

    path = Path(output_value)
    if not path.is_file():
        return result

    text = path.read_text(encoding="utf-8-sig")
    if full_song:
        text = text.replace("Star Full Song V3", "Star Full Song V3.1")
        text = text.replace("adaptive full-song decoder v3", "adaptive full-song decoder v3.1")
    else:
        text = text.replace("Star Preview V3", "Star Preview V3.1")
        text = text.replace("adaptive decoder v3", "adaptive decoder v3.1")
    path.write_text(text, encoding="utf-8-sig")

    new_name = path.name.replace(" V3]", " V3.1]")
    new_path = path.with_name(new_name)
    if new_path != path:
        if new_path.exists():
            new_path.unlink()
        path.replace(new_path)
        result["output"] = str(new_path.resolve())
    return result


def generate_preview(**kwargs) -> dict:
    """Run the proven V3 onset system with the relaxed V3.1 pattern policy."""
    _v3.constrained_tick_states = constrained_tick_states_v31
    kwargs = dict(kwargs)
    kwargs.setdefault("ln_start_margin", DEFAULT_LN_START_MARGIN)
    # max_chord=0 now means no artificial SR cap; physical 4K is the only cap.
    if int(kwargs.get("max_chord", 0)) <= 0:
        kwargs["max_chord"] = DEFAULT_AUTO_MAX_CHORD
    return _relabel_output(_v3.generate_preview(**kwargs), full_song=False)


def generate_full_song(**kwargs) -> dict:
    """Run full-song V3 windows with V3.1 lane/LN policy."""
    _full.constrained_tick_states = constrained_tick_states_v31
    kwargs = dict(kwargs)
    kwargs.setdefault("ln_start_margin", DEFAULT_LN_START_MARGIN)
    if int(kwargs.get("max_chord", 0)) <= 0:
        kwargs["max_chord"] = DEFAULT_AUTO_MAX_CHORD
    return _relabel_output(_full.generate_full_song(**kwargs), full_song=True)
