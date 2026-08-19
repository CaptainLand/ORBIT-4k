from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import numpy as np

EMPTY = 0
TAP = 1
LN_START = 2
LN_END = 3


@dataclass(frozen=True)
class TimingPoint:
    time_ms: float
    beat_length_ms: float
    meter: int

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass
class ParsedBeatmap:
    path: Path
    audio_path: Path
    title: str
    artist: str
    version: str
    creator: str
    mode: int
    keys: int
    timing_points: list[TimingPoint]
    chart: np.ndarray
    total_ticks: int
    quantization_errors_ms: list[float]
    object_count: int
    star_rating: float | None = None

    @property
    def offset_ms(self) -> float:
        return self.timing_points[0].time_ms

    @property
    def beat_length_ms(self) -> float:
        return self.timing_points[0].beat_length_ms

    @property
    def bpm(self) -> float:
        return self.timing_points[0].bpm

    @property
    def p95_quantization_error_ms(self) -> float:
        if not self.quantization_errors_ms:
            return 0.0
        return float(np.percentile(np.asarray(self.quantization_errors_ms), 95))


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is not None and line and not line.startswith("//"):
            sections[current].append(line)
    return sections


def _key_values(lines: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def _lane_from_x(x: int, keys: int = 4) -> int:
    # osu!mania lanes partition the [0, 512) x coordinate range evenly.
    return min(keys - 1, max(0, int(x * keys / 512)))


def _quantize_ms(time_ms: float, offset_ms: float, beat_length_ms: float, ticks_per_quarter: int) -> tuple[int, float]:
    tick_ms = beat_length_ms / ticks_per_quarter
    tick = int(round((time_ms - offset_ms) / tick_ms))
    reconstructed = offset_ms + tick * tick_ms
    return tick, abs(float(time_ms) - reconstructed)


def _star_rating(path: Path) -> float | None:
    try:
        import rosu_pp_py as rosu

        beatmap = rosu.Beatmap(path=str(path))
        if beatmap.is_suspicious():
            return None
        attrs = rosu.Difficulty().calculate(beatmap)
        return float(attrs.stars)
    except Exception:
        return None


def parse_osu_file(
    path: str | Path,
    *,
    ticks_per_quarter: int = 24,
    require_4k: bool = True,
    constant_bpm_only: bool = True,
    calculate_stars: bool = True,
) -> ParsedBeatmap:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    sections = _sections(text)
    general = _key_values(sections.get("General", []))
    difficulty = _key_values(sections.get("Difficulty", []))
    metadata = _key_values(sections.get("Metadata", []))

    mode = int(float(general.get("Mode", "0")))
    keys = int(round(float(difficulty.get("CircleSize", "0"))))
    if require_4k and not (mode == 3 and keys == 4):
        raise ValueError(f"not osu!mania 4K (Mode={mode}, CircleSize={keys})")

    timing_points: list[TimingPoint] = []
    for line in sections.get("TimingPoints", []):
        parts = line.split(",")
        if len(parts) < 7:
            continue
        uninherited = int(float(parts[6])) == 1
        beat_length = float(parts[1])
        if uninherited and beat_length > 0:
            timing_points.append(TimingPoint(float(parts[0]), beat_length, int(float(parts[2]))))
    if not timing_points:
        raise ValueError("no uninherited timing point found")
    timing_points.sort(key=lambda item: item.time_ms)
    if constant_bpm_only and len(timing_points) != 1:
        raise ValueError(f"V0 only supports constant BPM maps; found {len(timing_points)} red timing points")
    if timing_points[0].meter != 4:
        raise ValueError(f"V0 only supports 4/4 maps; meter={timing_points[0].meter}")

    timing = timing_points[0]
    events: list[tuple[int, int, int]] = []
    errors: list[float] = []
    max_tick = 0
    object_count = 0
    for line in sections.get("HitObjects", []):
        parts = line.split(",")
        if len(parts) < 6:
            continue
        x = int(parts[0])
        start_ms = float(parts[2])
        hit_type = int(parts[3])
        lane = _lane_from_x(x, keys)
        start_tick, start_error = _quantize_ms(start_ms, timing.time_ms, timing.beat_length_ms, ticks_per_quarter)
        if start_tick < 0:
            continue
        errors.append(start_error)
        object_count += 1
        if hit_type & 128:
            try:
                end_ms = float(parts[5].split(":", 1)[0])
            except ValueError as exc:
                raise ValueError(f"invalid LN end time in {path.name}: {line}") from exc
            end_tick, end_error = _quantize_ms(end_ms, timing.time_ms, timing.beat_length_ms, ticks_per_quarter)
            errors.append(end_error)
            if end_tick <= start_tick:
                end_tick = start_tick + 1
            events.append((start_tick, lane, LN_START))
            events.append((end_tick, lane, LN_END))
            max_tick = max(max_tick, end_tick)
        else:
            events.append((start_tick, lane, TAP))
            max_tick = max(max_tick, start_tick)

    chart = np.zeros((max_tick + 1, 4), dtype=np.uint8)
    collisions: list[tuple[int, int, int, int]] = []
    for tick, lane, state in events:
        previous = int(chart[tick, lane])
        if previous != EMPTY and previous != state:
            collisions.append((tick, lane, previous, state))
            continue
        chart[tick, lane] = state
    if collisions:
        raise ValueError(f"{len(collisions)} same-lane same-tick state collisions; first={collisions[0]}")

    audio_filename = general.get("AudioFilename")
    if not audio_filename:
        raise ValueError("AudioFilename missing")
    audio_path = path.parent / audio_filename
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    return ParsedBeatmap(
        path=path,
        audio_path=audio_path,
        title=metadata.get("TitleUnicode") or metadata.get("Title", path.stem),
        artist=metadata.get("ArtistUnicode") or metadata.get("Artist", "Unknown"),
        version=metadata.get("Version", path.stem),
        creator=metadata.get("Creator", "Unknown"),
        mode=mode,
        keys=keys,
        timing_points=timing_points,
        chart=chart,
        total_ticks=chart.shape[0],
        quantization_errors_ms=errors,
        object_count=object_count,
        star_rating=_star_rating(path) if calculate_stars else None,
    )


def scan_4k_beatmaps(
    root: str | Path,
    *,
    ticks_per_quarter: int = 24,
    constant_bpm_only: bool = True,
    max_quantization_error_ms: float = 8.0,
    calculate_stars: bool = True,
) -> tuple[list[ParsedBeatmap], list[dict[str, str]]]:
    root = Path(root)
    accepted: list[ParsedBeatmap] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.osu")):
        try:
            parsed = parse_osu_file(
                path,
                ticks_per_quarter=ticks_per_quarter,
                require_4k=True,
                constant_bpm_only=constant_bpm_only,
                calculate_stars=calculate_stars,
            )
            if parsed.p95_quantization_error_ms > max_quantization_error_ms:
                raise ValueError(
                    f"p95 quantization error {parsed.p95_quantization_error_ms:.2f}ms > {max_quantization_error_ms:.2f}ms"
                )
            accepted.append(parsed)
        except Exception as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    return accepted, rejected


def stable_audio_id(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
