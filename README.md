# ORBIT-4K

**Audio-conditioned end-to-end Transformer for osu!mania 4K chart generation.**

ORBIT-4K V0 deliberately removes the hard `rhythm planner -> arranger` bottleneck used by ORBIT-8. Given aligned audio, BPM/offset and a target star rating, one model directly predicts the final four-lane chart.

## V0 scope

- osu!mania **4K only**: preprocessing accepts only `Mode: 3` and `CircleSize: 4`.
- Constant BPM and 4/4 only for the first baseline.
- Manual/canonical BPM + offset. Training reads them from `.osu` timing points.
- Grid resolution: **24 ticks per quarter note = 96 ticks per 4/4 measure**.
- Output per lane: `EMPTY`, `TAP`, `LN_START`, `LN_END`.
- Difficulty conditioning: osu!mania star rating calculated with `rosu-pp-py`.
- Model: bidirectional Audio Transformer + causal Chart Transformer with cross-attention.

## Beat-synchronous audio tokenization

V0 audio feature version 2 no longer samples one Mel frame at the center of a chart tick.

For every 1/96 grid cell, ORBIT-4K treats the tick as a short musical interval. Five sub-samples cover the full interval from **-1/192 to +1/192 of a whole note** around the tick center and are pooled with a center-weighted kernel.

Each chart tick receives a **520-dimensional raw audio token**:

```text
128  raw local log-Mel
128  song-normalized local log-Mel
128  positive spectral flux
128  local-minus-context spectral contrast
  8  absolute + song-relative energy/dynamic scalars
---
520  -> learned projection -> 384-d model hidden
```

The context spectrum spans roughly +/-12 grid ticks (about +/-1/8 note). This helps distinguish a real local accent from a section that is simply loud all the time.

### Why both raw and normalized audio?

Only using raw audio makes chart density vulnerable to mastering level: a quietly exported song can look artificially unimportant. Only using song-normalized audio throws away useful real dynamics and broad timbral balance.

ORBIT-4K therefore exposes both:

- **raw view**: preserves cross-song level/timbre information;
- **song-normalized view**: makes musical structure comparable between quiet and loud masters;
- **flux / contrast**: highlights onsets and locally salient changes;
- **energy branch**: preserves total/low/mid/high energy plus song-relative loudness.

The waveform is no longer peak-normalized before feature extraction.

## Time alignment inside the model

Data and model use the same musical tick coordinate.

```text
AudioToken[t] <-> ChartToken[t]
```

V0 reinforces this correspondence in three ways:

1. audio and chart streams receive the same absolute sinusoidal tick position;
2. the encoded `AudioMemory[t]` is injected directly into `ChartHidden[t]`;
3. cross-attention keeps the whole audio window visible but applies a soft distance bias favoring nearby ticks.

This means local timing is explicit while section-level/global musical context is still available.

## Why preprocessing exists

The raw `.osu` and MP3 are **not** fed directly to the training loop. Dataset preparation turns them into stable numerical training data:

```text
Songs / beatmap-set zip
  -> scan .osu
  -> reject non-mania / non-4K
  -> reject unsupported timing for V0
  -> quantize HitObjects onto the 1/96 grid
  -> cache high-resolution song audio features once per unique audio file
  -> save chart matrices [T, 4]
  -> group splits by audio SHA-1
  -> index.jsonl
```

Audio caches keep frame-level information. The Dataset converts those frames into beat-synchronous 520-d tokens using each chart's BPM/offset, so multiple difficulties can share one song cache.

## Install

Python 3.11+ is recommended.

```powershell
pip install -e .
```

For development/tests:

```powershell
pip install -e ".[dev]"
```

## Prepare a dataset

Point the script at an osu! `Songs` directory, one beatmap-set directory, or a `.zip`:

```powershell
python scripts/prepare_dataset.py "D:\osu!\Songs"
```

Output:

```text
data/processed/v0/
├─ audio/       # unique frame-level audio caches
├─ charts/      # uint8 [T,4] chart tensors
├─ index.jsonl
├─ rejected.json
└─ summary.json
```

**Important:** audio feature version 2 changed the cache format. Running `prepare_dataset.py` automatically rebuilds old V1 audio caches when encountered.

Train/validation/test is assigned by audio SHA-1, not by `.osu` file. All difficulties of one song stay in the same split, preventing audio leakage.

## Train V0

```powershell
python scripts/train_v0.py
```

Default config: `configs/v0.yaml`. Training dynamically samples multiple 16-measure crops per chart each epoch.

Checkpoints:

```text
runs/v0/best.pt
runs/v0/last.pt
```

## Local lab UI

```powershell
python scripts/start_web.py
```

Open `http://127.0.0.1:8765/`.

The V0 page shows the beat-synchronous audio architecture/status and can inspect a beatmap-set ZIP before adding it to the dataset.

## V0 model contract

```text
Beat audio [B,T,520]
    -> spectral branch (512) + energy branch (8)
    -> shared absolute tick position
    -> bidirectional Audio Encoder
    -> Audio Memory
                  |\
                  | \ same-tick direct fusion
                  |  \
Previous chart -> causal Chart Decoder
Active-LN mask ->      + global cross-attention with soft local bias
BPM / SR / beat phase / measure phase
    -> logits [B,T,4,4]
```

Training uses teacher forcing plus three losses:

1. weighted per-lane state cross-entropy;
2. auxiliary per-tick onset loss;
3. window-level expected density loss.

The onset head is only an auxiliary loss. There is **no separate sampling stage** in the generation architecture.

## Data quality rules in V0

A map is rejected if it is not mania 4K, has no canonical timing point, uses multiple uninherited timing points (variable BPM), is not 4/4, contains an unrepresentable same-lane/same-tick state collision, or has excessive 1/96 quantization error.

This strict baseline is intentional. Once V0 proves the end-to-end approach, later versions can add variable BPM, SV, finer grids, style conditioning and faster autoregressive inference.
