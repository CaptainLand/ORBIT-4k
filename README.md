# ORBIT-4K

**Audio-conditioned end-to-end Transformer for osu!mania 4K chart generation.**

ORBIT-4K V0 deliberately removes the hard `rhythm planner -> arranger` bottleneck used by ORBIT-8. Given aligned audio, BPM/offset and a target star rating, one model directly predicts the final four-lane chart.

## V0 scope

- osu!mania **4K only**: preprocessing accepts only `Mode: 3` and `CircleSize: 4`.
- Constant BPM and 4/4 only for the first baseline.
- Manual/canonical BPM + offset. Training reads them from `.osu` timing points.
- Grid resolution: **24 ticks per quarter note = 96 ticks per 4/4 measure**.
- Output per lane: `EMPTY`, `TAP`, `LN_START`, `LN_END`.
- Audio: 128-bin normalized Log-Mel features aligned to the beat grid.
- Model: bidirectional Audio Transformer + causal Chart Transformer with cross-attention.
- Difficulty conditioning: osu!mania star rating calculated with `rosu-pp-py`.

## Why preprocessing exists

The raw `.osu` and MP3 are **not** fed directly to the training loop. Dataset preparation turns them into stable numerical training data:

```text
Songs / beatmap-set zip
  -> scan .osu
  -> reject non-mania / non-4K
  -> reject unsupported timing for V0
  -> quantize HitObjects onto the 1/96 grid
  -> cache audio Log-Mel once per unique audio file
  -> save chart matrices [T, 4]
  -> group splits by audio SHA-1
  -> index.jsonl
```

Audio features are cached once per song. They are aligned to each chart's BPM/offset window inside the Dataset, so multiple difficulties can share one audio cache without duplicating the entire spectrogram.

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
├─ audio/       # unique Log-Mel caches
├─ charts/      # uint8 [T,4] chart tensors
├─ index.jsonl  # metadata + split + SR + timing
├─ rejected.json
└─ summary.json
```

**Important:** train/validation/test is assigned by audio SHA-1, not by `.osu` file. All difficulties of one song stay in the same split, preventing audio leakage.

## Train V0

```powershell
python scripts/train_v0.py
```

Default config: `configs/v0.yaml`. The baseline is approximately 29.84M parameters and intentionally uses a simple reference architecture before optimization. Training dynamically samples multiple 16-measure crops per chart each epoch.

Checkpoints are written to:

```text
runs/v0/best.pt
runs/v0/last.pt
```

## Local lab UI

```powershell
python scripts/start_web.py
```

Open `http://127.0.0.1:8765/`.

The V0 page shows the architecture/status and can inspect a beatmap-set ZIP before you add it to a dataset. The ZIP inspection uses the same 4K/timing/quantization gate as the real dataset builder.

## V0 model contract

```text
Audio [B,T,128]
    -> bidirectional Audio Encoder
    -> Audio Memory
                        ↘ cross attention
Previous chart [B,T,4] -> causal Chart Decoder
Active-LN mask [B,T,4] ↗
BPM / SR / beat phase / measure phase ↗
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
