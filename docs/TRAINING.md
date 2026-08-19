# ORBIT-4K V0 Training Guide

## 1. Install and start the local Lab

```powershell
git pull
pip install -e ".[dev]"
python scripts/start_web.py
```

Open `http://127.0.0.1:8765/`.

The Lab server is intentionally bound to localhost. Folder paths entered in the page are resolved by the local Python backend, not uploaded through the browser.

## 2. Prepare the raw osu! Songs folder

The source can be an osu! `Songs` directory, a single beatmap-set directory, or a zip. The builder scans everything but only accepts osu!mania 4K (`Mode: 3`, `CircleSize: 4`).

V0 additionally rejects unsupported timing such as variable uninherited BPM, non-4/4 maps, and maps whose hit objects do not fit the configured 1/96 grid closely enough.

Example:

```text
Source: D:\osu!\Songs
Output: E:\ORBIT4K_DATA\v0
Config: configs/v0.yaml
```

Click **开始清洗**. The page displays the scan stage, accepted/rejected counts, current chart progress, unique-audio count and subprocess logs.

## 3. What preprocessing writes

The MP3 is not converted to text and is not copied into the processed dataset.

Each unique audio file is decoded once and cached as a compressed `.npz` containing numerical arrays such as:

- frame-level 128-bin raw log-Mel (`float16`)
- per-song Mel mean / standard deviation
- total / low / mid / high log-energy frames
- robust song-energy statistics
- audio feature version and timing metadata

At training time, the Dataset combines this cache with each chart's BPM and offset to build a 520-dimensional beat-synchronous audio token for every 1/96 chart tick.

Each accepted `.osu` is converted into a compressed chart array `[T, 4]`, with per-lane states:

```text
0 EMPTY
1 TAP
2 LN_START
3 LN_END
```

The processed folder contains:

```text
v0/
├─ audio/        # numerical audio caches (.npz)
├─ charts/       # [T,4] chart arrays (.npz)
├─ index.jsonl   # text metadata/index
├─ rejected.json
└─ summary.json
```

Train/validation/test assignment is grouped by audio hash. Multiple difficulties using the same song never cross splits.

## 4. Start training from the Lab

After preparation finishes, use the Training Control section.

Example:

```text
Dataset: E:\ORBIT4K_DATA\v0
Run dir: E:\ORBIT4K_RUNS\v0
Config: configs/v0.yaml
```

Click **开始训练**. The Lab launches `scripts/train_v0.py` as a separate Python process and displays:

- process state
- epoch
- train loss
- validation loss
- learning rate
- loss curve
- stdout/stderr log

Checkpoints and metrics are written to the selected run directory:

```text
best.pt
last.pt
metrics.jsonl
```

The Stop button terminates the active training process. Checkpoints already written by completed epochs remain on disk. V0 does not yet implement resume-from-last as the default workflow, so stopping mid-run should currently be treated as ending that run.

## 5. Recommended first run

Before committing a very large Songs library, first prepare a smaller representative subset and verify:

1. accepted/rejected reasons look sane;
2. `missing_star_ratings` is zero;
3. train/validation/test are all non-empty;
4. `pytest -q` passes;
5. CUDA is visible in the Lab;
6. a short training run decreases both state and onset losses.

Then prepare and train on the full dataset.

## 6. Disk-space note

Audio Feature V2 stores high-time-resolution (~5 ms hop) 128-bin `float16` frame caches. Processed data can therefore be comparable to, or larger than, the original compressed MP3 payload even though `.npz` compression is used. Keep the processed dataset and checkpoints on a drive with generous free space, especially for multi-gigabyte osu! libraries.
