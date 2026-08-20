from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data_v1 import Orbit4KCellDataset
from .losses_v1 import orbit4k_v1_loss
from .model_v1 import Orbit4KV1, parameter_count, warm_start_from_v0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dataset(config: dict, data_root: Path, split: str) -> Orbit4KCellDataset:
    return Orbit4KCellDataset(
        data_root,
        split,
        ticks_per_quarter=config["grid"]["ticks_per_quarter"],
        ticks_per_measure=config["grid"]["ticks_per_measure"],
        micro_ticks_per_cell=config["grid"]["micro_ticks_per_cell"],
        window_measures=config["window"]["measures"],
        stride_measures=config["window"]["stride_measures"],
        random_crop=split == "train",
        random_samples_per_chart=int(config["window"].get("train_samples_per_chart", 4)),
        audio_cache_size=int(config.get("dataset", {}).get("audio_cache_size", 1)),
    )


def _gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 1024**2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }


def run_epoch(model, loader, optimizer, scaler, scheduler, config, device, training: bool) -> dict[str, float]:
    model.train(training)
    accum = int(config["training"]["gradient_accumulation"])
    class_weights = torch.tensor(
        config["training"]["class_weights"],
        device=device,
        dtype=torch.float32,
    )
    totals: dict[str, float] = {}
    count = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    autocast_enabled = bool(config["training"]["amp"]) and device.type == "cuda"

    for step, batch in enumerate(loader, 1):
        tensor_batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                outputs = model(
                    tensor_batch["audio"],
                    tensor_batch["chart_input"],
                    tensor_batch["active_ln"],
                    tensor_batch["cell_tick"],
                    tensor_batch["bpm"],
                    tensor_batch["stars"],
                    tensor_batch["mask"],
                )
                loss, metrics = orbit4k_v1_loss(
                    outputs,
                    tensor_batch["target"],
                    tensor_batch["mask"],
                    tensor_batch["micro_mask"],
                    class_weights=class_weights,
                    cell_onset_weight=float(config["training"]["cell_onset_weight"]),
                    micro_onset_weight=float(config["training"]["micro_onset_weight"]),
                    density_weight=float(config["training"]["density_weight"]),
                    cell_onset_pos_weight=float(config["training"].get("cell_onset_pos_weight", 2.0)),
                    micro_onset_pos_weight=float(config["training"].get("micro_onset_pos_weight", 2.5)),
                )
                scaled_loss = loss / accum

            if training:
                scaler.scale(scaled_loss).backward()
                if step % accum == 0 or step == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(config["training"]["clip_grad_norm"]),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
        except torch.OutOfMemoryError as exc:
            memory = _gpu_memory(device)
            raise RuntimeError(
                "CUDA OOM in V1 batch. Dataset song count is not the per-step VRAM driver; "
                f"current batch_size={config['training']['batch_size']}, "
                f"window_measures={config['window']['measures']}, memory={memory}. "
                "Reduce batch_size first, then window_measures if needed."
            ) from exc

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1

        # Make tensor lifetimes explicit so future metric/logging changes cannot
        # accidentally retain computation graphs across a large epoch.
        del tensor_batch, outputs, loss, scaled_loss

    return {key: value / max(1, count) for key, value in totals.items()}


def _restore_scheduler_position(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    lr_scale,
) -> None:
    """Restore schedule position for legacy V1 checkpoints without scheduler state."""
    completed_steps = max(0, int(completed_steps))
    scheduler.last_epoch = completed_steps
    scheduler._step_count = completed_steps + 1
    for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
        group["lr"] = float(base_lr) * float(lr_scale(completed_steps))
    scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ORBIT-4K V1 32x3 hierarchical model")
    parser.add_argument("--config", type=Path, default=Path("configs/v1.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/v0"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v1"))
    parser.add_argument(
        "--warm-start-v0",
        type=Path,
        default=None,
        help="optional V0 best.pt; reuses Transformer backbone and compatible embeddings",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="ignore run-dir/last.pt and start a fresh V1 run",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(config, args.data, "train")
    val_ds = build_dataset(config, args.data, "validation")
    if not train_ds.rows:
        raise RuntimeError("training split is empty; prepare a larger dataset")

    workers = int(config["training"]["num_workers"])
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(config.get("dataset", {}).get("prefetch_factor", 1))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds.rows else None

    model = Orbit4KV1(
        audio_token_dim=config["model"]["audio_token_dim"],
        audio_micro_dim=config["model"]["audio_micro_dim"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        audio_layers=config["model"]["audio_layers"],
        chart_layers=config["model"]["chart_layers"],
        micro_layers=config["model"]["micro_layers"],
        micro_dim_feedforward=config["model"]["micro_dim_feedforward"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        local_audio_scale_cells=config["model"]["local_audio_scale_cells"],
        max_cross_attention_bias=config["model"]["max_cross_attention_bias"],
    )

    resume_checkpoint = None
    resume_path = args.run_dir / "last.pt"
    start_epoch = 1
    best = float("inf")
    warm_start_report = None

    if resume_path.is_file() and not args.restart:
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("architecture") != "v1_32x3":
            raise RuntimeError(f"cannot resume non-V1 checkpoint: {resume_path}")
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best = float(resume_checkpoint.get("best_score", resume_checkpoint.get("score", float("inf"))))
        warm_start_report = resume_checkpoint.get("warm_start")
        print(
            "resume="
            + json.dumps(
                {
                    "path": str(resume_path),
                    "completed_epoch": start_epoch - 1,
                    "next_epoch": start_epoch,
                    "optimizer_state": "optimizer" in resume_checkpoint,
                    "scheduler_state": "scheduler" in resume_checkpoint,
                },
                ensure_ascii=True,
            )
        )
    elif args.warm_start_v0 is not None:
        if not args.warm_start_v0.is_file():
            raise FileNotFoundError(f"V0 warm-start checkpoint not found: {args.warm_start_v0}")
        warm_start_report = warm_start_from_v0(model, args.warm_start_v0)
        print("warm_start=" + json.dumps(warm_start_report, ensure_ascii=True))

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and bool(config["training"]["amp"]),
    )
    optimizer_steps_per_epoch = max(
        1,
        math.ceil(len(train_loader) / int(config["training"]["gradient_accumulation"])),
    )
    total_steps = optimizer_steps_per_epoch * int(config["training"]["epochs"])
    warmup_steps = min(
        int(config["training"].get("warmup_steps", 0)),
        max(0, total_steps - 1),
    )

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-4, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    if resume_checkpoint is not None:
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scaler" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        else:
            # The first V1 checkpoints did not yet save optimizer/scheduler state.
            # Their model weights are still valuable; infer the scheduler position
            # from the number of completed full epochs instead of restarting warmup.
            _restore_scheduler_position(
                scheduler,
                optimizer,
                (start_epoch - 1) * optimizer_steps_per_epoch,
                lr_scale,
            )

    print(
        f"device={device} parameters={parameter_count(model):,} "
        f"train_charts={len(train_ds.rows)} train_samples={len(train_ds)} "
        f"cells_per_window={train_ds.window_cells} micro_slots=3"
    )

    total_epochs = int(config["training"]["epochs"])
    if start_epoch > total_epochs:
        print(json.dumps({"stage": "complete", "epoch": start_epoch - 1, "message": "run already complete"}))
        return

    for epoch in range(start_epoch, total_epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            scheduler,
            config,
            device,
            True,
        )
        train_memory = _gpu_memory(device)

        val_metrics = {}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    None,
                    scaler,
                    None,
                    config,
                    device,
                    False,
                )

        score = val_metrics.get("loss", train_metrics["loss"])
        is_best = score < best
        if is_best:
            best = score
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
            "parameters": parameter_count(model),
            "gpu_memory": train_memory,
            "architecture": "v1_32x3",
        }
        print(json.dumps(record, ensure_ascii=True))
        with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "score": score,
            "best_score": best,
            "architecture": "v1_32x3",
            "warm_start": warm_start_report,
        }
        torch.save(checkpoint, args.run_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, args.run_dir / "best.pt")


if __name__ == "__main__":
    main()
