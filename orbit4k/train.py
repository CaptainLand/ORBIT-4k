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

from .data import Orbit4KDataset
from .losses import orbit4k_loss
from .model import Orbit4KV0, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_dataset(config: dict, data_root: Path, split: str) -> Orbit4KDataset:
    return Orbit4KDataset(
        data_root,
        split,
        ticks_per_quarter=config["grid"]["ticks_per_quarter"],
        ticks_per_measure=config["grid"]["ticks_per_measure"],
        window_measures=config["window"]["measures"],
        stride_measures=config["window"]["stride_measures"],
        random_crop=split == "train",
        random_samples_per_chart=int(config["window"].get("train_samples_per_chart", 4)),
    )


def run_epoch(model, loader, optimizer, scaler, scheduler, config, device, training: bool) -> dict[str, float]:
    model.train(training)
    accum = int(config["training"]["gradient_accumulation"])
    class_weights = torch.tensor(config["training"]["class_weights"], device=device, dtype=torch.float32)
    totals: dict[str, float] = {}
    count = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    autocast_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    for step, batch in enumerate(loader, 1):
        tensor_batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(
                tensor_batch["audio"],
                tensor_batch["chart_input"],
                tensor_batch["active_ln"],
                tensor_batch["tick"],
                tensor_batch["bpm"],
                tensor_batch["stars"],
                tensor_batch["mask"],
            )
            loss, metrics = orbit4k_loss(
                outputs,
                tensor_batch["target"],
                tensor_batch["mask"],
                class_weights=class_weights,
                onset_weight=float(config["training"]["onset_weight"]),
                density_weight=float(config["training"]["density_weight"]),
            )
            scaled_loss = loss / accum
        if training:
            scaler.scale(scaled_loss).backward()
            if step % accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["clip_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ORBIT-4K V0")
    parser.add_argument("--config", type=Path, default=Path("configs/v0.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/v0"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v0"))
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_dataset(config, args.data, "train")
    val_ds = build_dataset(config, args.data, "validation")
    if not train_ds.rows:
        raise RuntimeError("training split is empty; prepare a larger dataset")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    ) if val_ds.rows else None

    model = Orbit4KV0(
        audio_token_dim=config["model"]["audio_token_dim"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        audio_layers=config["model"]["audio_layers"],
        chart_layers=config["model"]["chart_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        local_audio_scale_ticks=config["model"]["local_audio_scale_ticks"],
        max_cross_attention_bias=config["model"]["max_cross_attention_bias"],
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"]["amp"]))
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / int(config["training"]["gradient_accumulation"])))
    total_steps = optimizer_steps_per_epoch * int(config["training"]["epochs"])
    warmup_steps = min(int(config["training"].get("warmup_steps", 0)), max(0, total_steps - 1))

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-4, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} parameters={parameter_count(model):,} train_charts={len(train_ds.rows)}")

    best = float("inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, scheduler, config, device, True)
        val_metrics = {}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, None, scaler, None, config, device, False)
        score = val_metrics.get("loss", train_metrics["loss"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
            "parameters": parameter_count(model),
        }
        print(json.dumps(record))
        with (args.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch, "score": score}
        torch.save(checkpoint, args.run_dir / "last.pt")
        if score < best:
            best = score
            torch.save(checkpoint, args.run_dir / "best.pt")


if __name__ == "__main__":
    main()
