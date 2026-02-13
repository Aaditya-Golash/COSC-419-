from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset_wrapper import SoccerNetTrackletDataset
from models.baseline_resnet import BaselineResNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def grad_norm(model: torch.nn.Module) -> float:
    sq_sum = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().data.norm(2).item()
        sq_sum += value * value
    return math.sqrt(sq_sum)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity train loop for shape/backprop checks.")
    parser.add_argument("--config", type=str, required=True, help="Path to sanity yaml config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device(train_cfg.get("device", "cpu"))

    dataset = SoccerNetTrackletDataset(
        data_root=data_cfg["root"],
        split=data_cfg.get("split", "train"),
        num_frames=int(data_cfg.get("num_frames", 8)),
        image_size=int(data_cfg.get("image_size", 224)),
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = BaselineResNet(
        num_classes=int(model_cfg.get("num_classes", 101)),
        backbone=model_cfg.get("backbone", "resnet18"),
        pretrained=bool(model_cfg.get("pretrained", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    max_steps = int(train_cfg.get("max_steps", 3))
    completed_steps = 0

    print("Running sanity loop...")
    for step, batch in enumerate(loader):
        clips, labels, tracklet_ids = batch
        if step == 0:
            print(f"Input tensor shape: {tuple(clips.shape)}")
            print(f"Label tensor shape: {tuple(labels.shape)}")
            print(f"Example tracklet id: {tracklet_ids[0]}")

        clips = clips.to(device)
        labels = labels.to(device)

        logits = model(clips)
        if step == 0:
            print(f"Logits shape: {tuple(logits.shape)}")

        loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss is not finite at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(model)
        optimizer.step()

        print(f"Step {step+1}: loss={loss.item():.6f}, grad_norm={gnorm:.6f}")
        completed_steps += 1
        if completed_steps >= max_steps:
            break

    if completed_steps < 2:
        raise RuntimeError(
            f"Sanity run completed only {completed_steps} step(s). "
            "Need at least 2 mini-batches for your meeting proof."
        )

    print("Sanity run complete.")


if __name__ == "__main__":
    main()
