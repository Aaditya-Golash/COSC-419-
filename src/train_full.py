from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_wrapper import SoccerNetTrackletDataset, index_to_label
from models.baseline_resnet import BaselineResNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_ids(tracklet_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(tracklet_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_size = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:val_size]
    train_ids = ids[val_size:]
    if not train_ids:
        train_ids = val_ids
    return train_ids, val_ids


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for clips, labels, _ in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            logits = model(clips)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, labels)
            total_batches += 1
    if total_batches == 0:
        return 0.0, 0.0
    return total_loss / total_batches, total_acc / total_batches


def predict_split(
    model: torch.nn.Module,
    data_root: str,
    split: str,
    num_frames: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Dict[str, int]:
    dataset = SoccerNetTrackletDataset(
        data_root=data_root,
        split=split,
        num_frames=num_frames,
        image_size=image_size,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    predictions: Dict[str, int] = {}
    with torch.no_grad():
        for clips, _, tracklet_ids in tqdm(loader, desc=f"Predicting {split}", leave=False):
            clips = clips.to(device)
            logits = model(clips)
            class_indices = torch.argmax(logits, dim=1).cpu().tolist()
            for tid, idx in zip(tracklet_ids, class_indices):
                predictions[str(tid)] = int(index_to_label(int(idx)))
    return predictions


def compute_test_accuracy(data_root: str | Path, predictions: Dict[str, int]) -> float | None:
    gt_path = Path(data_root) / "test" / "test_gt.json"
    if not gt_path.exists():
        return None
    with gt_path.open("r", encoding="utf-8") as f:
        gt = {str(k): int(v) for k, v in json.load(f).items()}

    keys = list(gt.keys())
    if not keys:
        return None
    correct = 0
    for key in keys:
        pred = predictions.get(key, None)
        if pred is not None and int(pred) == int(gt[key]):
            correct += 1
    return correct / len(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training loop for Group 9 baseline.")
    parser.add_argument("--config", required=True, help="Path to training yaml config")
    parser.add_argument("--data_root", default=None, help="Override data root path")
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg.get("output", {})

    data_root = args.data_root or data_cfg["root"]
    output_dir = Path(args.output_dir or out_cfg.get("dir", "outputs/group9_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    requested_device = str(train_cfg.get("device", "cuda"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    num_frames = int(data_cfg.get("num_frames", 8))
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))

    full_train_dataset = SoccerNetTrackletDataset(
        data_root=data_root,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
        train=True,
    )
    train_ids, val_ids = split_ids(full_train_dataset.tracklet_ids, val_ratio=val_ratio, seed=seed)
    train_dataset = SoccerNetTrackletDataset(
        data_root=data_root,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
        train=True,
        tracklet_ids=train_ids,
    )
    val_dataset = SoccerNetTrackletDataset(
        data_root=data_root,
        split="train",
        num_frames=num_frames,
        image_size=image_size,
        train=False,
        tracklet_ids=val_ids,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = BaselineResNet(
        num_classes=int(model_cfg.get("num_classes", 101)),
        backbone=str(model_cfg.get("backbone", "resnet18")),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 8))
    best_val_acc = -1.0
    best_path = output_dir / "best.ckpt"
    last_path = output_dir / "last.ckpt"

    print(f"Training on device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batches = 0

        for clips, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(clips)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += accuracy_from_logits(logits, labels)
            train_batches += 1

        train_loss = train_loss_sum / max(1, train_batches)
        train_acc = train_acc_sum / max(1, train_batches)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": cfg,
        }
        torch.save(checkpoint, last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_path)
            print(f"Saved new best checkpoint to {best_path}")

    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    test_predictions = predict_split(
        model=model,
        data_root=data_root,
        split="test",
        num_frames=num_frames,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    pred_path = output_dir / "predictions_test.json"
    with pred_path.open("w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=2)
    print(f"Saved test predictions: {pred_path}")

    test_acc = compute_test_accuracy(data_root=data_root, predictions=test_predictions)
    if test_acc is not None:
        print(f"Local test accuracy: {test_acc:.4f}")
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({"best_val_acc": best_val_acc, "test_accuracy": test_acc}, f, indent=2)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
