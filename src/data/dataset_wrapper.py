from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_to_index(label: int) -> int:
    if label == -1:
        return 0
    if 0 <= label <= 99:
        return label + 1
    raise ValueError(f"Invalid jersey label {label}. Expected -1 or 0..99.")


def index_to_label(index: int) -> int:
    if index == 0:
        return -1
    if 1 <= index <= 100:
        return index - 1
    raise ValueError(f"Invalid class index {index}. Expected 0..100.")


def build_frame_transform(image_size: int, train: bool) -> Callable[[Image.Image], torch.Tensor]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return T.Compose(
            [
                T.Resize((image_size + 16, image_size + 16)),
                T.RandomCrop((image_size, image_size)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])


def _gt_filename_for_split(split: str) -> Optional[str]:
    if split == "train":
        return "train_gt.json"
    if split == "test":
        return "test_gt.json"
    if split == "challenge":
        return None
    # Non-standard split directories can still use a train-style json file.
    return f"{split}_gt.json"


def _list_frame_paths(tracklet_dir: Path) -> List[Path]:
    files = [p for p in tracklet_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(files, key=lambda p: p.name)


def _sample_indices(num_available: int, num_frames: int) -> List[Optional[int]]:
    if num_available <= 0:
        return [None] * num_frames
    if num_available >= num_frames:
        return [int(round(x)) for x in torch.linspace(0, num_available - 1, steps=num_frames).tolist()]
    indices: List[Optional[int]] = list(range(num_available))
    indices.extend([None] * (num_frames - num_available))
    return indices


class SoccerNetTrackletDataset(Dataset):
    """
    Returns tuples with:
      - clip tensor: [T, 3, H, W]
      - label index: int in [0, 100]
      - tracklet id: str
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        num_frames: int = 8,
        image_size: int = 224,
        train: bool = False,
        tracklet_ids: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.transform = transform or build_frame_transform(image_size=image_size, train=train)

        split_dir = self.data_root / split
        self.image_root = split_dir / "images"
        if not self.image_root.exists():
            raise FileNotFoundError(f"Missing image root: {self.image_root}")

        gt_filename = _gt_filename_for_split(split)
        gt_path = split_dir / gt_filename if gt_filename else None
        self.labels = self._load_labels(gt_path) if gt_path and gt_path.exists() else {}

        if tracklet_ids is not None:
            selected_ids = [str(x) for x in tracklet_ids]
        else:
            selected_ids = sorted([p.name for p in self.image_root.iterdir() if p.is_dir()])

        self.samples: List[Tuple[str, List[Path], int]] = []
        for tracklet_id in selected_ids:
            tracklet_dir = self.image_root / tracklet_id
            if not tracklet_dir.exists() or not tracklet_dir.is_dir():
                continue
            frame_paths = _list_frame_paths(tracklet_dir)
            if not frame_paths:
                continue
            label = int(self.labels.get(tracklet_id, -1))
            self.samples.append((tracklet_id, frame_paths, label))

        if not self.samples:
            raise ValueError(f"No usable samples found in {self.image_root}")

        self.tracklet_ids = [sample[0] for sample in self.samples]

    @staticmethod
    def _load_labels(gt_path: Path) -> Dict[str, int]:
        with gt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): int(v) for k, v in data.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        tracklet_id, frame_paths, label = self.samples[idx]
        frame_indices = _sample_indices(len(frame_paths), self.num_frames)
        frames: List[torch.Tensor] = []
        pad_frame = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        for frame_idx in frame_indices:
            if frame_idx is None:
                frames.append(pad_frame.clone())
                continue

            frame_path = frame_paths[frame_idx]
            try:
                with Image.open(frame_path) as img:
                    img = img.convert("RGB")
                    frame_tensor = self.transform(img)
                frames.append(frame_tensor)
            except (FileNotFoundError, OSError):
                # Missing or unreadable image files are represented as all-zero frames.
                frames.append(pad_frame.clone())

        clip_tensor = torch.stack(frames, dim=0)  # [T, 3, H, W]
        label_idx = torch.tensor(label_to_index(label), dtype=torch.long)
        return clip_tensor, label_idx, tracklet_id
