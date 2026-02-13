from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models.baseline_resnet import BaselineResNet


def test_forward_shape() -> None:
    model = BaselineResNet(num_classes=101, backbone="resnet18", pretrained=False)
    clips = torch.randn(2, 8, 3, 224, 224)
    logits = model(clips)
    assert logits.shape == (2, 101)
