from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.dataset_wrapper import index_to_label, label_to_index


def test_label_roundtrip() -> None:
    for label in [-1, 0, 1, 10, 99]:
        assert index_to_label(label_to_index(label)) == label


def test_mapping_edges() -> None:
    assert label_to_index(-1) == 0
    assert label_to_index(0) == 1
    assert label_to_index(99) == 100
