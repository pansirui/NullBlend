"""Dataset locations and the frozen configuration used in the paper."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
DATASETS = ("marmousi", "mobil", "seam")


@dataclass(frozen=True)
class NullBlendConfig:
    dataset: str
    device: str = "cuda:0"
    epochs: int = 200
    receiver_batch_size: int = 1
    learning_rate: float = 2e-4
    tafa_learning_rate: float = 5e-6
    output_head_learning_rate: float = 5e-6
    hidden_width: int = 32
    hierarchy_depth: int = 4
    wavelet_levels: int = 3
    receiver_radius: int = 2
    maximum_overlap: int = 4
    teboot_update_period: int = 8
    teboot_new_prediction_weight: float = 1.0
    seed: int = 20260818
    shot_shifts: tuple[int, ...] = (1, -1, 2, -2, 4, -4, 8, -8)

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.epochs <= 0 or self.receiver_batch_size <= 0:
            raise ValueError("epochs and receiver_batch_size must be positive")
        if self.teboot_update_period <= 0:
            raise ValueError("teboot_update_period must be positive")
        if self.teboot_new_prediction_weight != 1.0 or self.teboot_update_period != 8:
            raise ValueError("Teacher must be replaced directly every eight epochs")
        if (self.hidden_width,self.hierarchy_depth,self.wavelet_levels,self.receiver_radius)!=(32,4,3,2):
            raise ValueError("The release architecture uses width 32, four stages, three wavelet levels and radius two")

    @property
    def dataset_dir(self) -> Path:
        return DATA_ROOT / self.dataset

    def as_dict(self) -> dict:
        value = asdict(self)
        value["shot_shifts"] = list(self.shot_shifts)
        return value


REFERENCE_SNR_DB = {'mobil': 26.76445770263672, 'seam': 22.93248748779297, 'marmousi': 30.255144119262695}
