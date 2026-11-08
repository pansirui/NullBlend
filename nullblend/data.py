"""Load the three canonical NullBlend datasets from a uniform layout."""
from __future__ import annotations

from pathlib import Path

import torch


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. The release expects observed.pt and reference_clean.pt "
            "under data/<dataset>/."
        )
    return path


def load_observed_dataset(data_dir: Path) -> dict:
    """Load only information available to the self-supervised trainer."""
    observed_bundle = torch.load(
        _require(data_dir / "observed.pt"), map_location="cpu", weights_only=False
    )
    required = {"continuous", "dithers", "meta"}
    missing = required.difference(observed_bundle)
    if missing:
        raise KeyError(f"observed.pt is missing keys: {sorted(missing)}")
    return {
        "observed": observed_bundle["continuous"].float().contiguous(),
        "shot_starts": observed_bundle["dithers"].long().contiguous(),
        "pseudo_csg": observed_bundle.get("pseudo_csg"),
        "metadata": observed_bundle["meta"],
    }


def load_evaluation_reference(data_dir: Path) -> torch.Tensor:
    """Load clean data after training, solely for metrics and figures."""
    bundle = torch.load(
        _require(data_dir / "reference_clean.pt"),
        map_location="cpu",
        weights_only=False,
    )
    if "clean" not in bundle:
        raise KeyError("reference_clean.pt is missing key 'clean'")
    return bundle["clean"].float().contiguous()


def csg_to_crg(csg: torch.Tensor) -> torch.Tensor:
    """Convert [shot, receiver, time] to [receiver, shot, time]."""
    return csg.permute(1, 0, 2).contiguous()


def crg_to_csg(crg: torch.Tensor) -> torch.Tensor:
    """Convert [receiver, shot, time] to [shot, receiver, time]."""
    return crg.permute(1, 0, 2).contiguous()
