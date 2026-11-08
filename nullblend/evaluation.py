"""Evaluation-only metrics and figures for NullBlend outputs."""
from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import crg_to_csg, csg_to_crg

FIGURE_DPI = 600


def write_json(path: Path, value: dict) -> None:
    """Atomically write JSON, including retry handling for Windows file locks."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(value, indent=2, ensure_ascii=False)
    temporary.write_text(text, encoding="utf-8")
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05 * (attempt + 1))
    try:
        path.write_text(text, encoding="utf-8")
        if temporary.is_file():
            temporary.unlink(missing_ok=True)
    except Exception as error:
        raise last_error or error


def snr_db(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> float:
    signal = reference.pow(2).mean().clamp_min(eps)
    error = (reference - estimate).pow(2).mean().clamp_min(eps)
    return float(10.0 * torch.log10(signal / error))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def overlap_snr_db(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    overlap_order: torch.Tensor,
    orders: tuple[int, ...] = (1, 2, 3),
) -> dict[str, float]:
    error_squared = (prediction.double() - reference.double()).square()
    reference_squared = reference.double().square()
    order_map = overlap_order.to(device=prediction.device)
    result = {}
    for order in orders:
        mask = order_map == int(order)
        if not bool(mask.any()):
            result[f"order_{order}"] = float("nan")
            continue
        signal = float(reference_squared[:, mask].sum())
        error = float(error_squared[:, mask].sum())
        result[f"order_{order}"] = 10.0 * math.log10(signal / max(error, 1e-30))
    return result


def _display_limit(reference: torch.Tensor) -> float:
    _, upper = torch.quantile(
        reference.detach().float().reshape(-1), torch.tensor([0.05, 0.95])
    )
    return float(max(abs(upper.item()), 1e-12))


def _plot_crg_panel(
    reference: torch.Tensor,
    pseudo: torch.Tensor,
    prediction: torch.Tensor,
    path: Path,
    title: str,
    dt: float,
) -> None:
    panels = [
        ("Reference CRG", reference),
        ("Pseudo-deblend", pseudo),
        ("NullBlend", prediction),
        ("Difference", prediction - reference),
    ]
    limit = _display_limit(reference)
    figure, axes = plt.subplots(1, 4, figsize=(11, 6.5))
    for axis, (name, gather) in zip(axes, panels):
        panel_limit = limit * 0.5 if name == "Difference" else limit
        image = axis.imshow(
            gather.detach().cpu().numpy(),
            aspect="auto",
            cmap="seismic",
            vmin=-panel_limit,
            vmax=panel_limit,
            extent=[0, gather.shape[1], gather.shape[0] * dt, 0],
        )
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("Shot")
        axis.set_ylabel("Time (s)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def _plot_csg_panel(
    reference: torch.Tensor,
    pseudo: torch.Tensor,
    prediction: torch.Tensor,
    path: Path,
    title: str,
    dt: float,
    dx: float,
) -> None:
    panels = [
        ("Reference CSG", reference),
        ("Pseudo-deblend", pseudo),
        ("NullBlend", prediction),
        ("Difference", prediction - reference),
    ]
    limit = _display_limit(reference)
    figure, axes = plt.subplots(1, 4, figsize=(11, 6.5))
    for axis, (name, gather) in zip(axes, panels):
        panel_limit = limit * 0.5 if name == "Difference" else limit
        array = gather.detach().cpu().numpy()
        image = axis.imshow(
            array.T,
            aspect="auto",
            cmap="seismic",
            vmin=-panel_limit,
            vmax=panel_limit,
            extent=[0, array.shape[0] * dx / 1000.0, array.shape[1] * dt, 0],
        )
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("Distance (km)")
        axis.set_ylabel("Time (s)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def _write_snr_tables(
    output_dir: Path,
    reference_crg: torch.Tensor,
    pseudo_crg: torch.Tensor,
    prediction_crg: torch.Tensor,
) -> dict:
    reference_csg = crg_to_csg(reference_crg)
    pseudo_csg = crg_to_csg(pseudo_crg)
    prediction_csg = crg_to_csg(prediction_crg)
    crg_rows = [
        {
            "index": index,
            "kind": "crg",
            "snr_pred_db": snr_db(reference_crg[index], prediction_crg[index]),
            "snr_pseudo_db": snr_db(reference_crg[index], pseudo_crg[index]),
        }
        for index in range(reference_crg.shape[0])
    ]
    csg_rows = [
        {
            "index": index,
            "kind": "csg",
            "snr_pred_db": snr_db(reference_csg[index], prediction_csg[index]),
            "snr_pseudo_db": snr_db(reference_csg[index], pseudo_csg[index]),
        }
        for index in range(reference_csg.shape[0])
    ]

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "kind", "snr_pred_db", "snr_pseudo_db"],
            )
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "snr_per_crg.csv", crg_rows)
    write_csv(output_dir / "snr_per_csg.csv", csg_rows)
    summary = {
        "method": "NullBlend",
        "overall_snr_pred_db": snr_db(reference_crg, prediction_crg),
        "overall_snr_pseudo_db": snr_db(reference_crg, pseudo_crg),
        "n_crg": reference_crg.shape[0],
        "n_csg": reference_csg.shape[0],
        "crg_mean_snr_pred_db": sum(row["snr_pred_db"] for row in crg_rows)
        / max(len(crg_rows), 1),
        "csg_mean_snr_pred_db": sum(row["snr_pred_db"] for row in csg_rows)
        / max(len(csg_rows), 1),
        "crg": crg_rows,
        "csg": csg_rows,
    }
    write_json(output_dir / "snr_tables.json", summary)
    return summary


def evaluate_and_save(
    *,
    output_dir: Path,
    reference_crg: torch.Tensor,
    pseudo_csg: torch.Tensor,
    prediction_crg: torch.Tensor,
    dt: float,
    dx: float,
) -> dict:
    """Save final tensors, SNR tables and representative 600-dpi figures."""
    pseudo_crg = csg_to_crg(pseudo_csg.float())
    summary = _write_snr_tables(
        output_dir, reference_crg, pseudo_crg, prediction_crg
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    reference_csg = crg_to_csg(reference_crg)
    prediction_csg = crg_to_csg(prediction_crg)
    crg_rows = {row["index"]: row for row in summary["crg"]}
    csg_rows = {row["index"]: row for row in summary["csg"]}
    receiver_ids = sorted(
        {
            0,
            reference_crg.shape[0] // 8,
            reference_crg.shape[0] // 4,
            reference_crg.shape[0] // 2,
            3 * reference_crg.shape[0] // 4,
            7 * reference_crg.shape[0] // 8,
            reference_crg.shape[0] - 1,
        }
    )
    shot_ids = sorted(
        {
            0,
            reference_csg.shape[0] // 8,
            reference_csg.shape[0] // 4,
            reference_csg.shape[0] // 2,
            3 * reference_csg.shape[0] // 4,
            7 * reference_csg.shape[0] // 8,
            reference_csg.shape[0] - 1,
        }
    )
    for receiver in receiver_ids:
        row = crg_rows[receiver]
        _plot_crg_panel(
            reference_crg[receiver].T,
            pseudo_crg[receiver].T,
            prediction_crg[receiver].T,
            figure_dir / f"crg_r{receiver:03d}.png",
            f"NullBlend | CRG r={receiver} | S/N {row['snr_pred_db']:.2f} dB "
            f"(pseudo {row['snr_pseudo_db']:.2f} dB)",
            dt,
        )
    for shot in shot_ids:
        row = csg_rows[shot]
        _plot_csg_panel(
            reference_csg[shot],
            pseudo_csg[shot],
            prediction_csg[shot],
            figure_dir / f"csg_s{shot:03d}.png",
            f"NullBlend | CSG s={shot} | S/N {row['snr_pred_db']:.2f} dB "
            f"(pseudo {row['snr_pseudo_db']:.2f} dB)",
            dt,
            dx,
        )

    for axis_name, rows, xlabel, filename in (
        ("CRG", summary["crg"], "Receiver index", "snr_per_crg.png"),
        ("CSG", summary["csg"], "Shot index", "snr_per_csg.png"),
    ):
        plt.figure(figsize=(8, 3.5))
        plt.plot([row["snr_pseudo_db"] for row in rows], label="Pseudo-deblend", lw=1)
        plt.plot([row["snr_pred_db"] for row in rows], label="NullBlend", lw=1.5)
        plt.xlabel(xlabel)
        plt.ylabel("S/N (dB)")
        plt.title(
            f"Per-{axis_name} S/N | NullBlend | overall "
            f"{summary['overall_snr_pred_db']:.2f} dB"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
    return summary


def completion_time() -> str:
    return datetime.now(timezone.utc).isoformat()
