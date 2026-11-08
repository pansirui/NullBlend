from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import OUTPUT_ROOT, REFERENCE_SNR_DB, NullBlendConfig
from .data import csg_to_crg, load_evaluation_reference, load_observed_dataset
from .evaluation import (
    completion_time,
    count_parameters,
    evaluate_and_save,
    overlap_snr_db,
    snr_db,
    write_json,
)
from .hnsln import HierarchicalNullSpaceLearningNetwork
from .operators import (
    blend_shot_gathers,
    build_tafa_index,
    hard_data_consistency,
    minimum_norm_solution,
    overlap_counts,
    pseudo_deblend,
    receiver_context,
    shot_overlap_order,
    valid_shot_shift_mask,
)


def _autocast(enabled: bool):
    return torch.amp.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=enabled,
    )


def _inverse_frequency_weights(overlap_order: torch.Tensor) -> torch.Tensor:
    order = overlap_order[0] if overlap_order.dim() == 3 else overlap_order
    collided = order >= 2
    weights = torch.zeros_like(order, dtype=torch.float32)
    if not bool(collided.any()):
        return weights
    reference_count = ((order == 2) & collided).sum().float().clamp_min(1.0)
    for value in torch.unique(order[collided].long()).tolist():
        support = (order == value).sum().float().clamp_min(1.0)
        weights[order == value] = reference_count / support
    return weights / weights[collided].mean().clamp_min(1e-6)


def _overlap_weighted_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    overlap_order: torch.Tensor,
    valid_transform: torch.Tensor,
) -> torch.Tensor:
    
    weights = _inverse_frequency_weights(overlap_order).to(prediction.device)
    weights = weights * valid_transform.to(device=prediction.device, dtype=weights.dtype)
    weights = weights.unsqueeze(0)
    
    support = weights.sum().clamp_min(1.0)
    
    return ((prediction.float() - target.float()).abs() * weights).sum() / (
        support * prediction.shape[0]
    )


@torch.no_grad()
def update_teboot_teacher(
    teacher: torch.Tensor,
    prediction: torch.Tensor,
    new_prediction_weight: float,
) -> None:
    if new_prediction_weight != 1.0:
        raise ValueError("Only direct teacher replacement is supported")
    teacher.copy_(prediction)



def _add_hnsln_residual(
    model: HierarchicalNullSpaceLearningNetwork,
    particular: torch.Tensor,
    overlap_order: torch.Tensor,
    neighboring_receivers: torch.Tensor,
) -> torch.Tensor:
    
    scale = particular.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-4)
    
    residual = model(
        particular / scale,
        overlap_order,
        neighboring_receivers / scale.unsqueeze(1),
    )
    
    return particular + residual * scale


def _train_teboot_epoch(
    *,
    model: HierarchicalNullSpaceLearningNetwork,
    optimizer: torch.optim.Optimizer,
    transformed_teacher_cpu: torch.Tensor,
    virtual_particular_cpu: torch.Tensor,
    virtual_observed: torch.Tensor,
    overlap_order: torch.Tensor,
    valid_transform: torch.Tensor,
    shot_starts: torch.Tensor,
    counts: torch.Tensor,
    receiver_batch_size: int,
    receiver_radius: int,
) -> float:
    
    model.train()
    
    permutation = torch.randperm(transformed_teacher_cpu.shape[0])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_loss = 0.0
    batches = 0
    
    for start in range(0, transformed_teacher_cpu.shape[0], receiver_batch_size):
        receiver_indices = permutation[start : start + receiver_batch_size]
        
        particular = virtual_particular_cpu[receiver_indices].to(virtual_observed.device)
        
        neighbors = receiver_context(
            virtual_particular_cpu, receiver_indices, receiver_radius
        ).to(virtual_observed.device)
        
        target = transformed_teacher_cpu[receiver_indices].to(virtual_observed.device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with _autocast(virtual_observed.is_cuda):
            candidate = _add_hnsln_residual(
                model, particular, overlap_order, neighbors
            )
            prediction = hard_data_consistency(
                candidate,
                virtual_observed[receiver_indices.to(virtual_observed.device)],
                shot_starts,
                counts,
            )
            loss = _overlap_weighted_l1(
                prediction, target, overlap_order, valid_transform
            )
            
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return total_loss / max(batches, 1)


@torch.no_grad()
def _reconstruct(
    *,
    model: HierarchicalNullSpaceLearningNetwork,
    particular_cpu: torch.Tensor,
    observed_cpu: torch.Tensor,
    overlap_order: torch.Tensor,
    shot_starts: torch.Tensor,
    counts: torch.Tensor,
    device: torch.device,
    receiver_batch_size: int,
    receiver_radius: int,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, particular_cpu.shape[0], receiver_batch_size):
        receiver_indices = torch.arange(
            start, min(start + receiver_batch_size, particular_cpu.shape[0])
        )
        particular = particular_cpu[receiver_indices].to(device)
        neighbors = receiver_context(
            particular_cpu, receiver_indices, receiver_radius
        ).to(device)
        candidate = _add_hnsln_residual(
            model, particular, overlap_order, neighbors
        )
        prediction = hard_data_consistency(
            candidate,
            observed_cpu[receiver_indices].to(device),
            shot_starts,
            counts,
        )
        chunks.append(prediction.cpu())
    return torch.cat(chunks, dim=0)


def _optimizer(
    model: HierarchicalNullSpaceLearningNetwork,
    config: NullBlendConfig,
) -> tuple[torch.optim.Optimizer, dict]:
    groups = []
    assigned: set[int] = set()
    summary = {}

    def add(name: str, module: torch.nn.Module, learning_rate: float) -> None:
        parameters = [
            parameter for parameter in module.parameters() if id(parameter) not in assigned
        ]
        groups.append({"params": parameters, "lr": learning_rate})
        assigned.update(id(parameter) for parameter in parameters)
        summary[name] = {
            "learning_rate": learning_rate,
            "parameter_count": sum(parameter.numel() for parameter in parameters),
        }

    add("tafa", model.tafa_blocks, config.tafa_learning_rate)
    add("output_head", model.output_head, config.output_head_learning_rate)
    remaining = [
        parameter for parameter in model.parameters() if id(parameter) not in assigned
    ]
    groups.append({"params": remaining, "lr": config.learning_rate})
    summary["input_lift_and_mw_unet"] = {
        "learning_rate": config.learning_rate,
        "parameter_count": sum(parameter.numel() for parameter in remaining),
    }
    return torch.optim.Adam(groups), summary


def _timestamped_output(dataset: str, output_root: Path | None) -> Path:
    root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / dataset / f"nullblend_{stamp}"
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return output_dir


def train_nullblend(
    config: NullBlendConfig, output_root: Path | None = None
) -> Path:
    if config.teboot_new_prediction_weight != 1.0:
        raise ValueError("This model uses direct teacher replacement")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(config.seed)
    
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    
    output_dir = _timestamped_output(config.dataset, output_root)
    
    dataset = load_observed_dataset(config.dataset_dir)
    observed_cpu = dataset["observed"]
    shot_starts_cpu = dataset["shot_starts"]
    
    metadata = dataset["metadata"]
    nt = int(metadata["nt"])
    dt = float(metadata["dt"])
    dx = float(metadata.get("dx", 25.0))
    
    record_length = int(observed_cpu.shape[-1])

    shot_starts = shot_starts_cpu.to(device)
    observed = observed_cpu.to(device)
    
    counts = overlap_counts(shot_starts, nt, record_length)
    counts_cpu = counts.cpu()
    
    particular_cpu = minimum_norm_solution(
        observed_cpu, shot_starts_cpu, nt, counts_cpu
    ).contiguous()
    
    overlap_order_cpu = shot_overlap_order(shot_starts_cpu, nt, record_length)
    overlap_order = overlap_order_cpu.float().to(device)
    
    alignment_index, valid_slots = build_tafa_index(
        shot_starts_cpu, nt, record_length, config.maximum_overlap
    )

    model = HierarchicalNullSpaceLearningNetwork(
        width=config.hidden_width,
        depth=config.hierarchy_depth,
        wavelet_levels=config.wavelet_levels,
        receiver_radius=config.receiver_radius,
        maximum_overlap=config.maximum_overlap,
        record_length=record_length,
    ).to(device)
    
    model.set_tafa_index(
        alignment_index.to(device), valid_slots.to(device), record_length
    )
    
    optimizer, optimizer_groups = _optimizer(model, config)
    
    teboot_teacher_cpu = particular_cpu.detach().clone()
    latest_prediction_cpu = teboot_teacher_cpu

    run_config = config.as_dict() | {
        "method": "NullBlend",
        "network": "Hierarchical Null Space Learning Network (HNSLN)",
        "network_modules": ["MW-UNet", "TAFA"],
        "tafa_features": ["self", "aligned_mean"],
        "tafa_deviation_input": False,
        "input_channels": model.input_lift.in_channels,
        "auxiliary_input_channels": ["overlap_order", "time_coordinate", "shot_coordinate"],
        "training_strategy": "Bootstrapping with direct teacher replacement",
        "ema_enabled": False,
        "initial_teacher": "minimum-norm solution A^dagger y",
        "loss": "collision-frequency-weighted L1",
        "hard_data_consistency": True,
        "hnsln_trainable_from_epoch_1": True,
        "final_output": "last TEBoot teacher",
        "clean_reference_usage": "not loaded until all training epochs finish",
        "parameter_count": count_parameters(model),
        "optimizer_groups": optimizer_groups,
        "output_dir": str(output_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    
    write_json(output_dir / "config.json", run_config)
    history_path = output_dir / "training_history.csv"
    
    print(
        f"[NullBlend] dataset={config.dataset} device={device} "
        f"HNSLN={count_parameters(model):,} initialization=A^dagger_y "
        f"output={output_dir}",
        flush=True,
    )

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["loss"])
        writer.writeheader()
        
        for epoch in range(1, config.epochs + 1):
            shot_shift = config.shot_shifts[(epoch - 1) % len(config.shot_shifts)]
            
            transformed_teacher_cpu = torch.roll(
                teboot_teacher_cpu, shifts=shot_shift, dims=1
            )
            
            valid_transform = valid_shot_shift_mask(
                transformed_teacher_cpu.shape[1],
                transformed_teacher_cpu.shape[2],
                shot_shift,
                device,
            )
            
            virtual_observed_cpu = blend_shot_gathers(
                transformed_teacher_cpu, shot_starts_cpu, record_length
            )
            
            virtual_particular_cpu = minimum_norm_solution(
                virtual_observed_cpu,
                shot_starts_cpu,
                nt,
                counts_cpu,
            )
            
            loss = _train_teboot_epoch(
                model=model,
                optimizer=optimizer,
                transformed_teacher_cpu=transformed_teacher_cpu,
                virtual_particular_cpu=virtual_particular_cpu,
                virtual_observed=virtual_observed_cpu.to(device),
                overlap_order=overlap_order,
                valid_transform=valid_transform,
                shot_starts=shot_starts,
                counts=counts,
                receiver_batch_size=config.receiver_batch_size,
                receiver_radius=config.receiver_radius,
            )
            
            latest_prediction_cpu = _reconstruct(
                model=model,
                particular_cpu=particular_cpu,
                observed_cpu=observed_cpu,
                overlap_order=overlap_order,
                shot_starts=shot_starts,
                counts=counts,
                device=device,
                receiver_batch_size=config.receiver_batch_size,
                receiver_radius=config.receiver_radius,
            )
            
            teboot_updated = epoch % config.teboot_update_period == 0
            if teboot_updated:
                update_teboot_teacher(
                    teboot_teacher_cpu,
                    latest_prediction_cpu,
                    config.teboot_new_prediction_weight,
                )
            
            writer.writerow({"loss": loss})
            
            history_file.flush()
            print(f"[NullBlend] {epoch}/{config.epochs}", flush=True)
            write_json(
                output_dir / "status.json",
                {
                    "state": "running",
                    "method": "NullBlend",
                    "dataset": config.dataset,
                    "epoch": epoch,
                    "epochs": config.epochs,
                    "loss": loss,
                    "shot_shift": shot_shift,
                    "teboot_updated": teboot_updated,
                    "clean_reference_usage": "not loaded during training",
                },
            )
            
            torch.save(
                teboot_teacher_cpu, output_dir / "teboot_teacher_latest.pt"
            )
            torch.save(
                latest_prediction_cpu, output_dir / "deblended_crg_latest.pt"
            )
            torch.save(
                {
                    "method": "NullBlend",
                    "hnsln": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "teboot_teacher": teboot_teacher_cpu,
                },
                output_dir / "checkpoints" / "latest.pt",
            )

    final_prediction = teboot_teacher_cpu.detach().clone()
    torch.save(final_prediction, output_dir / "deblended_crg_final.pt")
    torch.save(latest_prediction_cpu, output_dir / "deblended_crg_last_student.pt")
    torch.save(model.state_dict(), output_dir / "hnsln_final.pt")
    if not (config.dataset_dir / "reference_clean.pt").is_file():
        write_json(output_dir / "status.json", {
            "state": "complete", "method": "NullBlend", "dataset": config.dataset,
            "epochs": config.epochs, "ema_enabled": False, "evaluation_available": False,
            "parameter_count": count_parameters(model), "output_dir": str(output_dir)})
        return output_dir
    reference_crg = csg_to_crg(load_evaluation_reference(config.dataset_dir))
    initial_snr = snr_db(reference_crg, particular_cpu)
    final_snr = snr_db(reference_crg, final_prediction)
    pseudo_csg = dataset["pseudo_csg"]
    if pseudo_csg is None:
        pseudo_csg = pseudo_deblend(observed_cpu, shot_starts_cpu, nt)
    summary = evaluate_and_save(
        output_dir=output_dir,
        reference_crg=reference_crg,
        pseudo_csg=pseudo_csg,
        prediction_crg=final_prediction,
        dt=dt,
        dx=dx,
    )
    overlap_summary = overlap_snr_db(
        final_prediction, reference_crg, overlap_order_cpu
    )
    
    torch.save(final_prediction, output_dir / "deblended_crg_final.pt")
    torch.save(
        latest_prediction_cpu,
        output_dir / "deblended_crg_last_student.pt",
    )
    torch.save(model.state_dict(), output_dir / "hnsln_final.pt")
    
    reference_snr = REFERENCE_SNR_DB[config.dataset]
    write_json(
        output_dir / "status.json",
        {
            "state": "complete",
            "method": "NullBlend",
            "dataset": config.dataset,
            "epochs": config.epochs,
            "initial_minimum_norm_snr_db_eval_only": initial_snr,
            "final_output": "last TEBoot teacher",
            "final_snr_db_eval_only": final_snr,
            "last_student_snr_db_eval_only": snr_db(
                reference_crg, latest_prediction_cpu
            ),
            "overall_snr_pred_db": summary["overall_snr_pred_db"],
            "overall_snr_pseudo_db": summary["overall_snr_pseudo_db"],
            "crg_mean_snr_pred_db": summary["crg_mean_snr_pred_db"],
            "csg_mean_snr_pred_db": summary["csg_mean_snr_pred_db"],
            "overlap_snr_db_eval_only": overlap_summary,
            "hnsln_parameters": count_parameters(model),
            "teboot_update_period": config.teboot_update_period,
            "ema_enabled": config.teboot_new_prediction_weight < 1.0,
            "teboot_new_prediction_weight": config.teboot_new_prediction_weight,
            "bundled_reference_snr_db": reference_snr,
            "difference_from_bundled_reference_db": final_snr - reference_snr,
            "clean_reference_usage": "evaluation only; never used for training, selection, or stopping",
            "completed_utc": completion_time(),
            "output_dir": str(output_dir),
        },
    )
    print(
        f"[NullBlend] complete dataset={config.dataset} final={final_snr:.3f} dB "
        f"reference={reference_snr:.3f} dB output={output_dir}",
        flush=True,
    )
    return output_dir
