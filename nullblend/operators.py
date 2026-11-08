from __future__ import annotations

import torch


def overlap_counts(shot_starts: torch.Tensor, nt: int, length: int) -> torch.Tensor:
    counts = torch.zeros(length, dtype=torch.float32, device=shot_starts.device)
    for start in shot_starts.long().tolist():
        counts[start : start + nt] += 1
    return counts.clamp_min(1.0)


def blend_shot_gathers(
    crg: torch.Tensor, shot_starts: torch.Tensor, length: int
) -> torch.Tensor:
    """Apply the blending operator: [receiver, shot, time] -> [receiver, T]."""
    blended = crg.new_zeros(crg.shape[0], length)
    for shot, start in enumerate(shot_starts.long().tolist()):
        blended[:, start : start + crg.shape[-1]] += crg[:, shot]
    return blended


def adjoint_extract(
    blended: torch.Tensor, shot_starts: torch.Tensor, nt: int
) -> torch.Tensor:
    return torch.stack(
        [blended[:, start : start + nt] for start in shot_starts.long().tolist()],
        dim=1,
    )


def minimum_norm_solution(
    observed: torch.Tensor,
    shot_starts: torch.Tensor,
    nt: int,
    counts: torch.Tensor | None = None,
) -> torch.Tensor:
    if counts is None:
        counts = overlap_counts(shot_starts, nt, observed.shape[-1])
    return adjoint_extract(observed / counts.unsqueeze(0), shot_starts, nt)


def hard_data_consistency(
    candidate: torch.Tensor,
    observed: torch.Tensor,
    shot_starts: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    residual = observed - blend_shot_gathers(
        candidate, shot_starts, observed.shape[-1]
    )
    correction = adjoint_extract(
        residual / counts.unsqueeze(0), shot_starts, candidate.shape[-1]
    )
    return candidate + correction


def shot_overlap_order(
    shot_starts: torch.Tensor, nt: int, length: int
) -> torch.Tensor:
    starts = shot_starts.detach().cpu().long()
    counts = torch.zeros(length, dtype=torch.long)
    for start in starts.tolist():
        counts[start : start + nt] += 1
    absolute_time = starts.unsqueeze(1) + torch.arange(nt)
    return counts[absolute_time]


def build_tafa_index(
    shot_starts: torch.Tensor,
    nt: int,
    length: int,
    maximum_overlap: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = shot_starts.detach().cpu().long()
    n_shots = int(starts.numel())
    absolute_time = (
        starts.unsqueeze(1) + torch.arange(nt).unsqueeze(0)
    ).clamp(0, length - 1)
    slots = torch.zeros(n_shots, nt, dtype=torch.long)
    occupancy = torch.zeros(length, dtype=torch.long)
    for shot in range(n_shots):
        times = absolute_time[shot]
        slots[shot] = occupancy[times]
        occupancy[times] += 1
    slots = slots.clamp(0, maximum_overlap - 1)
    flat_index = absolute_time * maximum_overlap + slots
    valid = torch.arange(maximum_overlap).unsqueeze(0) < occupancy.unsqueeze(1).clamp(
        max=maximum_overlap
    )
    return flat_index, valid


def receiver_context(
    volume: torch.Tensor, receiver_indices: torch.Tensor, radius: int
) -> torch.Tensor:
    n_receivers = volume.shape[0]
    windows = []
    for receiver in receiver_indices.tolist():
        indices = [
            min(max(receiver + delta, 0), n_receivers - 1)
            for delta in range(-radius, radius + 1)
        ]
        windows.append(volume[indices])
    return torch.stack(windows, dim=0)


def valid_shot_shift_mask(
    n_shots: int, nt: int, shift: int, device: torch.device
) -> torch.Tensor:
    mask = torch.ones(n_shots, nt, dtype=torch.bool, device=device)
    if shift > 0:
        mask[:shift] = False
    elif shift < 0:
        mask[shift:] = False
    return mask


def pseudo_deblend(
    observed: torch.Tensor, shot_starts: torch.Tensor, nt: int
) -> torch.Tensor:
    return adjoint_extract(observed, shot_starts, nt).permute(1, 0, 2).contiguous()
