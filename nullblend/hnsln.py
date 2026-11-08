"""Hierarchical Null Space Learning Network (HNSLN).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)


def _haar_analysis(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    even_h, odd_h = features[:, :, 0::2, :], features[:, :, 1::2, :]
    average_h = (even_h + odd_h) / _SQRT2
    difference_h = (even_h - odd_h) / _SQRT2

    def split_width(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (tensor[:, :, :, 0::2] + tensor[:, :, :, 1::2]) / _SQRT2,
            (tensor[:, :, :, 0::2] - tensor[:, :, :, 1::2]) / _SQRT2,
        )

    low_low, low_high = split_width(average_h)
    high_low, high_high = split_width(difference_h)
    
    return low_low, torch.cat([low_high, high_low, high_high], dim=1)


def _haar_synthesis(low_low: torch.Tensor, details: torch.Tensor) -> torch.Tensor:
    channels = low_low.shape[1]
    low_high, high_low, high_high = details.split(channels, dim=1)

    def merge_width(sum_part: torch.Tensor, difference_part: torch.Tensor) -> torch.Tensor:
        even = (sum_part + difference_part) / _SQRT2
        odd = (sum_part - difference_part) / _SQRT2
        merged = torch.zeros(
            even.shape[0],
            even.shape[1],
            even.shape[2],
            even.shape[3] * 2,
            device=even.device,
            dtype=even.dtype,
        )
        merged[:, :, :, 0::2] = even
        merged[:, :, :, 1::2] = odd
        return merged

    average_h = merge_width(low_low, low_high)
    difference_h = merge_width(high_low, high_high)
    even_h = (average_h + difference_h) / _SQRT2
    odd_h = (average_h - difference_h) / _SQRT2
    
    merged = torch.zeros(
        even_h.shape[0],
        even_h.shape[1],
        even_h.shape[2] * 2,
        even_h.shape[3],
        device=even_h.device,
        dtype=even_h.dtype,
    )
    merged[:, :, 0::2, :] = even_h
    merged[:, :, 1::2, :] = odd_h
    
    return merged


class MultiLevelWaveletUNet(nn.Module):

    def __init__(self, width: int, levels: int) -> None:
        super().__init__()
        self.levels = int(levels)
        
        self.coarse_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width, width, 1),
                )
                for _ in range(self.levels)
            ]
        )
        
        self.detail_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width * 3, width * 3, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width * 3, width * 3, 1),
                )
                for _ in range(self.levels)
            ]
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        
        self.local_path = nn.Conv2d(width, width, 3, padding=1)
        self.normalization = nn.GroupNorm(8, width)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        detail_skips = []
        coarse = features
        
        for level in range(self.levels):
            low, details = _haar_analysis(coarse)
            
            detail_skips.append(self.detail_blocks[level](details))
            coarse = self.coarse_blocks[level](low)
            
        coarse = self.bottleneck(coarse)
        
        for level in reversed(range(self.levels)):
            coarse = _haar_synthesis(coarse, detail_skips[level])
        
        return F.gelu(self.normalization(coarse + self.local_path(features)))


def _pack_time_aligned_features(
    features: torch.Tensor,
    alignment_index: torch.Tensor,
    record_length: int,
    maximum_overlap: int,
) -> torch.Tensor:
    
    batch, channels, n_shots, nt = features.shape
    source = features.reshape(batch, channels, n_shots * nt)
    index = alignment_index.reshape(1, 1, -1).expand(batch, channels, -1)
    packed = features.new_zeros(batch, channels, record_length * maximum_overlap)
    packed.scatter_(2, index, source)
    
    return packed.view(batch, channels, record_length, maximum_overlap)


def _unpack_time_aligned_features(
    packed: torch.Tensor,
    alignment_index: torch.Tensor,
    n_shots: int,
    nt: int,
) -> torch.Tensor:
    
    batch, channels, record_length, maximum_overlap = packed.shape
    flat = packed.reshape(batch, channels, record_length * maximum_overlap)
    index = alignment_index.reshape(1, 1, -1).expand(batch, channels, -1)
    
    return flat.gather(2, index).view(batch, channels, n_shots, nt)


class TimeAlignedFeatureAggregation(nn.Module):

    def __init__(self, width: int) -> None:
        super().__init__()
        # A fixed initialization budget keeps the published seeded random
        # stream stable. The learned projection has only self/mean inputs.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Conv2d(width * 2, width, 1)
        bound = 1.0 / math.sqrt(width * 3)
        torch.empty(width, width * 3, 1, 1).uniform_(-bound, bound)
        torch.empty(width).uniform_(-bound, bound)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        features: torch.Tensor,
        alignment_index: torch.Tensor,
        valid_slots: torch.Tensor,
        record_length: int,
        maximum_overlap: int,
    ) -> torch.Tensor:
        
        packed = _pack_time_aligned_features(
            features, alignment_index, record_length, maximum_overlap
        )
        mask = valid_slots.view(1, 1, record_length, maximum_overlap).to(
            dtype=packed.dtype
        )
        
        aligned_mean = (packed * mask).sum(dim=-1, keepdim=True) / mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        
        update = self.projection(
            torch.cat([packed, aligned_mean.expand_as(packed)], dim=1)
        )
        
        return _unpack_time_aligned_features(
            packed + update,
            alignment_index,
            features.shape[2],
            features.shape[3],
        )


class HierarchicalNullSpaceLearningNetwork(nn.Module):

    def __init__(
        self,
        width: int = 32,
        depth: int = 4,
        wavelet_levels: int = 3,
        receiver_radius: int = 2,
        maximum_overlap: int = 4,
        record_length: int = 1,
    ) -> None:
        super().__init__()
        
        self.receiver_radius = int(receiver_radius)
        self.maximum_overlap = int(maximum_overlap)
        self.record_length = int(record_length)
        
        receiver_channels = 2 * self.receiver_radius + 1

        self.input_lift = nn.Conv2d(receiver_channels + 3, width, 1)

        self.mw_unet_blocks = nn.ModuleList(
            [MultiLevelWaveletUNet(width, wavelet_levels) for _ in range(depth)]
        )
        
        self.tafa_blocks = nn.ModuleList(
            [TimeAlignedFeatureAggregation(width) for _ in range(depth)]
        )
        
        self.output_head = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, 1, 1),
        )
        
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        
        self.register_buffer(
            "tafa_alignment_index", torch.zeros(1, 1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "tafa_valid_slots",
            torch.zeros(1, maximum_overlap, dtype=torch.bool),
            persistent=True,
        )

    def set_tafa_index(
        self,
        alignment_index: torch.Tensor,
        valid_slots: torch.Tensor,
        record_length: int,
    ) -> None:
        self.record_length = int(record_length)
        self.register_buffer(
            "tafa_alignment_index", alignment_index.long(), persistent=True
        )
        self.register_buffer("tafa_valid_slots", valid_slots.bool(), persistent=True)

    def forward(
        self,
        particular_solution: torch.Tensor,
        overlap_order: torch.Tensor,
        neighboring_receivers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        if neighboring_receivers is None:
            neighboring_receivers = particular_solution.unsqueeze(1).expand(
                -1, 2 * self.receiver_radius + 1, -1, -1
            )
            
        image = neighboring_receivers.permute(0, 1, 3, 2).contiguous()
        
        order = overlap_order.to(dtype=image.dtype)
        if order.dim() == 2:
            order = order.unsqueeze(0).unsqueeze(0)
        elif order.dim() == 3:
            order = order.unsqueeze(1)
            
        order = order.transpose(2, 3)
        if order.shape[0] == 1 and image.shape[0] > 1:
            order = order.expand(image.shape[0], -1, -1, -1)

        batch, _, nt, n_shots = image.shape
        time = torch.linspace(-1.0, 1.0, nt, device=image.device, dtype=image.dtype)
        shot = torch.linspace(
            -1.0, 1.0, n_shots, device=image.device, dtype=image.dtype
        )
        
        time_grid = time.view(1, 1, nt, 1).expand(batch, 1, nt, n_shots)
        shot_grid = shot.view(1, 1, 1, n_shots).expand(batch, 1, nt, n_shots)
        
        inputs = torch.cat([image, order, time_grid, shot_grid], dim=1)
        features = self.input_lift(inputs)
        features = features.permute(0, 1, 3, 2).contiguous()

        for level, tafa in enumerate(self.tafa_blocks):
            features = features + self.mw_unet_blocks[level](features)
            features = tafa(
                features,
                self.tafa_alignment_index,
                self.tafa_valid_slots,
                self.record_length,
                self.maximum_overlap,
            )

        residual = self.output_head(features).squeeze(1)
        
        overlap_gate = (overlap_order >= 2).to(dtype=residual.dtype)
        
        if overlap_gate.dim() == 2:
            overlap_gate = overlap_gate.unsqueeze(0)
            
        if overlap_gate.shape[0] == 1 and residual.shape[0] > 1:
            overlap_gate = overlap_gate.expand(residual.shape[0], -1, -1)
            
        return residual * overlap_gate
