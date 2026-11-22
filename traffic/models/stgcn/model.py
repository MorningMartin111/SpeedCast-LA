"""Original PyTorch implementation of the STGCN block in Yu et al. (2018).

Reference: https://www.ijcai.org/proceedings/2018/0505.pdf
Architecture reference: https://github.com/VeritasYin/STGCN_IJCAI-18
No upstream source code or weights are copied. The direct 12-horizon head,
per-node channel normalization, and causal data preparation are project choices.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn


SOURCE = {
    "paper": "https://www.ijcai.org/proceedings/2018/0505.pdf",
    "reference_repository": "https://github.com/VeritasYin/STGCN_IJCAI-18",
    "implementation": "Original project PyTorch reimplementation; no copied code or pretrained weights",
    "reference_revision": "4ab75ab69af354d6a0ab86aefa082d389b187127",
    "project_changes": [
        "Direct per-node 12-step prediction head",
        "Per-node feature-channel LayerNorm",
        "max(A,A.T) symmetrization and fixed lambda_max=2 normalized Laplacian",
        "Temporal ablation replaces only graph convolution with a channel projection",
    ],
}


def chebyshev_supports(adj_dir: np.ndarray | Tensor, order: int) -> Tensor:
    """Return T_0..T_(order-1) of L_tilde = L - I, with lambda_max fixed at 2.

    A is symmetrized with max(A,A.T), then its diagonal is cleared. We define
    L=I-D^(-1/2) A D^(-1/2); isolated nodes have inverse degree zero and L_ii=1.
    Thus their L_tilde row is zero, avoiding a division by zero. This operation
    runs once per model and never mutates the shared directed graph.
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError("cheb_order must be a positive integer")
    adjacency = torch.as_tensor(adj_dir, dtype=torch.float32).detach().cpu().clone()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or not len(adjacency):
        raise ValueError("adj_dir must be a nonempty square matrix")
    if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
        raise ValueError("adj_dir must have finite nonnegative weights")
    adjacency = torch.maximum(adjacency, adjacency.T)
    adjacency.fill_diagonal_(0)
    degree = adjacency.sum(dim=1)
    inverse = torch.zeros_like(degree)
    nonzero = degree > 0
    inverse[nonzero] = degree[nonzero].rsqrt()
    scaled = -(inverse[:, None] * adjacency * inverse[None, :])
    supports = [torch.eye(len(adjacency), dtype=torch.float32)]
    if order > 1:
        supports.append(scaled)
    for _ in range(2, order):
        supports.append(2 * scaled @ supports[-1] - supports[-2])
    return torch.stack(supports)


class TemporalGLU(nn.Module):
    """A valid temporal convolution; kernel width across nodes is exactly one."""

    def __init__(self, input_channels: int, output_channels: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.convolution = nn.Conv2d(input_channels, 2 * output_channels, (kernel, 1))
        self.residual = (
            nn.Identity() if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.convolution(x).chunk(2, dim=1)
        residual = self.residual(x)[:, :, self.kernel - 1 :, :]
        return (value + residual) * torch.sigmoid(gate)


class GraphConvolution(nn.Module):
    def __init__(self, channels: int, order: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(order, channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, supports: Tensor) -> Tensor:
        # support[k,i,j] multiplies source j into target i. Supports are symmetric.
        propagated = torch.einsum("kij,bctj->bkcti", supports, x)
        return torch.einsum("bkctn,kco->botn", propagated, self.weight) + self.bias[None, :, None, None]


class STBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int, kernel: int, order: int,
                 dropout: float, temporal_only: bool):
        super().__init__()
        self.temporal_before = TemporalGLU(input_channels, channels, kernel)
        self.spatial = nn.Conv2d(channels, channels, 1) if temporal_only else GraphConvolution(channels, order)
        self.temporal_only = temporal_only
        self.temporal_after = TemporalGLU(channels, channels, kernel)
        self.normalization = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, supports: Tensor) -> Tensor:
        x = self.temporal_before(x)
        x = self.spatial(x) if self.temporal_only else self.spatial(x, supports)
        x = self.temporal_after(torch.relu(x))
        # Normalize only channels at each (batch,time,node), never across nodes.
        x = self.normalization(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class STGCN(nn.Module):
    """Accept x[B,12,N,3], return standardized forecasts[B,12,N]."""

    def __init__(self, adj_dir: np.ndarray | Tensor, config: Mapping[str, object]):
        super().__init__()
        model_name = config.get("model", "stgcn")
        if model_name not in {"stgcn", "stgcn_temporal"}:
            raise ValueError("model must be stgcn or stgcn_temporal")
        channels = int(config.get("hidden_channels", 32))
        blocks = int(config.get("blocks", 2))
        kernel = int(config.get("temporal_kernel", 3))
        order = int(config.get("cheb_order", 3))
        dropout = float(config.get("dropout", 0.1))
        if min(channels, blocks, kernel) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid STGCN widths, block count, kernel, or dropout")
        length = 12 - 2 * blocks * (kernel - 1)
        if length < 1:
            raise ValueError("Temporal convolutions exhaust the 12-step input")
        supports = chebyshev_supports(adj_dir, order)
        self.register_buffer("supports", supports)
        self.node_count = supports.shape[-1]
        self.blocks = nn.ModuleList([
            STBlock(3 if index == 0 else channels, channels, kernel, order, dropout,
                    temporal_only=model_name == "stgcn_temporal")
            for index in range(blocks)
        ])
        self.head = nn.Linear(channels * length, 12)
        self.temporal_lengths = [12 - index * (kernel - 1) for index in range(2 * blocks + 1)]

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (12, self.node_count, 3):
            raise ValueError(f"Expected x[B,12,{self.node_count},3], got {tuple(x.shape)}")
        x = x.permute(0, 3, 1, 2)
        for block in self.blocks:
            x = block(x, self.supports)
        # A separate application of the same head at every node; no node mixing.
        x = x.permute(0, 3, 2, 1).flatten(start_dim=2)
        return self.head(x).transpose(1, 2).contiguous()
