"""Degree-preserving rewiring for matched graph comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class RewireReport:
    requested_swaps: int
    completed_swaps: int
    attempts: int
    seed: int


def degree_preserving_rewire(
    edge_index: torch.Tensor,
    weights: torch.Tensor,
    *,
    seed: int,
    swaps: int | None = None,
    allow_self_loops: bool = False,
    allow_duplicates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, RewireReport]:
    """Swap edge destinations while preserving every directed in/out degree.

    The edge's weight travels with its source edge, preserving the outgoing weight/sign
    multiset of every neuron. The procedure is not mere neuron relabeling.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] != weights.numel():
        raise ValueError("edge_index must be [2, E] and align with weights.")
    count = edge_index.shape[1]
    if count < 2:
        raise ValueError("At least two edges are needed for degree-preserving rewiring.")
    requested = swaps if swaps is not None else 10 * count
    if requested < 1:
        raise ValueError("swaps must be positive.")
    generator = random.Random(seed)
    result = edge_index.detach().clone().cpu()
    pair_set = {(int(result[0, i]), int(result[1, i])) for i in range(count)}
    completed = 0
    attempts = 0
    max_attempts = max(requested * 50, 200)
    while completed < requested and attempts < max_attempts:
        attempts += 1
        first, second = generator.sample(range(count), 2)
        src_a, dst_a = int(result[0, first]), int(result[1, first])
        src_b, dst_b = int(result[0, second]), int(result[1, second])
        if dst_a == dst_b:
            continue
        candidate_a, candidate_b = (src_a, dst_b), (src_b, dst_a)
        if not allow_self_loops and (candidate_a[0] == candidate_a[1] or candidate_b[0] == candidate_b[1]):
            continue
        existing = pair_set - {(src_a, dst_a), (src_b, dst_b)}
        if not allow_duplicates and (candidate_a in existing or candidate_b in existing or candidate_a == candidate_b):
            continue
        pair_set.remove((src_a, dst_a))
        pair_set.remove((src_b, dst_b))
        pair_set.update((candidate_a, candidate_b))
        result[1, first], result[1, second] = dst_b, dst_a
        completed += 1
    return result.to(edge_index.device), weights.detach().clone(), RewireReport(requested, completed, attempts, seed)

