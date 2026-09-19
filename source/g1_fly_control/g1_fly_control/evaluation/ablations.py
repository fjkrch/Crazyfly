"""Resolve acute output-spike ablations and reproducible degree-matched controls."""

from __future__ import annotations

import random
from typing import Any

import torch

from g1_fly_control.connectome.schema import CircuitData


def ablation_mask(num_neurons: int, indices: list[int] | torch.Tensor) -> torch.Tensor:
    """Return a mask for valid, distinct circuit indices (empty means sham)."""
    selected = torch.as_tensor(indices, dtype=torch.long).flatten()
    if selected.numel() and (selected.min() < 0 or selected.max() >= num_neurons):
        raise ValueError("Ablation targets must be valid neuron indices.")
    if selected.unique().numel() != selected.numel():
        raise ValueError("Ablation targets must not repeat a neuron.")
    mask = torch.zeros(num_neurons, dtype=torch.bool)
    mask[selected] = True
    return mask


def resolve_target_indices(
    circuit: CircuitData,
    *,
    source_ids: list[str] | None = None,
    role: str | None = None,
    annotation: tuple[str, str] | None = None,
) -> list[int]:
    """Resolve exactly one anatomical selector against stable source IDs."""
    if sum(value is not None for value in (source_ids, role, annotation)) != 1:
        raise ValueError("Choose exactly one of source IDs, model role, or annotation.")
    id_to_index = {source_id: index for index, source_id in enumerate(circuit.neuron_ids)}
    if source_ids is not None:
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Ablation source IDs must be unique.")
        missing = sorted(set(source_ids) - id_to_index.keys())
        if missing:
            raise ValueError(f"Ablation source IDs are absent from the circuit: {', '.join(missing)}")
        selected = [id_to_index[source_id] for source_id in source_ids]
    elif role is not None:
        selected = [index for index, source_id in enumerate(circuit.neuron_ids)
                    if circuit.annotations[source_id].get("model_role") == role]
    else:
        assert annotation is not None
        field, value = annotation
        if not field or not value:
            raise ValueError("Annotation selector needs a non-empty field and value.")
        selected = [index for index, source_id in enumerate(circuit.neuron_ids)
                    if str(circuit.annotations[source_id].get(field)) == value]
    if not selected:
        raise ValueError("Ablation selector resolved to no circuit neurons.")
    return sorted(selected)


def directed_degrees(edge_index: torch.Tensor, num_neurons: int) -> tuple[list[int], list[int]]:
    """Return (out-degree, in-degree) for a pre -> post edge list."""
    if num_neurons <= 0 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected edge_index with shape [2, edges] and positive neuron count.")
    if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= num_neurons):
        raise ValueError("Edge endpoints exceed the declared neuron count.")
    return (
        torch.bincount(edge_index[0], minlength=num_neurons).tolist(),
        torch.bincount(edge_index[1], minlength=num_neurons).tolist(),
    )


def matched_random_groups(
    edge_index: torch.Tensor,
    target_indices: list[int],
    *,
    seed: int,
    count: int = 10,
    num_neurons: int | None = None,
) -> list[list[int]]:
    """Choose unique non-target groups of equal size, close in directed degree.

    For each target, controls are sampled without replacement from the ten
    closest remaining neurons by L1 distance in (out-degree, in-degree).
    Random tie/order choices make distinct replicates while keeping degree
    mismatch bounded by the local candidate pool. Exact matching is not
    guaranteed; callers should report the realized mismatch.
    """
    if count < 1:
        raise ValueError("At least one random control group is required.")
    if num_neurons is None:
        num_neurons = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    out_degree, in_degree = directed_degrees(edge_index, num_neurons)
    targets = list(target_indices)
    if not targets or len(set(targets)) != len(targets) or min(targets) < 0 or max(targets) >= num_neurons:
        raise ValueError("Target indices must be a non-empty, unique circuit subset.")
    pool = set(range(num_neurons)) - set(targets)
    if len(pool) < len(targets):
        raise ValueError("Insufficient non-target neurons for size-matched controls.")
    rng = random.Random(seed)
    order = sorted(targets, key=lambda index: (-(out_degree[index] + in_degree[index]), index))
    groups: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(count):
        for _attempt in range(1000):
            available = set(pool)
            chosen = []
            for target in order:
                rank = sorted(
                    available,
                    key=lambda index: (
                        abs(out_degree[index] - out_degree[target])
                        + abs(in_degree[index] - in_degree[target]),
                        rng.random(),
                    ),
                )
                candidate = rng.choice(rank[: min(10, len(rank))])
                chosen.append(candidate)
                available.remove(candidate)
            group = tuple(sorted(chosen))
            if group not in seen:
                seen.add(group)
                groups.append(list(group))
                break
        else:
            raise ValueError("Could not draw the requested number of distinct random control groups.")
    return groups


def degree_match_report(
    edge_index: torch.Tensor, target_indices: list[int], control_indices: list[int], *, num_neurons: int
) -> dict[str, Any]:
    """Report realized directed-degree imbalance using sorted marginal profiles."""
    if len(target_indices) != len(control_indices):
        raise ValueError("Target and control groups must have equal size.")
    out_degree, in_degree = directed_degrees(edge_index, num_neurons)
    target_out = sorted(out_degree[index] for index in target_indices)
    target_in = sorted(in_degree[index] for index in target_indices)
    control_out = sorted(out_degree[index] for index in control_indices)
    control_in = sorted(in_degree[index] for index in control_indices)
    return {
        "target_out_degree": target_out,
        "target_in_degree": target_in,
        "control_out_degree": control_out,
        "control_in_degree": control_in,
        "out_degree_l1_distance": sum(abs(a - b) for a, b in zip(target_out, control_out, strict=True)),
        "in_degree_l1_distance": sum(abs(a - b) for a, b in zip(target_in, control_in, strict=True)),
    }
