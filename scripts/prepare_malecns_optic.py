#!/usr/bin/env python3
"""Derive a deterministic, provenance-tracked optic-lobe circuit from MaleCNS.

The circuit contains only cells whose MaleCNS ``superclass`` annotations name
their biological role: 32 ``ol_sensory`` photoreceptors, 200
``ol_intrinsic`` cells, and 24 ``visual_projection`` readouts.  Every selected
cell lies on a directed sensory -> intrinsic -> projection path.

Raw edge weights are synapse counts, not conductances.  The exported signed
model weights are an explicit engineering transform of those counts.  Only
transmitters with an unambiguous sign under this model are allowed to originate
derived edges: photoreceptor histamine and intrinsic GABA are inhibitory;
intrinsic acetylcholine is excitatory.  Glutamatergic intrinsic cells are
excluded because a sign cannot be inferred without postsynaptic receptor data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

try:  # Package import under pytest and direct ``python scripts/...`` execution.
    from scripts.prepare_malecns import (
        ANNOTATION_COLUMNS,
        NT_COLUMNS,
        SOURCE_BASE_URL,
        SOURCE_FILES,
        _clean,
        _nodes_on_input_output_paths,
        _reaches_outputs,
        _sha256,
        _write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI invocation.
    from prepare_malecns import (
        ANNOTATION_COLUMNS,
        NT_COLUMNS,
        SOURCE_BASE_URL,
        SOURCE_FILES,
        _clean,
        _nodes_on_input_output_paths,
        _reaches_outputs,
        _sha256,
        _write_json,
    )


EXPECTED_SOURCE_SHA256 = {
    "annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "connectivity": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    "neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
}
ALLOWED_STATUS_LABELS = (
    "Reviewed",
    "Roughly traced",
    "Prelim Roughly traced",
)
SIDES = ("L", "R")
OPTIC_SIGNS = {
    "acetylcholine": 1,
    "gaba": -1,
    "histamine": -1,
}
OPTIC_ANNOTATION_COLUMNS = list(
    dict.fromkeys(
        ANNOTATION_COLUMNS
        + ["flywireType", "supertype", "assignedOlHex1", "assignedOlHex2"]
    )
)


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _verify_sources(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    details: dict[str, dict[str, str]] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing MaleCNS source file: {path}")
        actual = _sha256(path)
        expected = EXPECTED_SOURCE_SHA256[label]
        if actual != expected:
            raise ValueError(
                f"Pinned MaleCNS {label} SHA-256 mismatch: "
                f"expected={expected}, actual={actual}"
            )
        details[label] = {
            "filename": path.name,
            "sha256": actual,
            "url": SOURCE_BASE_URL + path.name,
        }
    return details


def _aggregate_batches(batches: list[pa.RecordBatch], label: str) -> pd.DataFrame:
    if not batches:
        raise ValueError(f"No raw MaleCNS edges matched {label}")
    result = pa.Table.from_batches(batches).to_pandas()
    if result.weight.isna().any() or (result.weight <= 0).any():
        raise ValueError(f"{label} edges must have positive finite synapse counts")
    return (
        result.groupby(["body_pre", "body_post"], as_index=False, sort=True)
        .weight.sum()
        .sort_values(["body_pre", "body_post"], kind="mergesort")
        .reset_index(drop=True)
    )


def _read_cross_layers(
    path: Path,
    sensory_ids: np.ndarray,
    intrinsic_ids: np.ndarray,
    projection_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, int, dict[str, int]]:
    """Read only sensory->intrinsic and intrinsic->projection raw edges."""

    reader = pa.ipc.open_file(pa.memory_map(str(path), "r"))
    sensory_values = pa.array(sensory_ids)
    intrinsic_values = pa.array(intrinsic_ids)
    projection_values = pa.array(projection_ids)
    sensory_intrinsic: list[pa.RecordBatch] = []
    intrinsic_projection: list[pa.RecordBatch] = []
    source_rows = 0
    raw_si = 0
    raw_ip = 0
    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number)
        source_rows += batch.num_rows
        si = batch.filter(
            pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=sensory_values),
                pc.is_in(batch.column("body_post"), value_set=intrinsic_values),
            )
        )
        ip = batch.filter(
            pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=intrinsic_values),
                pc.is_in(batch.column("body_post"), value_set=projection_values),
            )
        )
        if si.num_rows:
            raw_si += si.num_rows
            sensory_intrinsic.append(si)
        if ip.num_rows:
            raw_ip += ip.num_rows
            intrinsic_projection.append(ip)
    del reader
    return (
        _aggregate_batches(sensory_intrinsic, "ol_sensory -> ol_intrinsic"),
        _aggregate_batches(intrinsic_projection, "ol_intrinsic -> visual_projection"),
        source_rows,
        {"sensory_intrinsic": raw_si, "intrinsic_projection": raw_ip},
    )


def _read_selected_edges(path: Path, selected_ids: set[int]) -> tuple[pd.DataFrame, int]:
    reader = pa.ipc.open_file(pa.memory_map(str(path), "r"))
    selected_values = pa.array(np.asarray(sorted(selected_ids), dtype=np.int64))
    batches: list[pa.RecordBatch] = []
    raw_count = 0
    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number)
        selected = batch.filter(
            pc.and_(
                pc.is_in(batch.column("body_pre"), value_set=selected_values),
                pc.is_in(batch.column("body_post"), value_set=selected_values),
            )
        )
        if selected.num_rows:
            raw_count += selected.num_rows
            batches.append(selected)
    del reader
    return _aggregate_batches(batches, "selected induced"), raw_count


def _ranked_ids(
    frame: pd.DataFrame,
    score: pd.Series,
    *,
    side_column: str,
    per_side: int,
    label: str,
) -> list[int]:
    ranked = frame.assign(_score=frame.bodyId.map(score).fillna(0.0))
    chosen: list[int] = []
    for side in SIDES:
        group = ranked[(ranked[side_column] == side) & (ranked._score > 0)].sort_values(
            ["_score", "bodyId"], ascending=[False, True], kind="mergesort"
        )
        if len(group) < per_side:
            raise ValueError(
                f"Only {len(group)} connected {label} candidates on side {side}; "
                f"need {per_side}"
            )
        chosen.extend(map(int, group.head(per_side).bodyId))
    return chosen


def _coverage_map(
    sensory_intrinsic: pd.DataFrame,
    intrinsic_projection: pd.DataFrame,
    output_ids: set[int],
) -> tuple[dict[int, set[int]], dict[int, float]]:
    """Return projection coverage and two-edge strength for every sensory cell."""

    selected_ip = intrinsic_projection[intrinsic_projection.body_post.isin(output_ids)]
    outputs_by_intrinsic: dict[int, set[int]] = defaultdict(set)
    output_mass: dict[int, float] = {}
    for intrinsic, group in selected_ip.groupby("body_pre", sort=True):
        body = int(intrinsic)
        outputs_by_intrinsic[body] = set(map(int, group.body_post))
        output_mass[body] = float(np.log1p(group.weight.astype(float)).sum())

    coverage: dict[int, set[int]] = defaultdict(set)
    strength: dict[int, float] = defaultdict(float)
    for row in sensory_intrinsic.itertuples(index=False):
        sensory = int(row.body_pre)
        intrinsic = int(row.body_post)
        reachable = outputs_by_intrinsic.get(intrinsic)
        if not reachable:
            continue
        coverage[sensory].update(reachable)
        strength[sensory] += float(np.log1p(float(row.weight))) * output_mass[intrinsic]
    return dict(coverage), dict(strength)


def _balanced_greedy_cover(
    candidates: Iterable[int],
    coverage: dict[int, set[tuple[str, int]] | set[int]],
    score: dict[int, float],
    side_by_id: dict[int, str],
    required: set[tuple[str, int]] | set[int],
    *,
    per_side: int,
    label: str,
) -> list[int]:
    """Choose a deterministic balanced fixed-size set covering ``required``."""

    pool = sorted(set(map(int, candidates)))
    if any(side_by_id.get(body) not in SIDES for body in pool):
        raise ValueError(f"Every {label} candidate must have a declared L/R side")
    selected: list[int] = []
    counts = {side: 0 for side in SIDES}
    uncovered = set(required)
    while uncovered:
        eligible = [
            body
            for body in pool
            if body not in selected
            and counts[side_by_id[body]] < per_side
            and bool(set(coverage.get(body, set())) & uncovered)
        ]
        if not eligible:
            missing = sorted(uncovered, key=str)
            raise ValueError(f"Balanced {label} coverage failed for endpoints: {missing}")
        best = max(
            eligible,
            key=lambda body: (
                len(set(coverage.get(body, set())) & uncovered),
                float(score.get(body, 0.0)),
                -body,
            ),
        )
        selected.append(best)
        counts[side_by_id[best]] += 1
        uncovered -= set(coverage.get(best, set()))

    for side in SIDES:
        ranked = sorted(
            (
                body
                for body in pool
                if body not in selected
                and side_by_id[body] == side
                and coverage.get(body)
            ),
            key=lambda body: (-float(score.get(body, 0.0)), body),
        )
        need = per_side - counts[side]
        if len(ranked) < need:
            raise ValueError(
                f"Only {len(ranked)} remaining path-complete {label} candidates on "
                f"side {side}; need {need}"
            )
        selected.extend(ranked[:need])
        counts[side] += need
    if len(selected) != 2 * per_side or counts != {"L": per_side, "R": per_side}:
        raise RuntimeError(f"Internal balanced {label} selection error")
    return selected


def _select_populations(
    sensory: pd.DataFrame,
    intrinsic: pd.DataFrame,
    projection: pd.DataFrame,
    sensory_intrinsic: pd.DataFrame,
    intrinsic_projection: pd.DataFrame,
    *,
    sensory_count: int,
    intrinsic_count: int,
    projection_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if sensory_count % 2 or intrinsic_count % 2 or projection_count % 2:
        raise ValueError("Optic population counts must split evenly across L/R sides")

    sensory_side = dict(zip(sensory.bodyId.astype(int), sensory.rootSide, strict=True))
    intrinsic_side = dict(zip(intrinsic.bodyId.astype(int), intrinsic.somaSide, strict=True))

    # Stage 1 chooses a balanced seed retina only to rank projection readouts.
    sensory_total = sensory_intrinsic.groupby("body_pre").weight.sum()
    seed_sensory = set(
        _ranked_ids(
            sensory,
            sensory_total,
            side_column="rootSide",
            per_side=sensory_count // 2,
            label="ol_sensory seed",
        )
    )
    reachable_intrinsic = set(
        map(
            int,
            sensory_intrinsic[
                sensory_intrinsic.body_pre.isin(seed_sensory)
            ].body_post,
        )
    )
    projection_score = intrinsic_projection[
        intrinsic_projection.body_pre.isin(reachable_intrinsic)
    ].groupby("body_post").weight.sum()
    selected_projection_ids = set(
        _ranked_ids(
            projection,
            projection_score,
            side_column="somaSide",
            per_side=projection_count // 2,
            label="visual_projection",
        )
    )

    # Stage 2 reselects real photoreceptors against the fixed readouts.  Greedy
    # coverage prevents a high-score-only selection from orphaning a readout.
    sensory_outputs, sensory_strength = _coverage_map(
        sensory_intrinsic, intrinsic_projection, selected_projection_ids
    )
    sensory_coverage = {
        body: set(outputs) for body, outputs in sensory_outputs.items() if outputs
    }
    selected_sensory_ids = set(
        _balanced_greedy_cover(
            sensory_coverage,
            sensory_coverage,
            sensory_strength,
            sensory_side,
            selected_projection_ids,
            per_side=sensory_count // 2,
            label="ol_sensory",
        )
    )

    selected_si = sensory_intrinsic[
        sensory_intrinsic.body_pre.isin(selected_sensory_ids)
    ]
    selected_ip = intrinsic_projection[
        intrinsic_projection.body_post.isin(selected_projection_ids)
    ]
    direct_ids = set(map(int, selected_si.body_post)) & set(map(int, selected_ip.body_pre))
    if not direct_ids:
        raise ValueError("No direct optic intrinsic paths join selected inputs and outputs")

    incoming = selected_si.groupby("body_post").weight.sum()
    outgoing = selected_ip.groupby("body_pre").weight.sum()
    bridge_score = {
        body: float(np.sqrt((1.0 + float(incoming[body])) * (1.0 + float(outgoing[body]))))
        for body in direct_ids
    }
    intrinsic_coverage: dict[int, set[tuple[str, int]]] = {}
    for body in direct_ids:
        covered_inputs = set(
            map(int, selected_si[selected_si.body_post == body].body_pre)
        )
        covered_outputs = set(
            map(int, selected_ip[selected_ip.body_pre == body].body_post)
        )
        intrinsic_coverage[body] = {
            *(("sensory", endpoint) for endpoint in covered_inputs),
            *(("projection", endpoint) for endpoint in covered_outputs),
        }
    required_endpoints = {
        *(("sensory", body) for body in selected_sensory_ids),
        *(("projection", body) for body in selected_projection_ids),
    }
    selected_intrinsic_ids = set(
        _balanced_greedy_cover(
            direct_ids,
            intrinsic_coverage,
            bridge_score,
            intrinsic_side,
            required_endpoints,
            per_side=intrinsic_count // 2,
            label="ol_intrinsic",
        )
    )

    selected_sensory = sensory[sensory.bodyId.isin(selected_sensory_ids)].copy()
    selected_intrinsic = intrinsic[intrinsic.bodyId.isin(selected_intrinsic_ids)].copy()
    selected_projection = projection[
        projection.bodyId.isin(selected_projection_ids)
    ].copy()
    if (
        len(selected_sensory) != sensory_count
        or len(selected_intrinsic) != intrinsic_count
        or len(selected_projection) != projection_count
    ):
        raise RuntimeError("Internal optic fixed-size selection error")
    selection_audit = {
        "seed_sensory_count": len(seed_sensory),
        "seed_reachable_intrinsic_count": len(reachable_intrinsic),
        "direct_intrinsic_candidate_count": len(direct_ids),
        "sensory_selection_rule": (
            "balanced deterministic greedy coverage of fixed projection readouts; "
            "ties by descending two-edge log-count strength then ascending bodyId"
        ),
        "intrinsic_selection_rule": (
            "balanced deterministic greedy endpoint coverage; fill by descending "
            "sqrt((1+incoming raw count)*(1+outgoing raw count)), then ascending bodyId"
        ),
    }
    return selected_sensory, selected_intrinsic, selected_projection, selection_audit


def prepare(
    source_dir: Path,
    output_dir: Path,
    *,
    sensory_count: int = 32,
    intrinsic_count: int = 200,
    projection_count: int = 24,
    recurrent_gain: float = 0.5,
    lif_threshold: float = 0.05,
    surrogate_beta: float = 1.0,
) -> dict[str, Any]:
    if (sensory_count, intrinsic_count, projection_count) != (32, 200, 24):
        raise ValueError(
            "Primary optic circuit is frozen at 32 sensory, 200 intrinsic, "
            "and 24 visual-projection neurons"
        )
    if not 0 < recurrent_gain <= 1:
        raise ValueError("recurrent_gain must be in (0, 1]")
    if not np.isfinite(lif_threshold) or lif_threshold <= 0:
        raise ValueError("lif_threshold must be positive and finite")
    if not np.isfinite(surrogate_beta) or surrogate_beta <= 0:
        raise ValueError("surrogate_beta must be positive and finite")

    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    paths = {key: source_dir / filename for key, filename in SOURCE_FILES.items()}
    source_details = _verify_sources(paths)

    annotations = feather.read_table(
        paths["annotations"], columns=OPTIC_ANNOTATION_COLUMNS
    ).to_pandas()
    neurotransmitters = feather.read_table(
        paths["neurotransmitters"], columns=NT_COLUMNS
    ).to_pandas()
    if annotations.bodyId.isna().any() or annotations.bodyId.duplicated().any():
        raise ValueError("Annotation bodyId values must be present and unique")
    annotations = annotations.merge(
        neurotransmitters,
        left_on="bodyId",
        right_on="body",
        how="left",
        validate="one_to_one",
    )
    annotations["bodyId"] = annotations.bodyId.astype("int64")

    good_status = annotations.statusLabel.isin(ALLOWED_STATUS_LABELS)
    sensory = annotations[
        (annotations.superclass == "ol_sensory")
        & (annotations.consensus_nt == "histamine")
        & annotations.rootSide.isin(SIDES)
        & good_status
    ].copy()
    intrinsic = annotations[
        (annotations.superclass == "ol_intrinsic")
        & annotations.consensus_nt.isin(("acetylcholine", "gaba"))
        & annotations.somaSide.isin(SIDES)
        & good_status
    ].copy()
    projection = annotations[
        (annotations.superclass == "visual_projection")
        & annotations.somaSide.isin(SIDES)
        & good_status
    ].copy()
    pool = pd.concat([sensory, intrinsic, projection], ignore_index=True)
    if pool.bodyId.duplicated().any():
        raise ValueError("An optic candidate belongs to multiple biological roles")
    if len(sensory) < sensory_count:
        raise ValueError(
            f"Only {len(sensory)} genuinely annotated optic sensory cells qualify; "
            f"need {sensory_count}"
        )

    sensory_intrinsic, intrinsic_projection, source_edge_rows, cross_counts = (
        _read_cross_layers(
            paths["connectivity"],
            sensory.bodyId.to_numpy(dtype=np.int64),
            intrinsic.bodyId.to_numpy(dtype=np.int64),
            projection.bodyId.to_numpy(dtype=np.int64),
        )
    )
    selected_s, selected_i, selected_p, selection_audit = _select_populations(
        sensory,
        intrinsic,
        projection,
        sensory_intrinsic,
        intrinsic_projection,
        sensory_count=sensory_count,
        intrinsic_count=intrinsic_count,
        projection_count=projection_count,
    )
    sensory_ids = set(map(int, selected_s.bodyId))
    intrinsic_ids = set(map(int, selected_i.bodyId))
    projection_ids = set(map(int, selected_p.bodyId))
    input_ids = sensory_ids
    output_ids = projection_ids
    selected_ids = sensory_ids | intrinsic_ids | projection_ids
    selected = pool[pool.bodyId.isin(selected_ids)].sort_values("bodyId", kind="mergesort")

    selected_edges, selected_raw_edge_count = _read_selected_edges(
        paths["connectivity"], selected_ids
    )
    selected_edges = selected_edges[
        ~selected_edges.body_pre.isin(projection_ids)
        & (selected_edges.body_pre != selected_edges.body_post)
    ].copy()
    selected_edges = (
        selected_edges.groupby(["body_pre", "body_post"], as_index=False, sort=True)
        .weight.sum()
        .sort_values(["body_pre", "body_post"], kind="mergesort")
        .reset_index(drop=True)
    )
    good_inputs, good_outputs = _reaches_outputs(
        input_ids, output_ids, selected_ids, selected_edges
    )
    path_nodes = _nodes_on_input_output_paths(
        input_ids, output_ids, selected_ids, selected_edges
    )
    if (
        len(selected) != 256
        or good_inputs != input_ids
        or good_outputs != output_ids
        or path_nodes != selected_ids
    ):
        raise ValueError(
            "Optic circuit failed its 256-cell complete sensory-to-projection path contract"
        )

    selected_edges["log_count"] = np.log1p(selected_edges.weight.astype(float))
    incoming_mass = selected_edges.groupby("body_post").log_count.transform("sum")
    if (incoming_mass <= 0).any():
        raise ValueError("Every derived edge must have positive incoming log-count mass")
    selected_edges["model_weight"] = (
        recurrent_gain * selected_edges.log_count / incoming_mass
    )
    transmitter_by_id = dict(
        zip(selected.bodyId.astype(int), selected.consensus_nt, strict=True)
    )
    selected_edges["transmitter"] = selected_edges.body_pre.map(transmitter_by_id)
    selected_edges["sign"] = selected_edges.transmitter.map(OPTIC_SIGNS)
    if selected_edges.sign.isna().any():
        bad = sorted(set(selected_edges[selected_edges.sign.isna()].body_pre.astype(int)))
        raise ValueError(f"Selected optic source neurons have no declared model sign: {bad}")
    selected_edges["model_weight"] *= selected_edges.sign

    annotation_fields = list(
        dict.fromkeys(
            OPTIC_ANNOTATION_COLUMNS
            + [
                "consensus_nt",
                "predicted_nt",
                "predicted_nt_confidence",
                "ground_truth",
            ]
        )
    )
    neuron_records: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        body = int(row.bodyId)
        if body in sensory_ids:
            role = "optic_sensory_input"
        elif body in projection_ids:
            role = "visual_projection_output"
        else:
            role = "optic_intrinsic_interneuron"
        neuron_records.append(
            {
                "id": str(body),
                "annotations": {
                    **{field: _clean(row[field]) for field in annotation_fields},
                    "model_role": role,
                },
            }
        )
    edge_records = [
        {
            "pre": str(int(row.body_pre)),
            "post": str(int(row.body_post)),
            "raw_synapse_count": int(row.weight),
            "transmitter": str(row.transmitter),
            "sign": int(row.sign),
            "weight": float(row.model_weight),
        }
        for row in selected_edges.itertuples(index=False)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    neurons_path = output_dir / "neurons.json"
    edges_path = output_dir / "edges.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(neurons_path, neuron_records)
    _write_json(edges_path, edge_records)

    role_counts = {
        str(key): int(value)
        for key, value in pd.Series(
            [record["annotations"]["model_role"] for record in neuron_records]
        ).value_counts().sort_index().items()
    }
    derived_counts = {
        "neurons": len(neuron_records),
        "edges": len(edge_records),
        "sensory_inputs": len(sensory_ids),
        "intrinsic": len(intrinsic_ids),
        "projection_outputs": len(projection_ids),
        "excitatory_edges": int((selected_edges.sign > 0).sum()),
        "inhibitory_edges": int((selected_edges.sign < 0).sum()),
        "input_path_coverage": len(good_inputs),
        "output_path_coverage": len(good_outputs),
        "neurons_on_input_output_paths": len(path_nodes),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "real",
        "source_release": "MaleCNS v1.0 flat connectome, minconf 0.5",
        "source_files": source_details,
        "extraction_rule": (
            "Pinned MaleCNS superclass roles only: balanced real ol_sensory inputs, "
            "balanced signed ol_intrinsic interneurons, and balanced real "
            "visual_projection terminal readouts. Deterministic two-edge endpoint "
            "coverage and bodyId tie breaks require all 256 cells to lie on a "
            "sensory-to-projection path."
        ),
        "neuron_subset": (
            "256-neuron bilateral optic-lobe circuit: 32 ol_sensory, "
            "200 ol_intrinsic, 24 visual_projection"
        ),
        "selection_settings": {
            "allowed_status_labels": list(ALLOWED_STATUS_LABELS),
            "sides": list(SIDES),
            "sensory_count": sensory_count,
            "intrinsic_count": intrinsic_count,
            "projection_count": projection_count,
            "per_side_counts": {
                "sensory": sensory_count // 2,
                "intrinsic": intrinsic_count // 2,
                "projection": projection_count // 2,
            },
            "sensory_annotation": {
                "superclass": "ol_sensory",
                "consensus_nt": "histamine",
                "side_field": "rootSide",
            },
            "intrinsic_annotation": {
                "superclass": "ol_intrinsic",
                "allowed_consensus_nt": ["acetylcholine", "gaba"],
                "side_field": "somaSide",
                "excluded_consensus_nt_reason": (
                    "Glutamate polarity is receptor-dependent and is not inferred "
                    "from transmitter annotation alone."
                ),
            },
            "projection_annotation": {
                "superclass": "visual_projection",
                "side_field": "somaSide",
                "terminal_readout": True,
            },
            "selection_audit": selection_audit,
            "excluded_edges": "all visual_projection-origin and self-loop edges",
        },
        "input_neuron_ids": [str(body) for body in sorted(input_ids)],
        "output_neuron_ids": [str(body) for body in sorted(output_ids)],
        "duplicate_edge_policy": "reject",
        "raw_duplicate_edge_policy": (
            "Aggregate duplicate raw records by directed (body_pre, body_post) "
            "before model-weight mapping; derived duplicate edges are forbidden."
        ),
        "weight_mapping": (
            "Model assumption, not measured conductance: histamine=-1, GABA=-1, "
            f"ACh=+1; weight=sign*{recurrent_gain}*log1p(raw_synapse_count)/"
            "sum_incoming(log1p(count)) per selected post. visual_projection cells "
            "are terminal readouts."
        ),
        "sign_assumptions": {
            "acetylcholine": {
                "sign": 1,
                "interpretation": "excitatory model current",
            },
            "gaba": {"sign": -1, "interpretation": "inhibitory model current"},
            "histamine": {
                "sign": -1,
                "interpretation": (
                    "inhibitory photoreceptor model current; receptor details are "
                    "not represented"
                ),
            },
        },
        "neuron_model": {
            "dt": 0.004,
            "tau_membrane": 0.020,
            "tau_synapse": 0.010,
            "threshold": lif_threshold,
            "reset_value": 0.0,
            "refractory_steps": 1,
            "surrogate_beta": surrogate_beta,
            "neural_substeps": 5,
            "time_mapping": (
                "Five 4 ms neural substeps per 20 ms policy/control step; "
                "engineering choice."
            ),
        },
        "neurons_path": neurons_path.name,
        "edges_path": edges_path.name,
        "checksums": {
            "neurons": _sha256(neurons_path),
            "edges": _sha256(edges_path),
        },
        "derived_counts": derived_counts,
        "role_counts": role_counts,
        "side_counts": {
            "sensory_root_side": {
                str(key): int(value)
                for key, value in selected_s.rootSide.value_counts().sort_index().items()
            },
            "intrinsic_soma_side": {
                str(key): int(value)
                for key, value in selected_i.somaSide.value_counts().sort_index().items()
            },
            "projection_soma_side": {
                str(key): int(value)
                for key, value in selected_p.somaSide.value_counts().sort_index().items()
            },
        },
    }
    manifest["selection_fingerprint"] = _canonical_sha256(
        {
            "inputs": manifest["input_neuron_ids"],
            "outputs": manifest["output_neuron_ids"],
            "selected_neurons_sha256": manifest["checksums"]["neurons"],
            "settings": manifest["selection_settings"],
        }
    )
    _write_json(manifest_path, manifest)

    audit = {
        "status": "PASS",
        "checks": {
            "pinned_source_sha256": True,
            "genuine_superclass_roles": True,
            "fixed_size_256": len(selected_ids) == 256,
            "bilateral_population_balance": manifest["side_counts"]
            == {
                "sensory_root_side": {"L": 16, "R": 16},
                "intrinsic_soma_side": {"L": 100, "R": 100},
                "projection_soma_side": {"L": 12, "R": 12},
            },
            "all_inputs_reach_output": good_inputs == input_ids,
            "all_outputs_reached": good_outputs == output_ids,
            "all_neurons_on_input_output_path": path_nodes == selected_ids,
            "known_source_signs": not selected_edges.sign.isna().any(),
            "no_projection_origin_edges": not selected_edges.body_pre.isin(
                projection_ids
            ).any(),
            "no_self_loops": not (
                selected_edges.body_pre == selected_edges.body_post
            ).any(),
            "finite_nonzero_model_weights": bool(
                np.isfinite(selected_edges.model_weight).all()
                and (selected_edges.model_weight != 0).all()
            ),
        },
        "source_rows": {
            "connectivity": source_edge_rows,
            "annotations": len(annotations),
            "neurotransmitters": len(neurotransmitters),
        },
        "candidate_neurons": {
            "sensory": len(sensory),
            "intrinsic": len(intrinsic),
            "projection": len(projection),
        },
        "candidate_cross_edges_raw": cross_counts,
        "selected_induced_edges_raw": selected_raw_edge_count,
        "derived_counts": derived_counts,
        "role_counts": role_counts,
        "side_counts": manifest["side_counts"],
        "source_sha256": {
            key: value["sha256"] for key, value in source_details.items()
        },
        "derived_sha256": manifest["checksums"],
        "manifest_sha256": _sha256(manifest_path),
        "selection_fingerprint": manifest["selection_fingerprint"],
        "weight_and_sign_assumptions": {
            "mapping": manifest["weight_mapping"],
            "signs": manifest["sign_assumptions"],
            "caveat": (
                "The transform preserves directed topology, raw-count ordering "
                "within each postsynaptic normalization, and declared model sign; "
                "it does not estimate biophysical conductance."
            ),
        },
    }
    if not all(audit["checks"].values()):
        raise RuntimeError(f"Optic audit failed: {audit['checks']}")
    _write_json(output_dir / "audit.json", audit)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/connectome/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/connectome_optic"))
    parser.add_argument("--lif-threshold", type=float, default=0.05)
    parser.add_argument("--recurrent-gain", type=float, default=0.5)
    parser.add_argument("--surrogate-beta", type=float, default=1.0)
    args = parser.parse_args()
    manifest = prepare(
        args.source_dir,
        args.output_dir,
        lif_threshold=args.lif_threshold,
        recurrent_gain=args.recurrent_gain,
        surrogate_beta=args.surrogate_beta,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str((args.output_dir / "manifest.json").resolve()),
                "derived_counts": manifest["derived_counts"],
                "role_counts": manifest["role_counts"],
                "side_counts": manifest["side_counts"],
                "selection_fingerprint": manifest["selection_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
