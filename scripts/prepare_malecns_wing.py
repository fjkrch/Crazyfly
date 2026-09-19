#!/usr/bin/env python3
"""Derive a deterministic, provenance-tracked wing VNC circuit from MaleCNS.

This extractor is deliberately independent of ``data/connectome``.  It uses
the twenty annotated cholinergic wing mechanosensors, twelve ranked descending
inputs, two hundred thoracic intrinsic cells, and twenty-four ranked wing motor
readouts.  Raw synapse counts are graph evidence, not conductances.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from prepare_malecns import (
    ANNOTATION_COLUMNS,
    KNOWN_SIGNS,
    NT_COLUMNS,
    SOURCE_BASE_URL,
    SOURCE_FILES,
    _clean,
    _nodes_on_input_output_paths,
    _rank,
    _reaches_outputs,
    _sha256,
    _write_json,
)


EXPECTED_SOURCE_SHA256 = {
    "annotations": "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
    "connectivity": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    "neurotransmitters": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
}
WING_SENSORY_CLASSES = ("mechanosensory_proprioceptive", "mechanosensory_tactile")
WING_MOTOR_NERVES = ("ADMN", "MesoAN", "PDMNa", "PDMNp")
SIDES = ("L", "R")


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
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
                f"Pinned MaleCNS {label} SHA-256 mismatch: expected={expected}, actual={actual}"
            )
        details[label] = {
            "filename": path.name,
            "sha256": actual,
            "url": SOURCE_BASE_URL + path.name,
        }
    return details


def _read_candidate_edges(path: Path, candidate_ids: np.ndarray) -> tuple[pd.DataFrame, int, int]:
    reader = pa.ipc.open_file(pa.memory_map(str(path), "r"))
    value_set = pa.array(candidate_ids)
    batches = []
    source_rows = 0
    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number)
        source_rows += batch.num_rows
        mask = pc.and_(
            pc.is_in(batch.column("body_pre"), value_set=value_set),
            pc.is_in(batch.column("body_post"), value_set=value_set),
        )
        selected = batch.filter(mask)
        if selected.num_rows:
            batches.append(selected)
    if not batches:
        raise ValueError("No edges join the wing candidate pool")
    candidate_edges = sum(batch.num_rows for batch in batches)
    result = pa.Table.from_batches(batches).to_pandas()
    del batches, reader
    return result, source_rows, candidate_edges


def _rank_intrinsic(
    intrinsic: pd.DataFrame,
    edges: pd.DataFrame,
    input_ids: set[int],
    output_ids: set[int],
) -> pd.DataFrame:
    incoming = edges[
        edges.body_pre.isin(input_ids) & (edges.post_class == "I")
    ].groupby("body_post").weight.sum()
    outgoing = edges[
        edges.body_post.isin(output_ids) & (edges.pre_class == "I")
    ].groupby("body_pre").weight.sum()
    ranked = intrinsic.assign(
        _incoming=intrinsic.bodyId.map(incoming).fillna(0),
        _outgoing=intrinsic.bodyId.map(outgoing).fillna(0),
    )
    ranked["_bridge"] = np.sqrt((1 + ranked._incoming) * (1 + ranked._outgoing))
    return ranked.sort_values(["_bridge", "bodyId"], ascending=[False, True], kind="mergesort")


def _select_path_complete(
    ranked_s: pd.DataFrame,
    ranked_d: pd.DataFrame,
    ranked_i: pd.DataFrame,
    ranked_m: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    sensory_count: int,
    descending_count: int,
    interneuron_count: int,
    motor_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Select the first deterministic fixed-size combination with full coverage."""

    selected_s = ranked_s.head(sensory_count)
    if len(selected_s) != sensory_count:
        raise ValueError(f"Wing sensory pool has {len(selected_s)} cells, expected {sensory_count}")
    # Sensory membership is biologically predeclared.  Search only bounded
    # deterministic prefixes for descending, motor, and intrinsic replacements.
    max_d_offset = min(64, len(ranked_d) - descending_count + 1)
    max_m_offset = min(44, len(ranked_m) - motor_count + 1)
    max_i_offset = min(256, len(ranked_i) - interneuron_count + 1)
    attempts = 0
    for d_offset in range(max(0, max_d_offset)):
        selected_d = ranked_d.iloc[d_offset : d_offset + descending_count]
        input_ids = set(map(int, pd.concat([selected_s.bodyId, selected_d.bodyId])))
        for m_offset in range(max(0, max_m_offset)):
            selected_m = ranked_m.iloc[m_offset : m_offset + motor_count]
            output_ids = set(map(int, selected_m.bodyId))
            for i_offset in range(max(0, max_i_offset)):
                attempts += 1
                selected_i = ranked_i.iloc[i_offset : i_offset + interneuron_count]
                selected_ids = input_ids | output_ids | set(map(int, selected_i.bodyId))
                good_inputs, good_outputs = _reaches_outputs(
                    input_ids, output_ids, selected_ids, edges
                )
                if good_inputs != input_ids or good_outputs != output_ids:
                    continue
                selected_edges = edges[
                    edges.body_pre.isin(selected_ids) & edges.body_post.isin(selected_ids)
                ]
                path_nodes = _nodes_on_input_output_paths(
                    input_ids, output_ids, selected_ids, selected_edges
                )
                if path_nodes == selected_ids:
                    return selected_s, selected_d, selected_i, selected_m, attempts
    raise ValueError(
        "No deterministic fixed-size wing selection achieved complete input/output/path coverage "
        f"after {attempts} bounded prefix attempts"
    )


def prepare(
    source_dir: Path,
    output_dir: Path,
    *,
    sensory_count: int = 20,
    descending_count: int = 12,
    interneuron_count: int = 200,
    motor_count: int = 24,
    recurrent_gain: float = 0.5,
    lif_threshold: float = 0.05,
    surrogate_beta: float = 1.0,
    calibration_report: Path | None = None,
) -> dict[str, Any]:
    counts = (sensory_count, descending_count, interneuron_count, motor_count)
    if counts != (20, 12, 200, 24):
        raise ValueError("Primary wing circuit is frozen at 20 sensory, 12 descending, 200 intrinsic, 24 motor")
    if not 0 < recurrent_gain <= 1 or not np.isfinite(lif_threshold) or lif_threshold <= 0:
        raise ValueError("Recurrent gain and LIF threshold must be positive and finite")
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    paths = {key: source_dir / filename for key, filename in SOURCE_FILES.items()}
    source_details = _verify_sources(paths)

    annotations = feather.read_table(paths["annotations"], columns=ANNOTATION_COLUMNS).to_pandas()
    nt = feather.read_table(paths["neurotransmitters"], columns=NT_COLUMNS).to_pandas()
    if annotations.bodyId.isna().any() or annotations.bodyId.duplicated().any():
        raise ValueError("Annotation bodyId values must be present and unique")
    annotations = annotations.merge(nt, left_on="bodyId", right_on="body", how="left", validate="one_to_one")
    annotations["bodyId"] = annotations.bodyId.astype("int64")
    known = annotations.consensus_nt.isin(KNOWN_SIGNS)
    sensory = annotations[
        (annotations.superclass == "vnc_sensory")
        & (annotations.subclass == "wing")
        & annotations["class"].isin(WING_SENSORY_CLASSES)
        & (annotations.entryNerve == "ADMN")
        & known
    ]
    motor = annotations[
        (annotations.superclass == "vnc_motor")
        & (annotations.subclass == "wm")
        & annotations.exitNerve.isin(WING_MOTOR_NERVES)
        & annotations.somaSide.isin(SIDES)
    ]
    intrinsic = annotations[
        (annotations.superclass == "vnc_intrinsic")
        & annotations.somaNeuromere.isin(("T1", "T2", "T3"))
        & known
    ]
    descending = annotations[(annotations.superclass == "descending_neuron") & known]
    if len(sensory) != sensory_count:
        raise ValueError(f"Pinned wing sensory selector resolved {len(sensory)} cells, expected {sensory_count}")
    if set(sensory.consensus_nt) != {"acetylcholine"}:
        raise ValueError("Primary wing sensory population must be entirely cholinergic")
    pool = pd.concat([intrinsic, sensory, motor, descending], ignore_index=True)
    if pool.bodyId.duplicated().any():
        raise ValueError("A wing candidate belongs to multiple model roles")

    edges, source_edge_rows, candidate_edge_count = _read_candidate_edges(
        paths["connectivity"], pool.bodyId.to_numpy()
    )
    if edges.weight.isna().any() or (edges.weight <= 0).any():
        raise ValueError("Candidate edges must have positive finite counts")
    edges = edges[~edges.body_pre.isin(motor.bodyId) & (edges.body_pre != edges.body_post)].copy()
    labels = {
        int(body): label
        for label, frame in (("I", intrinsic), ("S", sensory), ("M", motor), ("D", descending))
        for body in frame.bodyId
    }
    edges["pre_class"] = edges.body_pre.map(labels)
    edges["post_class"] = edges.body_post.map(labels)
    sensory_score = edges[(edges.pre_class == "S") & (edges.post_class == "I")].groupby("body_pre").weight.sum()
    motor_score = edges[(edges.pre_class == "I") & (edges.post_class == "M")].groupby("body_post").weight.sum()
    descending_score = edges[(edges.pre_class == "D") & (edges.post_class == "I")].groupby("body_pre").weight.sum()
    ranked_s = _rank(sensory, sensory_score)
    ranked_d = _rank(descending, descending_score)
    ranked_m = _rank(motor, motor_score)
    provisional_inputs = set(map(int, pd.concat([ranked_s.head(sensory_count).bodyId, ranked_d.head(descending_count).bodyId])))
    provisional_outputs = set(map(int, ranked_m.head(motor_count).bodyId))
    ranked_i = _rank_intrinsic(intrinsic, edges, provisional_inputs, provisional_outputs)
    selected_s, selected_d, selected_i, selected_m, attempts = _select_path_complete(
        ranked_s, ranked_d, ranked_i, ranked_m, edges,
        sensory_count=sensory_count,
        descending_count=descending_count,
        interneuron_count=interneuron_count,
        motor_count=motor_count,
    )
    input_ids = set(map(int, pd.concat([selected_s.bodyId, selected_d.bodyId])))
    output_ids = set(map(int, selected_m.bodyId))
    selected_ids = input_ids | output_ids | set(map(int, selected_i.bodyId))
    selected = pool[pool.bodyId.isin(selected_ids)].sort_values("bodyId")
    selected_edges = edges[
        edges.body_pre.isin(selected_ids) & edges.body_post.isin(selected_ids)
    ].groupby(["body_pre", "body_post"], as_index=False).weight.sum()
    path_nodes = _nodes_on_input_output_paths(input_ids, output_ids, selected_ids, selected_edges)
    good_inputs, good_outputs = _reaches_outputs(input_ids, output_ids, selected_ids, selected_edges)
    if len(selected) != 256 or path_nodes != selected_ids or good_inputs != input_ids or good_outputs != output_ids:
        raise ValueError("Wing circuit failed its 256-cell path-completeness contract")

    selected_edges["log_count"] = np.log1p(selected_edges.weight.astype(float))
    selected_edges["model_weight"] = recurrent_gain * selected_edges.log_count / selected_edges.groupby("body_post").log_count.transform("sum")
    nt_by_id = dict(zip(selected.bodyId, selected.consensus_nt, strict=True))
    selected_edges["sign"] = selected_edges.body_pre.map(nt_by_id).map(KNOWN_SIGNS)
    if selected_edges.sign.isna().any():
        raise ValueError("A selected wing source neuron has no supported sign")
    selected_edges["model_weight"] *= selected_edges.sign
    selected_edges = selected_edges.sort_values(["body_pre", "body_post"])

    sensory_ids = set(map(int, selected_s.bodyId))
    descending_ids = set(map(int, selected_d.bodyId))
    fields = [
        "superclass", "class", "subclass", "type", "instance", "somaSide", "rootSide",
        "somaNeuromere", "entryNerve", "exitNerve", "statusLabel", "consensus_nt",
        "predicted_nt", "predicted_nt_confidence", "ground_truth",
    ]
    neuron_records = []
    for _, row in selected.iterrows():
        body = int(row.bodyId)
        role = (
            "wing_sensory_input" if body in sensory_ids else
            "descending_input" if body in descending_ids else
            "wing_motor_output" if body in output_ids else
            "vnc_interneuron"
        )
        neuron_records.append({
            "id": str(body),
            "annotations": {**{field: _clean(row[field]) for field in fields}, "model_role": role},
        })
    edge_records = [
        {
            "pre": str(int(row.body_pre)),
            "post": str(int(row.body_post)),
            "raw_synapse_count": int(row.weight),
            "weight": float(row.model_weight),
        }
        for row in selected_edges.itertuples(index=False)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    neurons_path = output_dir / "neurons.json"
    edges_path = output_dir / "edges.json"
    _write_json(neurons_path, neuron_records)
    _write_json(edges_path, edge_records)

    calibration = {
        "status": "pending_bounded_activity_pilot",
        "selected_threshold": lif_threshold,
        "report": None,
    }
    if calibration_report is not None:
        report_path = Path(calibration_report).expanduser().resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("selected_threshold") != lif_threshold:
            raise ValueError("Wing calibration report does not authorize the requested threshold")
        calibration = {
            "status": "frozen_after_bounded_activity_pilot",
            "selected_threshold": lif_threshold,
            "report": str(report_path),
            "report_sha256": _sha256(report_path),
            "selection_rule": report.get("selection_rule"),
        }

    manifest = {
        "schema_version": 1,
        "status": "real",
        "source_release": "MaleCNS v1.0 flat connectome, minconf 0.5",
        "source_files": source_details,
        "extraction_rule": "Twenty pinned cholinergic ADMN wing mechanosensors, twelve ranked signed descending inputs, two hundred ranked signed thoracic intrinsic cells, and twenty-four ranked wing motor readouts; bounded deterministic prefix search requires full path coverage.",
        "neuron_subset": "256-neuron T1/T2/T3 wing VNC sensor-descending-interneuron-motor circuit",
        "selection_settings": {
            "wing_sensory_entry_nerve": "ADMN",
            "wing_sensory_classes": list(WING_SENSORY_CLASSES),
            "wing_motor_exit_nerves": list(WING_MOTOR_NERVES),
            "sides": list(SIDES),
            "sensory_count": sensory_count,
            "descending_count": descending_count,
            "interneuron_count": interneuron_count,
            "motor_count": motor_count,
            "sensory_rank": "descending raw S->I count, tie ascending bodyId; all 20 retained",
            "descending_rank": "descending raw D->I count, tie ascending bodyId",
            "motor_rank": "descending raw I->M count, tie ascending bodyId",
            "interneuron_rank": "descending sqrt((1+selected-input count)*(1+selected-output count)), tie ascending bodyId",
            "path_completion": "bounded lexicographic prefix-offset search over descending, motor, and intrinsic rankings; first 256-cell solution with every selected node on an input-to-output path",
            "path_completion_attempts": attempts,
            "allowed_source_transmitters": list(KNOWN_SIGNS),
            "excluded_edges": "all motor-origin and self-loop edges",
        },
        "input_neuron_ids": [str(body) for body in sorted(input_ids)],
        "output_neuron_ids": [str(body) for body in sorted(output_ids)],
        "duplicate_edge_policy": "reject",
        "raw_duplicate_edge_policy": (
            "Aggregate duplicate raw records by directed (body_pre, body_post) pair before "
            "model-weight mapping; derived duplicate edges are forbidden."
        ),
        "weight_mapping": f"Model assumption: ACh=+1, GABA=-1; weight=sign*{recurrent_gain}*log1p(raw_synapse_count)/sum_incoming(log1p(count)) per selected post. Raw counts are not conductances; wing motor cells are terminal readouts.",
        "model_calibration": calibration,
        "neuron_model": {
            "dt": 0.004,
            "tau_membrane": 0.020,
            "tau_synapse": 0.010,
            "threshold": lif_threshold,
            "reset_value": 0.0,
            "refractory_steps": 1,
            "surrogate_beta": surrogate_beta,
            "neural_substeps": 5,
            "time_mapping": "Five 4 ms neural substeps per 20 ms policy/control step; engineering choice.",
        },
        "neurons_path": neurons_path.name,
        "edges_path": edges_path.name,
        "checksums": {"neurons": _sha256(neurons_path), "edges": _sha256(edges_path)},
        "derived_counts": {
            "neurons": len(neuron_records),
            "edges": len(edge_records),
            "sensory_inputs": len(sensory_ids),
            "descending_inputs": len(descending_ids),
            "intrinsic": len(selected_i),
            "motor_outputs": len(output_ids),
            "excitatory_edges": int((selected_edges.sign > 0).sum()),
            "inhibitory_edges": int((selected_edges.sign < 0).sum()),
            "input_path_coverage": len(good_inputs),
            "output_path_coverage": len(good_outputs),
            "neurons_on_input_output_paths": len(path_nodes),
        },
        "role_counts": {
            str(key): int(value)
            for key, value in pd.Series([row["annotations"]["model_role"] for row in neuron_records]).value_counts().sort_index().items()
        },
        "motor_nerve_counts": {
            str(key): int(value)
            for key, value in selected_m.exitNerve.value_counts().sort_index().items()
        },
        "motor_side_counts": {
            str(key): int(value)
            for key, value in selected_m.somaSide.value_counts().sort_index().items()
        },
    }
    manifest["selection_fingerprint"] = _canonical_sha256({
        "inputs": manifest["input_neuron_ids"],
        "outputs": manifest["output_neuron_ids"],
        "settings": manifest["selection_settings"],
        "checksums": manifest["checksums"],
    })
    _write_json(output_dir / "manifest.json", manifest)
    audit = {
        "status": "PASS",
        "source_rows": {
            "connectivity": source_edge_rows,
            "annotations": len(annotations),
            "neurotransmitters": len(nt),
        },
        "candidate_neurons": {
            "sensory": len(sensory),
            "descending": len(descending),
            "intrinsic": len(intrinsic),
            "motor": len(motor),
        },
        "candidate_induced_edges_before_terminal_filter": candidate_edge_count,
        "derived_counts": manifest["derived_counts"],
        "role_counts": manifest["role_counts"],
        "motor_nerve_counts": manifest["motor_nerve_counts"],
        "motor_side_counts": manifest["motor_side_counts"],
        "source_sha256": {key: value["sha256"] for key, value in source_details.items()},
        "derived_sha256": manifest["checksums"],
        "selection_fingerprint": manifest["selection_fingerprint"],
    }
    _write_json(output_dir / "audit.json", audit)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/connectome/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/connectome_wing"))
    parser.add_argument("--lif-threshold", type=float, default=0.05)
    parser.add_argument("--recurrent-gain", type=float, default=0.5)
    parser.add_argument("--surrogate-beta", type=float, default=1.0)
    parser.add_argument("--calibration-report", type=Path)
    args = parser.parse_args()
    manifest = prepare(
        args.source_dir,
        args.output_dir,
        lif_threshold=args.lif_threshold,
        recurrent_gain=args.recurrent_gain,
        surrogate_beta=args.surrogate_beta,
        calibration_report=args.calibration_report,
    )
    print(json.dumps({
        "status": "PASS",
        "manifest": str((args.output_dir / "manifest.json").resolve()),
        "derived_counts": manifest["derived_counts"],
        "selection_fingerprint": manifest["selection_fingerprint"],
        "calibration": manifest["model_calibration"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
