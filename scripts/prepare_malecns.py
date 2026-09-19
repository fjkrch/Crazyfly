#!/usr/bin/env python3
"""Derive a small, provenance-tracked leg VNC circuit from MaleCNS v1.0.

The supplied Feather weights are synapse counts, not membrane conductances. This
script selects an anatomically constrained graph and maps counts to dimensionless
LIF model weights. It does not infer neuron physiology from the connectome.
"""

from __future__ import annotations

import argparse
from collections import deque
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather


SOURCE_FILES = {
    "connectivity": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}
SOURCE_BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
LEG_NERVES = ("ProLN", "MesoLN", "MetaLN")
SIDES = ("L", "R")
KNOWN_SIGNS = {"acetylcholine": 1, "gaba": -1}
ANNOTATION_COLUMNS = [
    "bodyId", "superclass", "class", "subclass", "type", "instance",
    "somaSide", "rootSide", "somaNeuromere", "entryNerve", "exitNerve",
    "statusLabel",
]
NT_COLUMNS = ["body", "consensus_nt", "predicted_nt", "predicted_nt_confidence", "ground_truth"]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _clean(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value) if not isinstance(value, (str, int, float, bool)) else value


def _rank(frame: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    return (
        frame.assign(_score=frame.bodyId.map(score).fillna(0))
        .sort_values(["_score", "bodyId"], ascending=[False, True], kind="mergesort")
    )


def _group_selection(ranked: pd.DataFrame, column: str, per_group: int, label: str) -> pd.DataFrame:
    selected = []
    for nerve in LEG_NERVES:
        for side in SIDES:
            group = ranked[(ranked[column] == nerve) & (ranked["rootSide" if label == "sensory" else "somaSide"] == side)]
            if len(group) < per_group:
                raise ValueError(f"Not enough {label} cells for {nerve}/{side}: {len(group)} < {per_group}")
            selected.append(group.head(per_group))
    return pd.concat(selected, ignore_index=True)


def _reaches_outputs(inputs: set[int], outputs: set[int], selected: set[int], edges: pd.DataFrame) -> tuple[set[int], set[int]]:
    internal = edges[edges.body_pre.isin(selected) & edges.body_post.isin(selected)]
    adjacency: dict[int, set[int]] = {}
    for pre, post in zip(internal.body_pre, internal.body_post, strict=True):
        adjacency.setdefault(int(pre), set()).add(int(post))

    def reachable(start: int) -> set[int]:
        seen = {start}
        queue = deque([start])
        while queue:
            for node in adjacency.get(queue.popleft(), ()):
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        return seen

    reached = {node: reachable(node) for node in inputs}
    good_inputs = {node for node, nodes in reached.items() if nodes & outputs}
    good_outputs = set().union(*(nodes & outputs for nodes in reached.values())) if reached else set()
    return good_inputs, good_outputs


def _nodes_on_input_output_paths(inputs: set[int], outputs: set[int], selected: set[int], edges: pd.DataFrame) -> set[int]:
    forward: dict[int, set[int]] = {node: set() for node in selected}
    backward: dict[int, set[int]] = {node: set() for node in selected}
    for pre, post in zip(edges.body_pre, edges.body_post, strict=True):
        source, target = int(pre), int(post)
        if source in selected and target in selected:
            forward[source].add(target)
            backward[target].add(source)

    def flood(starts: set[int], graph: dict[int, set[int]]) -> set[int]:
        seen = set(starts)
        queue = deque(starts)
        while queue:
            for node in graph[queue.popleft()]:
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        return seen

    return flood(inputs, forward) & flood(outputs, backward)


def _repair_inputs(
    selected_s: pd.DataFrame,
    selected_d: pd.DataFrame,
    ranked_s: pd.DataFrame,
    ranked_d: pd.DataFrame,
    fixed_ids: set[int],
    output_ids: set[int],
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace disconnected inputs with the next-ranked same-group connected cell."""
    s_ids = set(map(int, selected_s.bodyId))
    d_ids = set(map(int, selected_d.bodyId))
    for _ in range(len(s_ids) + len(d_ids)):
        inputs = s_ids | d_ids
        good, _ = _reaches_outputs(inputs, output_ids, fixed_ids | inputs, edges)
        missing = sorted(inputs - good)
        if not missing:
            break
        old = missing[0]
        if old in s_ids:
            row = selected_s[selected_s.bodyId == old].iloc[0]
            pool = ranked_s[(ranked_s.entryNerve == row.entryNerve) & (ranked_s.rootSide == row.rootSide)]
        else:
            pool = ranked_d
        replacement = None
        for candidate in map(int, pool.bodyId):
            if candidate in inputs:
                continue
            proposed = (inputs - {old}) | {candidate}
            proposed_good, _ = _reaches_outputs(proposed, output_ids, fixed_ids | proposed, edges)
            if candidate in proposed_good and (good - {old}) <= proposed_good:
                replacement = candidate
                break
        if replacement is None:
            raise ValueError(f"Could not connect selected input {old} to a selected motor readout")
        if old in s_ids:
            s_ids.remove(old)
            s_ids.add(replacement)
        else:
            d_ids.remove(old)
            d_ids.add(replacement)
    else:
        raise ValueError("Input connectivity repair did not converge")
    return ranked_s[ranked_s.bodyId.isin(s_ids)], ranked_d[ranked_d.bodyId.isin(d_ids)]


def prepare(
    source_dir: Path,
    output_dir: Path,
    *,
    sensory_per_group: int = 4,
    motor_per_group: int = 4,
    descending_count: int = 8,
    interneuron_count: int = 200,
    status: str = "real",
    recurrent_gain: float = 0.5,
    lif_threshold: float = 0.047,
    surrogate_beta: float = 1.0,
) -> dict[str, Any]:
    if min(sensory_per_group, motor_per_group, descending_count, interneuron_count) < 1:
        raise ValueError("All subset counts must be positive")
    if not 0 < recurrent_gain <= 1:
        raise ValueError("recurrent_gain must be in (0, 1]")
    if not np.isfinite(lif_threshold) or lif_threshold <= 0:
        raise ValueError("lif_threshold must be positive and finite")
    if not np.isfinite(surrogate_beta) or surrogate_beta <= 0:
        raise ValueError("surrogate_beta must be positive and finite")
    if status not in {"real", "synthetic"}:
        raise ValueError("status must be real or synthetic")
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    paths = {key: source_dir / name for key, name in SOURCE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MaleCNS Feather file(s): " + ", ".join(missing))

    annotations = feather.read_table(paths["annotations"], columns=ANNOTATION_COLUMNS).to_pandas()
    nt = feather.read_table(paths["neurotransmitters"], columns=NT_COLUMNS).to_pandas()
    if annotations.bodyId.isna().any() or annotations.bodyId.duplicated().any():
        raise ValueError("Annotation bodyId values must be present and unique")
    if nt.body.isna().any() or nt.body.duplicated().any():
        raise ValueError("Neurotransmitter body values must be present and unique")
    annotations = annotations.merge(nt, left_on="bodyId", right_on="body", how="left", validate="one_to_one")
    annotations["bodyId"] = annotations.bodyId.astype("int64")
    known = annotations.consensus_nt.isin(KNOWN_SIGNS)
    intrinsic = annotations[(annotations.superclass == "vnc_intrinsic") & annotations.somaNeuromere.isin(("T1", "T2", "T3")) & known]
    sensory = annotations[(annotations.superclass == "vnc_sensory") & (annotations["class"] == "mechanosensory_proprioceptive") & annotations.entryNerve.isin(LEG_NERVES) & known]
    motor = annotations[(annotations.superclass == "vnc_motor") & annotations.exitNerve.isin(LEG_NERVES)]
    descending = annotations[(annotations.superclass == "descending_neuron") & known]
    pool = pd.concat([intrinsic, sensory, motor, descending], ignore_index=True)
    if pool.bodyId.duplicated().any():
        raise ValueError("An annotated body belongs to multiple selection classes")

    # Feather v2 is an Arrow IPC file. Process its record batches (65,536 rows
    # in the real release) so the 152M-row graph is never resident in RAM.
    reader = pa.ipc.open_file(pa.memory_map(str(paths["connectivity"]), "r"))
    candidate_ids = pa.array(pool.bodyId.to_numpy())
    candidate_batches = []
    source_edge_rows = 0
    for batch_number in range(reader.num_record_batches):
        batch = reader.get_batch(batch_number)
        source_edge_rows += batch.num_rows
        mask = pc.and_(
            pc.is_in(batch.column("body_pre"), value_set=candidate_ids),
            pc.is_in(batch.column("body_post"), value_set=candidate_ids),
        )
        filtered = batch.filter(mask)
        if filtered.num_rows:
            candidate_batches.append(filtered)
    if not candidate_batches:
        raise ValueError("No connectivity edges join the annotated candidate pool")
    candidate_edge_count = sum(batch.num_rows for batch in candidate_batches)
    edges = pa.Table.from_batches(candidate_batches).to_pandas()
    del candidate_batches, reader
    if edges.empty or edges.weight.isna().any() or (edges.weight <= 0).any():
        raise ValueError("Candidate edges must include positive finite synapse counts")
    # Motor cells are terminal decoder readouts; their transmitter is often
    # glutamate/unclear and cannot be assigned a signed central synapse here.
    edges = edges[~edges.body_pre.isin(motor.bodyId) & (edges.body_pre != edges.body_post)].copy()
    labels = {int(body): label for label, frame in (("I", intrinsic), ("S", sensory), ("M", motor), ("D", descending)) for body in frame.bodyId}
    edges["pre_class"] = edges.body_pre.map(labels)
    edges["post_class"] = edges.body_post.map(labels)
    sensory_score = edges[(edges.pre_class == "S") & (edges.post_class == "I")].groupby("body_pre").weight.sum()
    motor_score = edges[(edges.pre_class == "I") & (edges.post_class == "M")].groupby("body_post").weight.sum()
    descending_score = edges[(edges.pre_class == "D") & (edges.post_class == "I")].groupby("body_pre").weight.sum()
    ranked_s = _rank(sensory, sensory_score)
    ranked_m = _rank(motor, motor_score)
    ranked_d = _rank(descending, descending_score)
    selected_s = _group_selection(ranked_s, "entryNerve", sensory_per_group, "sensory")
    selected_m = _group_selection(ranked_m, "exitNerve", motor_per_group, "motor")
    if len(ranked_d) < descending_count:
        raise ValueError("Not enough signed descending neurons")
    selected_d = ranked_d.head(descending_count)
    selected_input_ids = set(map(int, pd.concat([selected_s.bodyId, selected_d.bodyId])))
    output_ids = set(map(int, selected_m.bodyId))
    incoming = edges[edges.body_pre.isin(selected_input_ids) & (edges.post_class == "I")].groupby("body_post").weight.sum()
    outgoing = edges[edges.body_post.isin(output_ids) & (edges.pre_class == "I")].groupby("body_pre").weight.sum()
    ranked_i = intrinsic.assign(
        _incoming=intrinsic.bodyId.map(incoming).fillna(0),
        _outgoing=intrinsic.bodyId.map(outgoing).fillna(0),
    )
    ranked_i["_bridge"] = np.sqrt((1 + ranked_i._incoming) * (1 + ranked_i._outgoing))
    ranked_i = ranked_i.sort_values(["_bridge", "bodyId"], ascending=[False, True], kind="mergesort")
    if len(ranked_i) < interneuron_count:
        raise ValueError("Not enough signed VNC interneurons")
    selected_i = ranked_i.head(interneuron_count)
    fixed_ids = set(map(int, pd.concat([selected_i.bodyId, selected_m.bodyId])))
    selected_s, selected_d = _repair_inputs(selected_s, selected_d, ranked_s, ranked_d, fixed_ids, output_ids, edges)
    input_ids = set(map(int, pd.concat([selected_s.bodyId, selected_d.bodyId])))
    selected_ids = fixed_ids | input_ids
    good_inputs, good_outputs = _reaches_outputs(input_ids, output_ids, selected_ids, edges)
    if good_inputs != input_ids or good_outputs != output_ids:
        raise ValueError("Selected circuit lacks input-to-output path coverage")

    selected = pool[pool.bodyId.isin(selected_ids)].sort_values("bodyId")
    selected_edges = edges[edges.body_pre.isin(selected_ids) & edges.body_post.isin(selected_ids)]
    selected_edges = selected_edges.groupby(["body_pre", "body_post"], as_index=False).weight.sum()
    if selected_edges.empty:
        raise ValueError("Selected circuit has no edges")
    path_nodes = _nodes_on_input_output_paths(input_ids, output_ids, selected_ids, selected_edges)
    if path_nodes != selected_ids:
        raise ValueError(f"{len(selected_ids - path_nodes)} selected neurons are not on an input-to-output path")
    # Signed, per-post L1 normalization bounds the original circuit's absolute
    # incoming model weight by recurrent_gain. After degree-preserving rewiring,
    # the per-post bound need not hold; that difference is measured in comparisons.
    selected_edges["log_count"] = np.log1p(selected_edges.weight.astype(float))
    post_mass = selected_edges.groupby("body_post").log_count.transform("sum")
    selected_edges["model_weight"] = recurrent_gain * selected_edges.log_count / post_mass
    nt_by_id = dict(zip(selected.bodyId, selected.consensus_nt, strict=True))
    selected_edges["sign"] = selected_edges.body_pre.map(nt_by_id).map(KNOWN_SIGNS)
    if selected_edges.sign.isna().any():
        raise ValueError("A selected source neuron has no supported sign")
    selected_edges["model_weight"] *= selected_edges.sign
    selected_edges = selected_edges.sort_values(["body_pre", "body_post"])

    neuron_records = []
    fields = ["superclass", "class", "subclass", "type", "instance", "somaSide", "rootSide", "somaNeuromere", "entryNerve", "exitNerve", "statusLabel", "consensus_nt", "predicted_nt", "predicted_nt_confidence", "ground_truth"]
    for _, row in selected.iterrows():
        body = int(row.bodyId)
        role = "sensory_input" if body in set(map(int, selected_s.bodyId)) else "descending_input" if body in set(map(int, selected_d.bodyId)) else "motor_output" if body in output_ids else "vnc_interneuron"
        neuron_records.append({"id": str(body), "annotations": {**{field: _clean(row[field]) for field in fields}, "model_role": role}})
    edge_records = [
        {
            "pre": str(int(row.body_pre)), "post": str(int(row.body_post)),
            "raw_synapse_count": int(row.weight), "weight": float(row.model_weight),
        }
        for row in selected_edges.itertuples(index=False)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    neurons_path = output_dir / "neurons.json"
    edges_path = output_dir / "edges.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(neurons_path, neuron_records)
    _write_json(edges_path, edge_records)
    source_details = {
        key: {"filename": path.name, "sha256": _sha256(path), "url": SOURCE_BASE_URL + path.name}
        for key, path in paths.items()
    }
    manifest = {
        "schema_version": 1,
        "status": status,
        "source_release": "MaleCNS v1.0 flat connectome, minconf 0.5",
        "source_files": source_details,
        "extraction_rule": "Leg-nerve proprioceptive sensory and motor readouts; signed descending and thoracic VNC interneurons ranked by weighted source connectivity, with path-repair; see selection_settings.",
        "neuron_subset": f"{len(selected)}-neuron T1/T2/T3 leg VNC sensor-descending-interneuron-motor circuit",
        "selection_settings": {
            "leg_nerves": list(LEG_NERVES), "sides": list(SIDES),
            "sensory_per_nerve_side": sensory_per_group,
            "motor_per_nerve_side": motor_per_group,
            "descending_count": descending_count,
            "interneuron_count": interneuron_count,
            "sensory_rank": "descending sum of raw S->I counts, tie ascending bodyId",
            "motor_rank": "descending sum of raw I->M counts, tie ascending bodyId",
            "descending_rank": "descending sum of raw D->I counts, tie ascending bodyId",
            "interneuron_rank": "descending sqrt((1+count from selected S/D)*(1+count to selected M)), tie ascending bodyId",
            "input_path_repair": "Replace an input with the highest-ranked unused candidate in the same nerve/side or descending pool that reaches a selected motor without disconnecting previously connected inputs; keep selected interneurons fixed.",
            "allowed_source_transmitters": list(KNOWN_SIGNS),
            "excluded_edges": "all motor-origin and self-loop edges",
        },
        "input_neuron_ids": [str(body) for body in sorted(input_ids)],
        "output_neuron_ids": [str(body) for body in sorted(output_ids)],
        "duplicate_edge_policy": "reject",
        "weight_mapping": f"Model assumption: ACh=+1, GABA=-1; weight=sign*{recurrent_gain}*log1p(raw_synapse_count)/sum_pre(log1p(count)) for each selected post. Raw counts are not conductances. Motor cells are terminal readouts, so their ambiguous transmitter labels are not signed.",
        "model_calibration": "Dimensionless LIF threshold selected before the main comparison from 100-control-step G1 encoder-current pilots: threshold 1.0 gave no spikes; 0.05 became nearly motor-silent after one PPO update; 0.047 retained input, interneuron, and motor activity without motor saturation on offline replay. Surrogate beta controls only the backward optimization derivative. These are engineering calibrations, not measured fly parameters or evidence of learned behavior.",
        "neuron_model": {
            "dt": 0.004, "tau_membrane": 0.020, "tau_synapse": 0.010,
            "threshold": lif_threshold, "reset_value": 0.0, "refractory_steps": 1,
            "surrogate_beta": surrogate_beta, "neural_substeps": 5,
            "time_mapping": "Five 4 ms neural substeps per 20 ms policy/control step; engineering choice, not measured fly physiology.",
        },
        "neurons_path": neurons_path.name,
        "edges_path": edges_path.name,
        "checksums": {"neurons": _sha256(neurons_path), "edges": _sha256(edges_path)},
        "derived_counts": {
            "neurons": len(neuron_records), "edges": len(edge_records),
            "excitatory_edges": int((selected_edges.sign > 0).sum()),
            "inhibitory_edges": int((selected_edges.sign < 0).sum()),
            "input_path_coverage": len(good_inputs), "output_path_coverage": len(good_outputs),
            "neurons_on_input_output_paths": len(path_nodes),
        },
    }
    _write_json(manifest_path, manifest)
    audit = {
        "source_rows": {
            "connectivity": source_edge_rows,
            "annotations": len(annotations),
            "neurotransmitters": len(nt),
        },
        "candidate_neurons": {
            "sensory": len(sensory), "descending": len(descending),
            "interneurons": len(intrinsic), "motor": len(motor),
        },
        "candidate_induced_edges_before_terminal_filter": candidate_edge_count,
        "selected_neurons": {
            "sensory": len(selected_s), "descending": len(selected_d),
            "interneurons": len(selected_i), "motor": len(selected_m),
        },
        "selected_induced_edges": len(edge_records),
        "selected_neurons_on_input_output_paths": {"covered": len(path_nodes), "total": len(selected_ids)},
        "input_path_coverage": {"covered": len(good_inputs), "total": len(input_ids)},
        "output_path_coverage": {"covered": len(good_outputs), "total": len(output_ids)},
        "source_sha256": {key: details["sha256"] for key, details in source_details.items()},
        "derived_sha256": manifest["checksums"],
    }
    _write_json(output_dir / "audit.json", audit)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output-dir", type=Path, default=Path("data/connectome"))
    parser.add_argument("--sensory-per-group", type=int, default=4)
    parser.add_argument("--motor-per-group", type=int, default=4)
    parser.add_argument("--descending-count", type=int, default=8)
    parser.add_argument("--interneuron-count", type=int, default=200)
    parser.add_argument("--status", choices=("real", "synthetic"), default="real")
    parser.add_argument("--recurrent-gain", type=float, default=0.5)
    parser.add_argument("--lif-threshold", type=float, default=0.047)
    parser.add_argument("--surrogate-beta", type=float, default=1.0)
    args = parser.parse_args()
    manifest = prepare(
        args.source_dir, args.output_dir,
        sensory_per_group=args.sensory_per_group,
        motor_per_group=args.motor_per_group,
        descending_count=args.descending_count,
        interneuron_count=args.interneuron_count,
        status=args.status,
        recurrent_gain=args.recurrent_gain,
        lif_threshold=args.lif_threshold,
        surrogate_beta=args.surrogate_beta,
    )
    print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), **manifest["derived_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
