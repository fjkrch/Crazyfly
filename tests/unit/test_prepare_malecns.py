"""End-to-end conversion of a tiny Feather circuit, including provenance and signs."""

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
feather = pytest.importorskip("pyarrow.feather")

from scripts.prepare_malecns import ANNOTATION_COLUMNS, NT_COLUMNS, SOURCE_FILES, prepare
from g1_fly_control.connectome.loader import load_connectome


def _write_feather(path: Path, rows: list[dict], columns: list[str]) -> None:
    feather.write_feather(pa.Table.from_pylist([{key: row.get(key) for key in columns} for row in rows]), path)


def test_prepare_small_signed_circuit_from_feather(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "derived"
    source.mkdir()
    annotations = []
    nt = []
    edges = []

    def cell(body: int, superclass: str, transmitter: str, **metadata: str) -> None:
        annotations.append({"bodyId": body, "superclass": superclass, **metadata})
        nt.append({"body": body, "consensus_nt": transmitter, "predicted_nt": transmitter, "predicted_nt_confidence": 1.0})

    for index, (nerve, side) in enumerate((n, s) for n in ("ProLN", "MesoLN", "MetaLN") for s in ("L", "R")):
        sensory, intrinsic, motor = 100 + index, 200 + index, 300 + index
        cell(sensory, "vnc_sensory", "acetylcholine", **{"class": "mechanosensory_proprioceptive", "entryNerve": nerve, "rootSide": side})
        cell(intrinsic, "vnc_intrinsic", "gaba" if index % 2 else "acetylcholine", somaNeuromere=("T1", "T2", "T3")[index // 2])
        cell(motor, "vnc_motor", "glutamate", exitNerve=nerve, somaSide=side)
        edges.extend([
            {"body_pre": sensory, "body_post": intrinsic, "weight": 10},
            {"body_pre": intrinsic, "body_post": motor, "weight": 8},
            {"body_pre": motor, "body_post": intrinsic, "weight": 100},  # terminal-source exclusion
            {"body_pre": intrinsic, "body_post": intrinsic, "weight": 100},  # self-loop exclusion
        ])
    cell(400, "descending_neuron", "gaba")
    for index in range(6):
        edges.append({"body_pre": 400, "body_post": 200 + index, "weight": 4})
    edges.append({"body_pre": 100, "body_post": 200, "weight": 2})  # aggregate duplicate raw evidence
    _write_feather(source / SOURCE_FILES["annotations"], annotations, ANNOTATION_COLUMNS)
    _write_feather(source / SOURCE_FILES["neurotransmitters"], nt, NT_COLUMNS)
    _write_feather(source / SOURCE_FILES["connectivity"], edges, ["body_pre", "body_post", "weight"])

    manifest = prepare(source, output, sensory_per_group=1, motor_per_group=1, descending_count=1, interneuron_count=6, status="synthetic")
    assert manifest["derived_counts"] == {
        "neurons": 19,
        "edges": 18,
        "excitatory_edges": 9,
        "inhibitory_edges": 9,
        "input_path_coverage": 7,
        "output_path_coverage": 6,
        "neurons_on_input_output_paths": 19,
    }
    assert manifest["neuron_model"]["threshold"] == 0.047
    assert manifest["neuron_model"]["surrogate_beta"] == 1.0
    circuit = load_connectome(output / "manifest.json", allow_synthetic=True)
    assert circuit.num_neurons == 19
    records = json.loads((output / "edges.json").read_text())
    assert next(edge for edge in records if edge["pre"] == "100" and edge["post"] == "200")["raw_synapse_count"] == 12
    assert all(not edge["pre"].startswith("3") and edge["pre"] != edge["post"] for edge in records)
    assert all(edge["weight"] < 0 for edge in records if edge["pre"] in {"201", "203", "205", "400"})
    audit = json.loads((output / "audit.json").read_text())
    assert audit["source_rows"]["connectivity"] == len(edges)
    assert audit["input_path_coverage"] == {"covered": 7, "total": 7}
    first_checksums = manifest["checksums"]
    rerun = prepare(source, output, sensory_per_group=1, motor_per_group=1, descending_count=1, interneuron_count=6, status="synthetic")
    assert rerun["checksums"] == first_checksums
    with pytest.raises(ValueError, match="lif_threshold"):
        prepare(source, output, lif_threshold=0)
    with pytest.raises(ValueError, match="surrogate_beta"):
        prepare(source, output, surrogate_beta=0)
