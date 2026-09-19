from pathlib import Path

import pytest

from g1_fly_control.connectome import ConnectomeValidationError, load_connectome
from g1_fly_control.connectome.rewire import degree_preserving_rewire


FIXTURE = Path(__file__).parents[1] / "fixtures" / "synthetic_circuit" / "manifest.json"


def test_synthetic_requires_explicit_opt_in():
    with pytest.raises(ConnectomeValidationError, match="Synthetic"):
        load_connectome(FIXTURE)
    circuit = load_connectome(FIXTURE, allow_synthetic=True)
    assert circuit.num_neurons == 4
    assert circuit.dense_weight_matrix()[1, 0] == 1.0


def test_degree_preserving_rewire_preserves_in_and_out_degree():
    circuit = load_connectome(FIXTURE, allow_synthetic=True)
    rewired, weights, report = degree_preserving_rewire(circuit.edge_index, circuit.weights, seed=4, swaps=3, allow_self_loops=False)
    assert report.completed_swaps >= 0
    assert weights.tolist() == circuit.weights.tolist()
    for row in (0, 1):
        assert (rewired[row].bincount(minlength=4) == circuit.edge_index[row].bincount(minlength=4)).all()

