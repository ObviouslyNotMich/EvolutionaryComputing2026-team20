"""Genotype -> fixed-width vector, used by dashboard diversity/novelty."""

from ariel.visualisation.dashboard.dashboard_new import _geno_vec

GRAPH = {
    "nodes": {
        "0": {"type": "CORE", "rotation": "DEG_0"},
        "1": {"type": "BRICK", "rotation": "DEG_90"},
    },
    "edges": [{"parent": 0, "child": 1, "face": "FRONT"}],
}


def test_flat_genotype_passes_through() -> None:
    assert _geno_vec([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]


def test_graph_genotype_counts() -> None:
    v = _geno_vec(GRAPH)
    assert v[0] == 2  # nodes
    assert v[1] == 1  # edges
    assert v.sum() > 0


def test_fixed_width_across_sizes() -> None:
    big = {
        "nodes": {str(i): {"type": "HINGE", "rotation": "DEG_45"} for i in range(9)},
        "edges": [{"parent": 0, "child": i, "face": "TOP"} for i in range(1, 9)],
    }
    assert len(_geno_vec(GRAPH)) == len(_geno_vec(big))


def test_different_shapes_differ() -> None:
    assert _geno_vec(GRAPH).tolist() != _geno_vec({"nodes": {}, "edges": []}).tolist()
