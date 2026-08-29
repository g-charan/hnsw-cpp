import numpy as np
import pytest

import hnsw_cpp


def brute_force(base, queries, k):
    # cosine on unit vectors, so inner product is the ranking
    return np.argsort(-(queries.astype(np.float64) @ base.T.astype(np.float64)), axis=1)[:, :k]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    base = rng.standard_normal((5000, 64), dtype=np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    queries = rng.standard_normal((100, 64), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return base, queries


def recall(ids, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(ids, truth)])


def test_recall_against_brute_force(data):
    base, queries = data
    idx = hnsw_cpp.Index(64, len(base), metric="ip")
    idx.add(base)
    assert len(idx) == len(base)
    ids, dists = idx.search_many(queries, k=10, ef=100)
    assert ids.shape == (100, 10)
    assert recall(ids, brute_force(base, queries, 10)) > 0.95
    # ip distance is 1 - dot, so it rises along the row
    assert np.all(np.diff(dists, axis=1) >= -1e-6)


def test_save_load_roundtrip(data, tmp_path):
    base, queries = data
    idx = hnsw_cpp.Index(64, len(base), metric="ip")
    idx.add(base)
    idx.save(str(tmp_path / "t.idx"))
    loaded = hnsw_cpp.Index.load(str(tmp_path / "t.idx"))
    assert len(loaded) == len(base) and loaded.dim == 64
    a, _ = idx.search_many(queries, k=5, ef=50)
    b, _ = loaded.search_many(queries, k=5, ef=50)
    assert np.array_equal(a, b)
    with pytest.raises(RuntimeError):
        loaded.add(base[:1])


def test_single_query_and_dim_check(data):
    base, queries = data
    idx = hnsw_cpp.Index(64, len(base), metric="ip")
    idx.add(base)
    ids, dists = idx.search(queries[0], k=3)
    assert ids.shape == (3,) and dists.shape == (3,)
    with pytest.raises(ValueError):
        idx.search(np.zeros(32, dtype=np.float32))
    with pytest.raises(ValueError):
        hnsw_cpp.Index(64, 10, metric="cosine")
