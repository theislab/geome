from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree

__all__ = ["expand", "morton_order", "spatial_graph"]


def spatial_graph(coords: np.ndarray, n_neighbors: int = 6) -> sparse.csr_array:
    """Symmetric kNN graph with self-loops, normalized as in a GCN: :math:`D^{-1/2} (A + I) D^{-1/2}`.

    The weights are computed once on the whole slide, so a tile's subgraph keeps its cells' global degrees and a
    GCN layer with ``normalize=False`` gives a tile's cells exactly their full-graph output.
    """
    n = coords.shape[0]
    _, idx = KDTree(coords).query(coords, k=n_neighbors + 1, workers=-1)
    rows = np.repeat(np.arange(n), n_neighbors)
    a = sparse.csr_array((np.ones(rows.size, dtype=np.float32), (rows, idx[:, 1:].ravel())), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32) + sparse.eye_array(n, dtype=np.float32, format="csr")
    d = 1.0 / np.sqrt(np.asarray(a.sum(axis=1)).ravel())
    return sparse.csr_array(sparse.diags_array(d) @ a @ sparse.diags_array(d), dtype=np.float32)


def _spread(v: np.ndarray) -> np.ndarray:
    """Interleave zero bits into the low 32 bits of ``v`` (one half of a 2-D Morton code)."""
    v = v & np.uint64(0x00000000FFFFFFFF)
    for shift, mask in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF), (4, 0x0F0F0F0F0F0F0F0F),
                        (2, 0x3333333333333333), (1, 0x5555555555555555)):  # fmt: skip
        v = (v | (v << np.uint64(shift))) & np.uint64(mask)
    return v


def morton_order(coords: np.ndarray) -> np.ndarray:
    """Permutation sorting cells along a Z-order curve, so cells close in space are close in row order."""
    q = coords[:, :2] - coords[:, :2].min(axis=0)
    q = (q / max(float(q.max()), 1e-12) * (2**16 - 1)).astype(np.uint64)
    return np.argsort(_spread(q[:, 0]) | (_spread(q[:, 1]) << np.uint64(1)), kind="stable")


def expand(graph: sparse.csr_array, core: np.ndarray, n_hops: int) -> np.ndarray:
    """Sorted rows of ``core`` plus every cell within ``n_hops`` edges of it."""
    seen = np.zeros(graph.shape[0], dtype=bool)
    seen[core] = True
    frontier = core
    for _ in range(n_hops):
        nbrs = np.unique(graph[frontier].indices)
        frontier = nbrs[~seen[nbrs]]
        seen[frontier] = True
    return np.flatnonzero(seen)
