from __future__ import annotations

import os
from collections.abc import Iterator

import anndata as ad
import numpy as np
import torch
from anndata.acc import A, LayerAcc, MultiAcc
from torch_geometric.data import Data

from ._graph import expand, morton_order, spatial_graph

__all__ = ["TileLoader", "write_sorted"]

Rep = LayerAcc | MultiAcc  # a whole array: A.X, A.layers[k] or A.obsm[k]


def _read_rows(x, rows: np.ndarray) -> np.ndarray:
    """Dense float32 rows ``rows`` (ascending) of ``x``, reading each run of consecutive rows as one slice.

    Works the same for a backed zarr array or ``CSRDataset`` and for in-memory arrays. On a spatially sorted store a
    tile is one long run and its neighbour margin a few short ones.
    """
    runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
    parts = []
    for run in runs:
        sub = x[int(run[0]) : int(run[-1]) + 1]
        sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
        parts.append(sub.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def write_sorted(adata: ad.AnnData, path: str | os.PathLike, *, spatial: Rep = A.obsm["spatial"]) -> np.ndarray:
    """Write ``adata`` to zarr with its cells in Morton order of ``spatial``; returns the order used.

    A :class:`TileLoader` over the written store reads each tile as one contiguous slice.
    """
    order = morton_order(np.asarray(spatial.get(adata)))
    adata[order].write_zarr(path)
    return order


class TileLoader:
    """Spatial tiles of a slide as PyG :class:`~torch_geometric.data.Data`, read out of core.

    Cells are cut into tiles of ``tile_size`` consecutive cells along a Z-order curve. Each tile comes with every cell
    within ``n_hops`` edges of it (its margin) and the edges among them, weighted by :func:`spatial_graph` on the whole
    slide. For a GCN with at most ``n_hops`` layers and ``normalize=False``, the outputs on a tile's own cells
    (``core_mask``) equal the full-graph outputs, so summing a per-cell loss over all tiles before an optimizer step is
    full-batch training.

    Only the coordinates and the graph are held in memory; ``rep`` is read per tile.

    Parameters
    ----------
    adata
        The slide, in memory or backed (see :meth:`from_path`).
    rep
        Features per cell, as an :mod:`anndata.acc` reference: ``A.X``, ``A.layers[k]`` or ``A.obsm[k]``.
    spatial
        Coordinates, as an :mod:`anndata.acc` reference.
    n_neighbors
        Neighbours per cell in the kNN graph, before symmetrizing.
    n_hops
        Depth of the margin. Set it to the number of message-passing layers for exact outputs.
    tile_size
        Cells per tile, before the margin.
    shuffle
        Visit tiles in a new random order each epoch.
    seed
        Seed for the tile order.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        *,
        rep: Rep = A.X,
        spatial: Rep = A.obsm["spatial"],
        n_neighbors: int = 6,
        n_hops: int = 2,
        tile_size: int = 50_000,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        coords = np.asarray(spatial.get(adata)[:])
        self.graph = spatial_graph(coords, n_neighbors)
        order = morton_order(coords)
        self.tiles = [np.sort(order[i : i + tile_size]) for i in range(0, coords.shape[0], tile_size)]
        self.n_hops = n_hops
        self.n_obs = coords.shape[0]
        self._x = rep.get(adata)
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_path(
        cls, path: str | os.PathLike, *, rep: Rep = A.X, spatial: Rep = A.obsm["spatial"], **kwargs
    ) -> TileLoader:
        """Open a zarr store backed with :func:`scfit.data.open_adata`, reading only ``rep`` and ``spatial``."""
        from scfit.data import open_adata

        return cls(open_adata(path, rep, spatial), rep=rep, spatial=spatial, **kwargs)

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self) -> Iterator[Data]:
        order = self._rng.permutation(len(self.tiles)) if self._shuffle else range(len(self.tiles))
        for i in order:
            yield self.tile(i)

    def tile(self, i: int) -> Data:
        """Tile ``i`` with its margin: ``x``, ``edge_index``, ``edge_weight``, ``core_mask`` and global ``rows``."""
        core = self.tiles[i]
        rows = expand(self.graph, core, self.n_hops)
        sub = self.graph[rows][:, rows].tocoo()
        return Data(
            x=torch.from_numpy(_read_rows(self._x, rows)),
            # PyG edges point source -> target; the graph is symmetric, so col -> row is the same set.
            edge_index=torch.from_numpy(np.stack([sub.col, sub.row]).astype(np.int64)),
            edge_weight=torch.from_numpy(sub.data),
            core_mask=torch.from_numpy(np.isin(rows, core)),
            rows=torch.from_numpy(rows),
        )
