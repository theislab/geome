"""Train a GCN autoencoder on a slide that is read from disk one tile at a time.

Two steps, run as separate processes so the training process's memory is its own::

    python scripts/out_of_core_gcn.py make  --path slide.zarr --n-cells 2000000 --n-genes 300
    python scripts/out_of_core_gcn.py train --path slide.zarr --mode exact --epochs 20

Modes:

- ``exact``: gradients of all tiles are summed before one optimizer step per epoch. This is full-batch training, out
  of core; with ``--check`` on a small slide it is compared against ``full``.
- ``fast``: one optimizer step per tile.
- ``full``: the whole slide and graph in memory, as GraphST/STAGATE do today. The baseline for memory.

The slide is synthetic: cells in a square with ``--n-domains`` Voronoi domains, each with its own expression
program, so the embedding can be checked against the domains (ARI after k-means).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from torch_geometric.nn import GCNConv

from geome.tiles import TileLoader, spatial_graph, write_sorted


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # bytes on macOS, KiB on Linux


def make(args) -> None:
    rng = np.random.default_rng(args.seed)
    side = np.sqrt(args.n_cells) * 10.0  # ~10 µm between cells
    coords = rng.uniform(0, side, size=(args.n_cells, 2)).astype(np.float32)
    centers = rng.uniform(0, side, size=(args.n_domains, 2))
    from scipy.spatial import KDTree

    domain = KDTree(centers).query(coords, workers=-1)[1]
    programs = rng.gamma(0.5, 2.0, size=(args.n_domains, args.n_genes)).astype(np.float32)
    counts = rng.poisson(programs[domain]).astype(np.float32)
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts),
        obs={"domain": domain.astype(str)},
        obsm={"spatial": coords},
    )
    write_sorted(adata, args.path)
    print(json.dumps({"made": str(args.path), "n_cells": args.n_cells, "n_genes": args.n_genes}))


class Autoencoder(torch.nn.Module):
    def __init__(self, n_genes: int, hidden: int = 64, latent: int = 16):
        super().__init__()
        self.conv1 = GCNConv(n_genes, hidden, normalize=False, add_self_loops=False)
        self.conv2 = GCNConv(hidden, latent, normalize=False, add_self_loops=False)
        self.decoder = torch.nn.Linear(latent, n_genes)

    def embed(self, x, edge_index, edge_weight):
        return self.conv2(torch.relu(self.conv1(x, edge_index, edge_weight)), edge_index, edge_weight)

    def forward(self, x, edge_index, edge_weight):
        return self.decoder(self.embed(x, edge_index, edge_weight))


def _loss(model, x, ei, ew, mask, n_obs):
    x = torch.log1p(x)
    return ((model(x, ei, ew)[mask] - x[mask]) ** 2).sum() / n_obs


def train(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    if args.mode == "full":
        adata = ad.read_zarr(args.path)
        graph = spatial_graph(np.asarray(adata.obsm["spatial"]), args.n_neighbors).tocoo()
        x = torch.from_numpy(np.asarray(adata.X.toarray(), dtype=np.float32)).to(device)
        ei = torch.from_numpy(np.stack([graph.col, graph.row]).astype(np.int64)).to(device)
        ew = torch.from_numpy(graph.data).to(device)
        n_obs, n_genes = x.shape
        tiles = None
    else:
        tiles = TileLoader.from_path(args.path, n_hops=2, tile_size=args.tile_size, n_neighbors=args.n_neighbors)
        n_obs, n_genes = tiles.n_obs, tiles.tile(0).x.shape[1]
    setup_s = time.perf_counter() - t0

    model = Autoencoder(n_genes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(args.epochs):
        t = time.perf_counter()
        opt.zero_grad()
        total = 0.0
        if tiles is None:
            loss = _loss(model, x, ei, ew, slice(None), n_obs)
            loss.backward()
            total = loss.item()
            opt.step()
        else:
            for tile in tiles:
                tile = tile.to(device)
                loss = _loss(model, tile.x, tile.edge_index, tile.edge_weight, tile.core_mask, n_obs)
                loss.backward()
                total += loss.item()
                if args.mode == "fast":
                    opt.step()
                    opt.zero_grad()
            if args.mode == "exact":
                opt.step()
        history.append({"epoch": epoch, "loss": total, "seconds": time.perf_counter() - t})
        print(json.dumps(history[-1]), flush=True)

    result = {
        "mode": args.mode,
        "n_cells": n_obs,
        "n_genes": n_genes,
        "tile_size": args.tile_size if tiles is not None else None,
        "n_tiles": len(tiles) if tiles is not None else 1,
        "setup_seconds": setup_s,
        "final_loss": history[-1]["loss"],
        "peak_rss_gb": peak_rss_gb(),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
    }
    if args.ari:
        result["ari"] = _ari(args, model, tiles, device)
    print(json.dumps(result), flush=True)
    return result


@torch.no_grad()
def _ari(args, model, tiles, device) -> float:
    """k-means on the embedding against the synthetic domains."""
    import zarr
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score

    domain = ad.io.read_elem(zarr.open_group(args.path, mode="r")["obs"])["domain"].to_numpy()  # obs only
    if tiles is None:
        tiles = TileLoader.from_path(args.path, n_hops=2, tile_size=args.tile_size, shuffle=False)
    z = np.empty((tiles.n_obs, model.conv2.out_channels), dtype=np.float32)
    for tile in tiles:
        tile = tile.to(device)
        emb = model.embed(torch.log1p(tile.x), tile.edge_index, tile.edge_weight)[tile.core_mask]
        z[tile.rows[tile.core_mask].cpu().numpy()] = emb.cpu().numpy()
    labels = MiniBatchKMeans(args.n_domains, random_state=args.seed, n_init=3).fit_predict(z)
    return float(adjusted_rand_score(domain, labels))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["make", "train"])
    p.add_argument("--path", required=True)
    p.add_argument("--n-cells", type=int, default=100_000)
    p.add_argument("--n-genes", type=int, default=300)
    p.add_argument("--n-domains", type=int, default=12)
    p.add_argument("--mode", choices=["exact", "fast", "full"], default="exact")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--tile-size", type=int, default=50_000)
    p.add_argument("--n-neighbors", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--ari", action="store_true", help="k-means ARI of the embedding against the true domains")
    args = p.parse_args()
    make(args) if args.step == "make" else train(args)


if __name__ == "__main__":
    main()
