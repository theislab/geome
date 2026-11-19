import anndata as ad
import numpy as np
import pytest
import torch
from scipy import sparse
from torch_geometric.nn import GCNConv

pytest.importorskip("anndata.acc")
from geome.tiles import TileLoader, morton_order, spatial_graph, write_sorted


def _slide(n=1500, g=8, seed=0, sparse_x=False):
    rng = np.random.default_rng(seed)
    x = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    return ad.AnnData(
        X=sparse.csr_matrix(x) if sparse_x else x,
        obsm={"spatial": rng.uniform(0, 100, size=(n, 2))},
    )


class _GCN(torch.nn.Module):
    def __init__(self, g, h=16, n_layers=2):
        super().__init__()
        dims = [g] + [h] * n_layers
        self.convs = torch.nn.ModuleList(
            GCNConv(a, b, normalize=False, add_self_loops=False) for a, b in zip(dims, dims[1:], strict=False)
        )

    def forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x


def _full(adata, graph):
    coo = graph.tocoo()
    x = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    return (
        torch.from_numpy(np.asarray(x, dtype=np.float32)),
        torch.from_numpy(np.stack([coo.col, coo.row]).astype(np.int64)),
        torch.from_numpy(coo.data),
    )


def test_graph_symmetric_normalized():
    w = spatial_graph(np.random.default_rng(0).uniform(size=(200, 2)), n_neighbors=5)
    assert abs(w - w.T).max() < 1e-6
    assert (w.diagonal() > 0).all()


def test_morton_order_is_permutation():
    order = morton_order(np.random.default_rng(1).uniform(size=(500, 2)))
    np.testing.assert_array_equal(np.sort(order), np.arange(500))


@pytest.mark.parametrize("sparse_x", [False, True])
def test_tiles_cover_every_cell_once(sparse_x):
    loader = TileLoader(_slide(sparse_x=sparse_x), tile_size=300, shuffle=False)
    core = np.concatenate([t.rows[t.core_mask].numpy() for t in loader])
    np.testing.assert_array_equal(np.sort(core), np.arange(loader.n_obs))


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_tile_outputs_equal_full_graph(n_layers):
    adata = _slide()
    loader = TileLoader(adata, n_hops=n_layers, tile_size=300, shuffle=False)
    torch.manual_seed(0)
    model = _GCN(adata.n_vars, n_layers=n_layers)
    full = model(*_full(adata, loader.graph))
    for t in loader:
        out = model(t.x, t.edge_index, t.edge_weight)[t.core_mask]
        torch.testing.assert_close(out, full[t.rows[t.core_mask]], atol=1e-5, rtol=1e-5)


def test_margin_too_thin_differs():
    adata = _slide()
    loader = TileLoader(adata, n_hops=1, tile_size=300, shuffle=False)
    torch.manual_seed(0)
    model = _GCN(adata.n_vars, n_layers=2)
    full = model(*_full(adata, loader.graph))
    t = loader.tile(0)
    out = model(t.x, t.edge_index, t.edge_weight)[t.core_mask]
    assert not torch.allclose(out, full[t.rows[t.core_mask]], atol=1e-5)


def test_accumulated_gradient_equals_full_batch():
    adata = _slide()
    loader = TileLoader(adata, n_hops=2, tile_size=300, shuffle=False)
    torch.manual_seed(0)
    model = _GCN(adata.n_vars, h=adata.n_vars)
    x, ei, ew = _full(adata, loader.graph)

    ((model(x, ei, ew) - x) ** 2).sum().div(loader.n_obs).backward()
    full_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    for t in loader:
        err = model(t.x, t.edge_index, t.edge_weight)[t.core_mask] - t.x[t.core_mask]
        (err**2).sum().div(loader.n_obs).backward()
    for g_full, p in zip(full_grads, model.parameters(), strict=True):
        torch.testing.assert_close(p.grad, g_full, atol=1e-5, rtol=1e-4)


def test_backed_store_matches_in_memory(tmp_path):
    pytest.importorskip("scfit")
    adata = _slide(sparse_x=True)
    path = tmp_path / "slide.zarr"
    order = write_sorted(adata, path)
    on_disk = TileLoader.from_path(path, tile_size=300, shuffle=False)
    in_mem = TileLoader(adata[order].copy(), tile_size=300, shuffle=False)
    for a, b in zip(on_disk, in_mem, strict=True):
        torch.testing.assert_close(a.x, b.x)
        torch.testing.assert_close(a.edge_index, b.edge_index)
