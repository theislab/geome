import anndata as ad
import numpy as np
import torch

from geome import ann2data, iterables, transforms


def test_sample_case_ann2data_basic():
    coordinates = np.random.rand(50, 2)
    # make sure that there are two clusters of spatial coordinates
    # so that the resulting splits number of edges will be the same
    # as the sum of the number of edges in each cluster
    func_args = {"radius": 4.0, "coord_type": "generic"}
    cell_type = ["a"] * 20 + ["b"] * 20 + ["c"] * 5 + ["d"] * 5
    image_id = list("xy" * 20) + ["z"] * 10
    # make clusters for each cell type
    for i, ct in enumerate(set(cell_type)):
        idx = np.where(np.array(cell_type) == ct)[0]
        coordinates[idx, 0] += 100 * i
        coordinates[idx, 1] += 100 * i

    adata_gt = ad.AnnData(
        np.random.rand(50, 2),
        obs={"cell_type": cell_type, "image_id": image_id},
        obsm={"spatial_init": coordinates},
    )
    a2d = ann2data.Ann2DataByCategory(
        fields={
            "x": ["X"],
            "obs_names": ["obs_names"],
            "var_names": ["var_names"],
            "edge_index": ["uns/edge_index"],
            "edge_weight": ["uns/edge_weight"],
            "y": ["obs/cell_type"],
        },
        category="cell_type",
        preprocess=transforms.Categorize(keys=["cell_type", "image_id"]),
        transform=transforms.AddEdgeIndex(
            spatial_key="spatial_init",
            key_added="graph",
            edge_index_key="edge_index",
            edge_weight_key="edge_weight",
            func_args=func_args,
        ),
    )
    datas = list(a2d(adata_gt.copy()))
    assert len(datas) == 4
    big_adata_tf = transforms.Compose(
        [
            transforms.Categorize(keys=["cell_type", "image_id"]),
            transforms.AddEdgeIndex(
                spatial_key="spatial_init",
                key_added="graph",
                edge_index_key="edge_index",
                edge_weight_key="edge_weight",
                func_args=func_args,
            ),
        ]
    )
    big_adata = big_adata_tf(adata_gt.copy())
    # check if the concatenation of the two datasets is the same as the big dataset
    assert torch.allclose(torch.cat([d.x for d in datas]), torch.from_numpy(big_adata.X).to(torch.float))
    assert sum([d.edge_index.shape[1] for d in datas]) == big_adata.uns["edge_index"].shape[1]
    adatas = list(iterables.ToCategoryIterator(category="cell_type")(big_adata))
    assert len(adatas) == 4
    # this line is the for loop version of the last two assertions

    for a in adatas:
        ct = a.obs["cell_type"].values[0]
        ct_idx = np.where(np.array(cell_type) == ct)[0]
        np.allclose(
            a.obsp["graph_distances"].todense(),
            big_adata.obsp["graph_distances"][ct_idx, :][:, ct_idx].todense(),
        )
