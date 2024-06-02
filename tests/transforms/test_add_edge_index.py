import anndata as ad
import numpy as np
import scipy.sparse as sp
import squidpy as sq
import torch

from geome import transforms


def test_add_edge_index_from_adj():
    # Define the adjacency matrix
    adjacency_matrix = np.array([[0, 2, 0, 0], [2, 0, 3, 0], [0, 3, 0, 1], [0, 0, 1, 0]])

    # Create the edge index tensor
    edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight = np.array([2, 2, 3, 3, 1, 1])

    adata = ad.AnnData(
        X=np.random.rand(4, 4),
        obsp={"dense": adjacency_matrix, "sparse": sp.csr_matrix(adjacency_matrix)},
    )
    tf_dense = transforms.AddEdgeIndexFromAdj(
        adj_matrix_loc="obsp/dense",
        edge_index_key="edge_index_dense",
        edge_weight_key="edge_weight_dense",
    )
    tf_sparse = transforms.AddEdgeIndexFromAdj(
        adj_matrix_loc="obsp/sparse",
        edge_index_key="edge_index_sparse",
        edge_weight_key="edge_weight_sparse",
    )
    adata = tf_sparse(tf_dense(adata))
    # check if they are equal unsorted also the weights
    assert torch.equal(
        adata.uns["edge_index_dense"],
        torch.tensor(edge_index, dtype=torch.long),
    )
    assert torch.equal(
        adata.uns["edge_weight_dense"],
        torch.tensor(edge_weight, dtype=torch.float),
    )
    assert torch.equal(
        adata.uns["edge_index_sparse"],
        torch.tensor(edge_index, dtype=torch.long),
    )
    assert torch.equal(
        adata.uns["edge_weight_sparse"],
        torch.tensor(edge_weight, dtype=torch.float),
    )


def test_add_edge_index():
    # Define the coordinates as a numpy array
    coordinates = np.random.rand(50, 2)
    median_dist = np.median(np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1))

    # Create an AnnData object
    adata = ad.AnnData(np.random.rand(50, 2), obsm={"spatial_init": coordinates})
    sq.gr.spatial_neighbors(
        adata, coord_type="generic", radius=median_dist, spatial_key="spatial_init", key_added="gt", n_neighs=4
    )
    tf = transforms.AddEdgeIndex(
        spatial_key="spatial_init",
        key_added="pred",
        func_args={"radius": median_dist, "n_neighs": 4, "coord_type": "generic"},
        edge_index_key="edge_index",
        edge_weight_key="edge_weight",
        gets_connectivities=False, # gets distances
    )
    # Extract the adjacency matrix from the results
    adj_matrix = adata.obsp["gt_distances"]

    # Convert the adjacency matrix to edge indices
    nodes1, nodes2 = adj_matrix.nonzero()
    edge_index_gt = torch.tensor(np.array([nodes1, nodes2]), dtype=torch.long)
    edge_weight_gt = torch.tensor(adj_matrix[nodes1, nodes2], dtype=torch.double)

    adata = tf(adata)
    assert torch.equal(adata.uns["edge_index"], edge_index_gt)
    assert torch.allclose(adata.uns["edge_weight"], edge_weight_gt)
    assert np.allclose(adata.obsp["pred_distances"].A, adata.obsp["gt_distances"].A)
