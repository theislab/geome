import numpy as np
import torch
import anndata as ad
from geome import transforms
import scipy.sparse as sp


def test_add_edge_index():
    # Define the adjacency matrix
    adjacency_matrix = np.array(
        [[0, 2, 0, 0], [2, 0, 3, 0], [0, 3, 0, 1], [0, 0, 1, 0]]
    )

    # Create the edge index tensor
    edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_weight = np.array([2, 2, 3, 3, 1, 1])

    adata = ad.AnnData(
        X=np.random.rand(4, 4),
        obsp={"dense": adjacency_matrix, "sparse": sp.csr_matrix(adjacency_matrix)},
    )
    tf_dense = transforms.AddEdgeIndex(
        adj_matrix_loc="obsp/dense",
        edge_index_key="edge_index_dense",
        edge_weight_key="edge_weight_dense",
    )
    tf_sparse = transforms.AddEdgeIndex(
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
