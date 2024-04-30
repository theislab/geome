import numpy as np
import torch
import anndata as ad
from geome import transforms
import scipy.sparse as sp

def test_add_edge_index():
    # Define the adjacency matrix
    adjacency_matrix = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )

    # Find the non-zero entries in the adjacency matrix (i.e., edges)
    # Since it's an undirected graph, we need to consider each edge in both directions.
    nodes1, nodes2 = adjacency_matrix.nonzero()

    # Create the edge index tensor
    edge_index = torch.tensor(np.array([nodes1, nodes2]), dtype=torch.long)

    # Print the adjacency matrix and the edge index tensor
    print("Adjacency Matrix:")
    print(adjacency_matrix)
    print("\nEdge Index Tensor:")
    print(edge_index)

    adata = ad.AnnData(
        X=np.random.rand(4, 4),
        obsp={"dense": adjacency_matrix, "sparse": sp.csr_matrix(adjacency_matrix)},
    )
    tf = transforms.AddEdgeIndex(
        adj_matrix_loc="obsp/dense",
        edge_index_key="edge_index",
        edge_weight_key="edge_weight",
    )
    adata = tf(adata)
    # check if they are equal unsorted
    edge_index_set1 = set(map(tuple, edge_index.T.numpy()))
    edge_index_set2 = set(map(tuple, adata.uns["edge_index"].T.numpy()))
    assert edge_index_set1 == edge_index_set2
