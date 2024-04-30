import torch
from anndata import AnnData

from geome.transforms.utils import check_adj_matrix_loc
from geome.utils import get_from_loc, set_to_loc

from torch_geometric.utils import from_scipy_sparse_matrix

import scipy.sparse as sp

from .base.transform import Transform

from dataclasses import dataclass, field


@dataclass
class AddEdgeIndex(Transform):
    """Add the edge_index to the AnnData object.

    Args:
    ----
    adj_matrix_loc (str): The location where the adjacency matrix is stored.
    edge_index_key (str): The key to store the edge_index in adata.uns.
    edge_weight_key (str, optional): The key to store the edge weights in adata.uns. Defaults to None.
    """

    adj_matrix_loc: str
    edge_index_key: str
    edge_weight_key: str | None = None

    def __call__(self, adata: AnnData) -> AnnData:
        """Add the edge_index derived from the adjacency matrix to adata.uns.

        Args:
        ----
        adata (AnnData): The AnnData object.

        Returns
        -------
        AnnData: The updated AnnData object with the added edge_index.
        """
        # Extract adjacency matrix
        check_adj_matrix_loc(self.adj_matrix_loc)
        adj_matrix = get_from_loc(adata, self.adj_matrix_loc)
        if sp.issparse(adj_matrix):
            edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)
        else:

            # Convert adjacency matrix to edge_index
            # TODO(syelman): There is probably a better way to do this
            nodes1, nodes2 = adj_matrix.nonzero()
            edge_index = torch.vstack(
                [
                    torch.from_numpy(nodes1).to(torch.long),
                    torch.from_numpy(nodes2).to(torch.long),
                ]
            )
            edge_weight = adj_matrix[nodes1, nodes2]

        # Store edge_weight in adata.uns
        set_to_loc(adata, f"uns/{self.edge_index_key}", edge_index, True)
        if self.edge_weight_key is not None:
            set_to_loc(adata, f"uns/{self.edge_weight_key}", edge_weight, True)

        return adata
