from __future__ import annotations

from dataclasses import dataclass

import scipy.sparse as sp
import torch
from anndata import AnnData
from torch_geometric.utils import from_scipy_sparse_matrix

from geome.transforms.utils import check_adj_matrix_loc
from geome.utils import get_from_loc, set_to_loc

from .add_adj_matrix import AddAdjMatrix
from .base.transform import Transform
from .compose import Compose


@dataclass
class AddEdgeIndexFromAdj(Transform):
    """Add the edge_index to the AnnData object given an adjacency matrix.

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
            edge_weight = torch.from_numpy(adj_matrix[nodes1, nodes2]).to(torch.float)

        # Store edge_weight in adata.uns
        set_to_loc(adata, f"uns/{self.edge_index_key}", edge_index, True)
        if self.edge_weight_key is not None:
            set_to_loc(adata, f"uns/{self.edge_weight_key}", edge_weight, True)

        return adata

@dataclass
class AddEdgeIndex(Transform):
    """Add the edge_index to the AnnData object given spatial coordinates.

    Internally composes the AddAdjMatrix and AddEdgeIndexFromAdj transforms.
    The AddAdjMatrix, calls the `spatial_neighbors` function from `squidpy` to calculate the spatial connectivities matrix.

    Args:
    ----
    spatial_key (str): The location where the spatial coordinates are stored.
    key_added (str): The key to add the adjacency matrix to.
    func_args (dict): Additional arguments to pass to the `spatial_neighbors` function.
    edge_index_key (str): The key to store the edge_index in adata.uns.
    gets_connectivities (bool): Whether to get the connectivities matrix otherwise will get distance matrix. Defaults to True.
    """

    spatial_key: str
    key_added: str
    func_args: dict
    edge_index_key: str
    edge_weight_key: str | None = None
    gets_connectivities: bool = True

    def __post_init__(self):
        postfix = "connectivities" if self.gets_connectivities else "distances"
        self._transform = Compose(
            [
                AddAdjMatrix(
                    spatial_key=self.spatial_key,
                    key_added=self.key_added,
                    func_args=self.func_args,
                ),
                AddEdgeIndexFromAdj(
                    adj_matrix_loc=f"obsp/{self.key_added}_{postfix}",
                    edge_index_key=self.edge_index_key,
                    edge_weight_key=self.edge_weight_key,
                ),
            ]
        )

    def __call__(self, adata: AnnData) -> AnnData:  # noqa: D102
        return self._transform(adata)
