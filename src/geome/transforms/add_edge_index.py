import torch
from anndata import AnnData

from geome.transforms.utils import check_adj_matrix_loc
from geome.utils import get_from_loc, set_to_loc

from .transform import Transform


class AddEdgeIndex(Transform):
    """Add the edge_index to the AnnData object.

    Args:
    ----
    adj_matrix_loc (str): The location where the adjacency matrix is stored.
    edge_index_key (str): The key to store the edge_index in adata.uns.
    overwrite (bool, optional): Whether to overwrite the edge_index if it already exists. Defaults to False.
    """

    def __init__(self, adj_matrix_loc: str, edge_index_key: str, overwrite: bool = False):
        check_adj_matrix_loc(adj_matrix_loc)
        self.adj_matrix_loc = adj_matrix_loc
        self.edge_index_key = edge_index_key
        self.overwrite = overwrite

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
        adj_matrix = get_from_loc(adata, self.adj_matrix_loc)
        # Convert adjacency matrix to edge_index
        # TODO(syelman): There is probably a better way to do this
        nodes1, nodes2 = adj_matrix.nonzero()
        edge_index = torch.vstack(
            [
                torch.from_numpy(nodes1).to(torch.long),
                torch.from_numpy(nodes2).to(torch.long),
            ]
        )

        # np.column_stack(adj_matrix.nonzero()).astype(np.int64)
        # Store edge_index in adata.uns
        set_to_loc(adata, f"uns/{self.edge_index_key}", edge_index, self.overwrite)

        return adata
