import scipy.sparse as sparse
import torch
from anndata import AnnData

from geome.utils import get_from_loc, set_to_loc

from .transform import Transform


class AddEdgeWeight(Transform):
    """Add the edge weights to the AnnData object.

    Args:
    ----
    edge_index_key (str): The key where the edge_index is stored in adata.uns.
    weight_matrix_loc (str): The location where the weight matrix is stored.
    edge_weight_key (str): The key to store the computed edge weights in adata.uns.
    overwrite (bool): Whether to overwrite the edge weights if they already exist.
    """

    def __init__(
        self,
        edge_index_key: str,
        weight_matrix_loc: str,
        edge_weight_key: str,
        overwrite: bool = False,
    ):
        self.edge_index_key = edge_index_key
        self.weight_matrix_loc = weight_matrix_loc
        self.edge_weight_key = edge_weight_key
        self.overwrite = overwrite

    def __call__(self, adata: AnnData) -> AnnData:
        """Will compute and add the edge weights to adata.uns.

        Args:
        ----
        adata: The AnnData object.
        """
        weight_matrix = get_from_loc(adata, self.weight_matrix_loc)
        edge_index = get_from_loc(adata, f"uns/{self.edge_index_key}")
        edge_weights = torch.empty(0)
        if edge_index.shape[1] != 0:
            if sparse.issparse(weight_matrix):
                # Convert to CSR format
                csr_matrix = weight_matrix.tocsr()
                edge_index_np = edge_index.numpy()
                # Get the values
                values = csr_matrix[edge_index_np[0], edge_index_np[1]].A1
                # Convert the numpy array to a torch tensor
                edge_weights = torch.tensor(values)
            else:
                edge_weights = weight_matrix[edge_index[0, :], edge_index[1, :]]
        set_to_loc(adata, f"uns/{self.edge_weight_key}", edge_weights, self.overwrite)

        return adata
