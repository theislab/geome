
import numpy as np
import pandas as pd
from anndata import AnnData

from geome.transforms.utils import check_adj_matrix_loc
from geome.utils import check_loc, get_from_loc, set_to_loc

from .transform import Transform


def design_matrix(A: np.ndarray, Xl: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Returns the design matrix given the adjacency matrix, cell types, and domains.

    Args:
    ----
    A (np.ndarray): The adjacency matrix.
    Xl (np.ndarray): The cell types.
    Xc (np.ndarray): The domains.

    Returns:
    -------
    np.ndarray: The design matrix.
    """
    N, L = Xl.shape
    Xs = (A @ Xl > 0).astype(np.float64)  # N x L
    Xts = np.einsum("bp,br->bpr", Xs, Xl).reshape((N, L * L))
    Xts = (Xts > 0).astype(np.float64)
    Xd = np.hstack((Xl, Xts, Xc))
    return Xd


class AddDesignMatrix(Transform):
    """Adds the design matrix defined in NCEM paper to adata.obsm."""

    def __init__(self, xl_loc: str, xc_loc: str, adj_matrix_loc: str, output_name: str, overwrite: bool = False):
        check_loc(xl_loc)
        self.xl_loc = xl_loc
        check_loc(xc_loc)
        self.xc_loc = xc_loc
        check_adj_matrix_loc(adj_matrix_loc)
        self.adj_matrix_loc = adj_matrix_loc
        self.output_name = output_name
        self.overwrite = overwrite

    def __call__(self, adata: AnnData) -> AnnData:
        """Adds the design matrix to the given AnnData object in the specified field.

        Args:
        ----
        adata: The AnnData object.
        xl_loc (str): The name of the field containing the cell types.
        xc_loc (str): The name of the field containing the domains.
        output_name (str): The name of the field to store the design matrix in.
        """
        dm = design_matrix(
            get_from_loc(adata, self.adj_matrix_loc),
            pd.get_dummies(get_from_loc(adata, self.xl_loc)).to_numpy(),
            pd.get_dummies(get_from_loc(adata, self.xc_loc)).to_numpy(),
        )
        set_to_loc(adata, f"obsm/{self.output_name}", dm, self.overwrite)
        return adata
