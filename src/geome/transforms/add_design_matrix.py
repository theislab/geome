
import numpy as np
import pandas as pd
from anndata import AnnData

from geome.utils import get_adjacency_from_adata, get_from_address


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



class AddDesignMatrix:
    def __init__(self, xl_name: str, xc_name: str, output_name: str):
        self.xl_name = xl_name
        self.xc_name = xc_name
        self.output_name = output_name

    def __call__(self, adata: AnnData) -> AnnData:
        """Store processed data in `obsm` field of `adata`.

        Store the new address for each processed data in `uns` field of `adata`.

        Parameters
        ----------
        adata : AnnData
            AnnData object.
        fields : Dict[str, List[str]]
            Dictionary of fields and their associated attributes.

        Returns
        -------
        Processed adata
        """
        """Adds the design matrix to the given AnnData object in the specified field.

        Args:
        ----
        adata: The AnnData object.
        xl_name (str): The name of the field containing the cell types.
        xc_name (str): The name of the field containing the domains.
        output_name (str): The name of the field to store the design matrix in.
        """
        adata.obsm[self.output_name] = design_matrix(
            get_adjacency_from_adata(adata),
            pd.get_dummies(get_from_address(adata, self.xl_name)).to_numpy(),
            pd.get_dummies(get_from_address(adata, self.xc_name)).to_numpy(),
        )
        return adata

