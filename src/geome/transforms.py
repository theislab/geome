import numpy as np
import squidpy as sq
import pandas as pd
from typing import Any, List


def get_adjacency_from_adata(adata: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    """Returns the spatial connectivities matrix from an AnnData object.

    Args:
        adata: The AnnData object.

    Returns:
        np.ndarray: The spatial connectivities matrix.
    """
    if "adjacency_matrix_connectivities" in adata.obsp.keys():
        spatial_connectivities = adata.obsp["adjacency_matrix_connectivities"]
    else:
        spatial_connectivities, _ = sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            key_added="spatial",
            copy=True,
        )
    return spatial_connectivities


def design_matrix(A: np.ndarray, Xl: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Returns the design matrix given the adjacency matrix, cell types, and domains.

    Args:
        A (np.ndarray): The adjacency matrix.
        Xl (np.ndarray): The cell types.
        Xc (np.ndarray): The domains.

    Returns:
        np.ndarray: The design matrix.
    """
    N, L = Xl.shape
    Xs = (A @ Xl > 0).astype(np.float64)  # N x L
    Xts = np.einsum("bp,br->bpr", Xs, Xl).reshape((N, L * L))
    Xts = (Xts > 0).astype(np.float64)
    Xd = np.hstack((Xl, Xts, Xc))
    return Xd


def add_design_matrix(adata: Any, xl_name: str, xc_name: str, output_name: str) -> None:
    """Adds the design matrix to the given AnnData object in the specified field.

    Args:
        adata: The AnnData object.
        xl_name (str): The name of the field containing the cell types.
        xc_name (str): The name of the field containing the domains.
        output_name (str): The name of the field to store the design matrix in.
    """
    adata.obsm[output_name] = design_matrix(
        get_adjacency_from_adata(adata),
        pd.get_dummies(get_from_address(adata, xl_name)).to_numpy(),
        pd.get_dummies(get_from_address(adata, xc_name)).to_numpy(),
    )


def get_from_address(adata: Any, address: str) -> Any:
    """Gets the object at the given address from the AnnData object.

    Args:
        adata: The AnnData object.
        address (str): The address of the object.

    Returns:
        Any: The object at the given address.
    """
    # TODO check if address exists
    # TODO change function location
    attrs = address.split("/")
    assert len(attrs) <= 2, "assumes at most one delimiter"

    obj = adata
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)  # obj.attr
        else:
            obj = obj[attr]
    return obj


def categorize_obs(adata: Any, obs_list: List[str]) -> None:
    """Converts the given list of observation columns in the AnnData object to categorical.

    Args:
        adata: The AnnData object.
        obs_list (list[str]): The list of observation columns to convert to categorical.
    """
    for cat in obs_list:
        adata.obs[cat] = adata.obs[cat].astype("category")
