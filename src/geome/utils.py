from typing import Any

import numpy as np
import squidpy as sq
from anndata import AnnData


def get_from_address(adata: AnnData, address: str) -> Any:
    """Gets the object at the given address from the AnnData object.

    Args:
    ----
    adata: The AnnData object.
    address (str): The address of the object.

    Returns:
    -------
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


def get_adjacency_from_adata(adata: Any, *args: Any, **kwargs: Any) -> np.ndarray:
    """Returns the spatial connectivities matrix from an AnnData object.

    Args:
    ----
    adata: The AnnData object.

    Returns:
    -------
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


