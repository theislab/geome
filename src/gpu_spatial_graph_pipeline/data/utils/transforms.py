import numpy as np
import squidpy as sq
import pandas as pd


def get_adjacency_from_adata(adata, *args, **kwargs):
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


def design_matrix(A, Xl, Xc):
    N, L = Xl.shape
    Xs = (A @ Xl > 0).astype(np.float64)  # N x L
    Xts = np.einsum("bp,br->bpr", Xs, Xl).reshape((N, L * L))
    Xts = (Xts > 0).astype(np.float64)
    Xd = np.hstack((Xl, Xts, Xc))
    return Xd


def add_design_matrix(adata, xl_name, xc_name, output_name):
    """Adds design matrix to the given adata object to the given field.

    A: Adj. matrix
    Xl: Cell types
    Xc: Domain

    Args:
        adata (_type_): _description_
        xl_names (_type_): field name Xl is in adata.
        xc_name (_type_): field name Xc is in adata.
        output_name (_type_): Where to store the matrix in adata.
    """
    adata.obsm[output_name] = design_matrix(
        get_adjacency_from_adata(adata),
        pd.get_dummies(get_from_address(adata, xl_name)).to_numpy(),
        pd.get_dummies(get_from_address(adata, xc_name)).to_numpy(),
    )


def get_from_address(adata, address):
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
