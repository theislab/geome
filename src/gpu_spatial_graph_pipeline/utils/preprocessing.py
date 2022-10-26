import numpy as np
import squidpy as sq
from scipy import sparse

def get_address(adata, address):
    #TODO check if address exists

    obj = adata
    for attr in address.split("/"):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)  # obj.attr
        else:
            obj = obj[attr]
    if sparse.issparse(obj):
        obj = np.array(obj.todense())
    return obj

def get_adjacency(adata, *args, **kwargs):
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
    Xs = (A @ Xl > 0).to(np.float)  # N x L
    Xts = np.einsum("bp,br->bpr", Xs, Xl).reshape((N, L * L))
    Xts = (Xts > 0).to(np.float)
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
    adata.obsm[output_name] = design_matrix(get_adjacency(adata), get_address(adata, xl_name), get_address(adata, xc_name))
    return