import numpy as np


def design_matrix(A, Xl, Xc):
    N, L = Xl.shape
    Xs = (A @ Xl > 0).to(np.float)  # N x L
    Xts = np.einsum("bp,br->bpr", Xs, Xl).reshape((N, L * L))
    Xts = (Xts > 0).to(np.float)
    Xd = np.hstack((Xl, Xts, Xc))
    return Xd


def add_design_matrix(adata, input_names, output_name):
    """Adds design matrix to the given adata object to the given field.

    A: Adj. matrix
    Xl: Cell types
    Xc: Domain

    Args:
        adata (_type_): _description_
        input_names (_type_): Dictionary of field names where A, Xl and Xc are
            in adata.
        output_name (_type_): Where to store the matrix in adata.
    """
    pass
