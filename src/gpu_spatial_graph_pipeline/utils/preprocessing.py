import squidpy as sq
import torch
import pandas as pd
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data
from anndata import AnnData
import scipy
import torch.nn as nn


def design_matrix(A, Xl, Xc):
    N, L = Xl.shape
    Xs = (A @ Xl > 0).to(np.float)  # N x L
    Xts = np.einsum("bp,br->bpr", Xs, Xl).reshape((N, L * L))
    Xts = (Xts > 0).to(np.float)
    Xd = np.hstack((Xl, Xts, Xc))
    return Xd
