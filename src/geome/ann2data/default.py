from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse

from geome.utils import get_from_loc

from .ann2data import Ann2Data


class Ann2DataDefault(Ann2Data):
    """Convert anndata object into a dictionary of torch.tensors then create pyg.Data from them."""

    def __init__(
        self,
        fields: dict[str, list[str]],
        adata2iter: Callable[[AnnData], AnnData] | None = None,
        preprocess: list[Callable[[AnnData], AnnData]] | None = None,
        transform: list[Callable[[AnnData], AnnData]] | None = None,
    ) -> None:
        """Convert anndata object into a dictionary of arrays.

        Assumes adata.obsp["adjacency_matrix_connectivities"] exists
        if not it is computed with sq.gr.spatial_neighbors.
        Works for squidpy datasets and the datasets of this package.

        Example for fields argument:
            fields = {
                'features': ['obs/Cluster', 'obs/donor'],
                'labels': ['X']
            }

        Args:
        ----
        fields: Dictionary that maps field names to their locations in the AnnData object.
        adj_matrix_loc: Location of the adjacency matrix within the AnnData object.
        adata2iter: Optional function that converts AnnData objects to iterable.
                    If not given will assume that an iterable is already provided.
        preprocess: List of functions to preprocess the AnnData object before conversion.
                    A default preprocessing step (AddAdjMatrix) is added if adj_matrix_loc is provided.
        transform: List of functions to transform the AnnData object after preprocessing.
        edge_index_key: Key for the edge index in the converted data. Defaults to 'edge_index'.
        """
        super().__init__(fields, adata2iter, preprocess, transform)

        self._preprocess = preprocess
        self._transform = transform
        self._adata2iter = adata2iter

    def merge_field(self, adata: AnnData, field: str, locations: list[str]) -> torch.Tensor:
        """Abstract method for merging multiple fields in an AnnData object.

        Args:
        ----
        adata: AnnData object.
        field: Name of the new field.
        locations: List of locations of the fields to be merged.

        Returns:
        -------
            Merged array corresponding to field.
        """
        if len(locations) == 1:
            obj = get_from_loc(adata, locations[0])
            return self._convert_to_array(obj)
        else:
            arrs = []
            for loc in locations:
                arrs.append(self._convert_to_array(get_from_loc(adata, loc)))
            return torch.from_numpy(np.concatenate(arrs, axis=-1)).to(torch.float)

    def _convert_to_array(self, obj):
        # if obj is categorical
        if isinstance(obj, torch.Tensor):
            return obj
        if obj.dtype.name == "category":
            return pd.get_dummies(obj).to_numpy()
        elif not np.issubdtype(obj.dtype, np.number):
            return obj.astype(np.float)
        elif sparse.issparse(obj):
            return np.array(obj.todense())
        else:
            return obj
