from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy import sparse

from geome.utils import get_from_loc

from .base.abstract import Ann2DataAbstract


class Ann2DataBasic(Ann2DataAbstract):
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
        self._fields = fields.copy()
        self._last_fields_info = {}
        for field, locations in self.fields.items():
            self._last_fields_info[field] = {"locations": locations}
            # fill in at merge_fields
            self._last_fields_info[field]["sizes"] = []

    def merge_field(self, adata: AnnData, field: str, locations: list[str]) -> torch.Tensor:
        """Method for merging multiple fields in an AnnData object.

        Args:
        ----
        adata: AnnData object.
        field: Name of the new field.
        locations: List of locations of the fields to be merged.

        Returns
        -------
        Merged array corresponding to field.
        """
        arrs = []
        for loc in locations:
            arrs.append(self._convert_to_tensor(get_from_loc(adata, loc)))
        self._last_fields_info[field]["sizes"] = [
            arr.shape if len(arrs) == 1 else (*(len(arr.shape) - 1) * ["-"], arr.shape[-1]) for arr in arrs
        ]
        return torch.cat(arrs, dim=-1)

    def _convert_to_tensor(self, obj):
        if torch.is_tensor(obj):
            return obj
        if sparse.issparse(obj):
            return torch.from_numpy(obj.todense()).to(torch.float)
        if obj.dtype.name == "category":
            return torch.from_numpy(pd.get_dummies(obj).to_numpy()).to(torch.float)
        if not np.issubdtype(obj.dtype, np.number):
            return torch.from_numpy(obj.astype(np.float64)).to(torch.float)
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(torch.float)
        else:
            return obj  # TODO(selman): throw error here

    def __repr__(self) -> str:
        s = ""
        pad = " " * 4
        for field, info in self._last_fields_info.items():
            s += f"{field}:" + "\n"
            sizes = info["sizes"] if info["sizes"] else [("?")] * len(info["locations"])
            s += pad + "-----------------------------------\n"
            for loc, size in zip(info["locations"], sizes):
                s += pad + f"{loc} {tuple(size)}\n"
            s += pad + "-----------------------------------\n"
        return s
