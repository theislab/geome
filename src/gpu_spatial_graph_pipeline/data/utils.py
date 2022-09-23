import squidpy as sq
import torch
import pandas as pd
import numpy as np
from typing import Dict, Sequence, Set, Union
from torch_geometric.data import Data
from anndata import AnnData
import scipy
import torch.nn as nn


class AnnData2DataCallable:
    def __init__(
        self,
        x_names: Dict[str, str],
        y_names: Dict[str, str],
        is_sq: bool = False,
        has_edge_index: bool = True,
    ):
        self.is_sq = is_sq
        self.has_edge_index = has_edge_index
        if is_sq:
            self._adata_iter = AnnData2DataCallable._get_sq_adata_iter
        else:
            self._adata_iter = lambda x: x  # identity

        self.xs = x_names
        self.ys = y_names

    @staticmethod
    def _get_sq_adata_iter(adata):
        cats = adata.obs.library_id.dtypes.categories
        for cat in cats:
            yield adata[adata.obs.library_id == cat]

    @staticmethod
    def _get_adj_matrix(adata):
        """helper function to create adj matrix depending on the adata"""

        # Get adjacency matrices
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

    def _extract_features(self, obj, adata):
        raise NotImplementedError()

    def _concat_features(adata, obj, features_dict):
        raise NotImplementedError()

    def _create_data_obj(self, adata, spatial_connectivities):
        obj = dict()
        if self.has_edge_index:
            nodes1, nodes2 = spatial_connectivities.nonzero()
            obj["edge_index"] = torch.vstack(
                [
                    torch.from_numpy(nodes1).to(torch.long),
                    torch.from_numpy(nodes2).to(torch.long),
                ]
            )

        # extract general features e.g., gene_expression, cell_type
        features_dict = self._extract_features(adata, obj)

        # assign x's and y's here as one torch tensor
        obj = self._concat_features(adata, obj, features_dict)

        return Data(**obj)

    def __call__(self, adatas):
        dataset = []

        for adata in self._adata_iter(adatas):
            spatial_connectivities = self._get_adj_matrix(adata)
            data = self._create_data_obj(adata, spatial_connectivities)
            dataset.append(data)

        return dataset
