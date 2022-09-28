from dataclasses import field
from typing import overload
import squidpy as sq
import torch
import pandas as pd
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data
from scipy import sparse

class AnnData2DataCallable:
    def __init__(self, fields, adata_iter=None, *args, **kwargs):
        self._adata_iter = adata_iter
        self.fields = fields

    def _get_adj_matrix(self, adata, *args, **kwargs):
        return NotImplementedError()

    def _create_data_obj(self, adata, spatial_connectivities, *args, **kwargs):
        return NotImplementedError()

    def __call__(self, adatas):
        dataset = []

        for adata in self._adata_iter(adatas, self.fields):
            spatial_connectivities = self._get_adj_matrix(adata)
            data = self._create_data_obj(adata, spatial_connectivities)
            dataset.append(data)

        return dataset


class AnnData2DataCallableDefault(AnnData2DataCallable):
    def __init__(self, fields, adata_iter, yields_edge_index=True):
        """
        assumes is called adjacency_matrix_connectivities in adata

        TODO: Specify fields format.
        """
        super().__init__(fields, adata_iter)
        if self._adata_iter is None:
            self._adata_iter = lambda x, _: x
        self.yields_edge_index = yields_edge_index

    # @overload
    def _get_adj_matrix(self, adata, *args, **kwargs):
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

    @staticmethod
    def _get_sq_adata_iter(adata, fields):
        adata.uns["processed_index"] = dict()

        for k, addresses in fields.items():

            for address in addresses:
                attrs = address.split("/")
                assert len(attrs) <= 2, "assumes at most one delimiter"

                if len(attrs) != 1:
                    # for each attr we find the corresponding value
                    obj = adata
                    for attr in attrs:
                        obj = getattr(obj, attr)

                    last_attr = attrs[-1]
                    # if obj is categorical
                    if obj.dtype.name == "category":
                        adata.obsm[last_attr] = pd.get_dummies(obj).to_numpy()
                        adata.uns["processed_index"][address] = "obsm/" + last_attr
                    else:
                        adata.uns["processed_index"][address] = address

                else:
                    adata.uns["processed_index"][address] = address

        cats = adata.obs.library_id.dtypes.categories
        for cat in cats:
            yield adata[adata.obs.library_id == cat]

    @staticmethod
    def _get_as_array(adata, address):
        processed_address = adata.uns["processed_index"][address]
        obj = adata
        attr = processed_address.split("/")
        for attr in processed_address.split("/"):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)  # obj.attr
            else:
                obj = obj[attr]
        if sparse.issparse(obj):
            obj = np.array(obj.todense())
        return obj

    # @overload
    def _create_data_obj(self, adata, spatial_connectivities):
        obj = dict()
        if self.yields_edge_index:
            nodes1, nodes2 = spatial_connectivities.nonzero()
            obj["edge_index"] = torch.vstack(
                [
                    torch.from_numpy(nodes1).to(torch.long),
                    torch.from_numpy(nodes2).to(torch.long),
                ]
            )

        # just for pytorch_geometric naming thing
        sq2pyg = {
            "features": "x",
            "labels": "y",
            "condition": "xc",
        }

        for field, addresses in self.fields.items():
            arrs = []
            for address in addresses:
                arrs.append(AnnData2DataCallableDefault._get_as_array(adata, address))
            obj[sq2pyg[field]] = torch.from_numpy(np.concatenate(arrs, axis=-1))

        return Data(**obj)
