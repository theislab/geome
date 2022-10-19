import squidpy as sq
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from scipy import sparse
from abc import ABC, abstractmethod


class AnnData2Data(ABC):
    def __init__(self, fields, adata_iter=None, preprocess=None, *args, **kwargs):
        """_summary_

        Args:
            fields (_type_): _description_
            adata_iter (_type_, optional): _description_. Defaults to None.
            preprocess (_type_, optional): List of callables with signature of fn(adatas,fields).
                Defaults to None.
        """
        self._adata_iter = adata_iter
        self._preprocess = preprocess
        self.fields = fields

    @abstractmethod
    def get_adj_matrix(self, adata, *args, **kwargs):
        pass

    def get_edge_index(self, adata, adj_matrix):
        nodes1, nodes2 = adj_matrix.nonzero()
        return torch.vstack([
            torch.from_numpy(nodes1).to(torch.long),
            torch.from_numpy(nodes2).to(torch.long),
        ])

    @abstractmethod
    def get_as_array(self, adata, address, *args, **kwargs):
        pass

    def create_data_obj(self, adata, adj_matrix):
        obj = dict()
        if self.yields_edge_index:
            obj["edge_index"] = self.get_edge_index(adata, adj_matrix)

        # just for pytorch_geometric naming thing
        sq2pyg = {
            "features": "x",
            "labels": "y",
            "condition": "xc",
        }

        for field, addresses in self.fields.items():
            arrs = []
            for address in addresses:
                arrs.append(self.get_as_array(adata, address))
            obj[sq2pyg[field]] = torch.from_numpy(np.concatenate(arrs, axis=-1))

        return Data(**obj)

    def __call__(self, adatas):
        dataset = []
        # do the given preprocessing steps.
        for process in self._preprocess:
            process(adatas, self.fields)
        # iterate trough adata.
        for adata in self._adata_iter(adatas, self.fields):
            adj_matrix = self.get_adj_matrix(adata)
            data = self.create_data_obj(adata, adj_matrix)
            dataset.append(data)

        return dataset


class AnnData2DataDefault(AnnData2Data):
    def __init__(self, fields, adata_iter=None, preprocess=None, yields_edge_index=True):
        """
        Assumes adata.obsp["adjacency_matrix_connectivities"] exists
        if not it is computed with sq.gr.spatial_neighbors.
        Works for squidpy datasets and the datasets of this package.

        Example for fields argument:
            fields = {
                'features':['obs/Cluster','obs/donor'],
                'labels':['X']
            }

        Args:
            fields (_type_): _description_
            adata_iter (_type_, optional): _description_. If set to None, will 
                be equivalent to
                the identity function. Which is the use case when the class is
                called with a list of adata objects,
                so the iterator would be the list object itself.
                Defaults to None.
            preprocess (_type_, optional): _description_.
                This class by default adds a preprocessing step.
                See the static method default_preprocess.
                Defaults to None.
            yields_edge_index (bool, optional): _description_.
                Defaults to True.
        """
        super().__init__(fields, adata_iter, preprocess)
        # Default is the identity function.
        if self._adata_iter is None:
            self._adata_iter = lambda x, _: x
        if self._preprocess is None:
            self._preprocess = []
        # Add preprocessing of the addresses to the front.
        # So that get_as_array works properly.
        self._preprocess.insert(0, AnnData2DataDefault.default_preprocess)
        self.yields_edge_index = yields_edge_index

    def get_adj_matrix(self, adata, *args, **kwargs):
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

    def get_as_array(self, adata, address):
        """This version assumes the addresses are stored on adata.uns

        Args:
            adata (_type_): _description_
            address (_type_): _description_

        Returns:
            _type_: _description_
        """    
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

    @staticmethod
    def default_preprocess(adata, fields):
        """Adds the addresses that map fields addresses with
        the addresses on the adata it exists on.
        """
        adata.uns["processed_index"] = dict()

        for _, addresses in fields.items():

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


class AnnData2DataSq(AnnData2DataDefault):
    @staticmethod
    def sq_adata_iter(adata, fields):
        cats = adata.obs.library_id.dtypes.categories
        for cat in cats:
            yield adata[adata.obs.library_id == cat]




class AnnData2DataSq(AnnData2DataCallable):
    def __init__(self, fields, adata_iter, yields_edge_index=True):
        """
        Assumes is called adjacency_matrix_connectivities in adata.
        Works for squidpy datasets and the datasets of this package.

        Example for fields argument:
            fields = {
                'features':['obs/Cluster','obs/donor'],
                'labels':['X']
            }
        """
        super().__init__(fields, adata_iter)
        # Default is the identity function.
        if self._adata_iter is None:
            self._adata_iter = lambda x, _: x
        self.yields_edge_index = yields_edge_index

    def get_adj_matrix(self, adata, *args, **kwargs):
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
    def sq_adata_iter(adata, fields):
        cats = adata.obs.library_id.dtypes.categories
        for cat in cats:
            yield adata[adata.obs.library_id == cat]

    def get_as_array(self, adata, address):
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



