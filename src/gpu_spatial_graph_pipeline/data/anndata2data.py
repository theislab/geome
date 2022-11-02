import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from scipy import sparse
from abc import ABC, abstractmethod
from .utils.transforms import get_from_address, get_adjacency_from_adata


class AnnData2Data(ABC):
    def __init__(self, fields, adata_iter=None, preprocess=None, *args, **kwargs):
        """_summary_

        Args:
            fields (_type_): _description_
            adata_iter (_type_, optional): _description_. Defaults to None.
            preprocess (_type_, optional): List of callables with signature of fn(adata,fields).
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
        return torch.vstack(
            [
                torch.from_numpy(nodes1).to(torch.long),
                torch.from_numpy(nodes2).to(torch.long),
            ]
        )

    @abstractmethod
    def array_from_address(self, adata, address, *args, **kwargs):
        pass

    def create_data_obj(self, adata, adj_matrix):
        obj = dict()
        if self.yields_edge_index:
            obj["edge_index"] = self.get_edge_index(adata, adj_matrix)

        for field, addresses in self.fields.items():
            arrs = []
            for address in addresses:
                arrs.append(self.array_from_address(adata, address))
            obj[field] = torch.from_numpy(np.concatenate(arrs, axis=-1)).to(torch.float)

        return Data(**obj)

    def __call__(self, adata):
        dataset = []
        # do the given preprocessing steps.
        for process in self._preprocess:
            process(adata, self.fields)
        # iterate trough adata.
        for subadata in self._adata_iter(adata):
            adj_matrix = self.get_adj_matrix(subadata)
            data = self.create_data_obj(subadata, adj_matrix)
            dataset.append(data)

        return dataset


class AnnData2DataDefault(AnnData2Data):
    def __init__(self, fields, adata_iter, preprocess=None, yields_edge_index=True):
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
            adata_iter (_type_, optional): _description_.
            preprocess (_type_, optional): _description_.
                This class by default adds a preprocessing step.
                See the static method default_preprocess.
                Defaults to None.
            yields_edge_index (bool, optional): _description_.
                Defaults to True.
        """
        super().__init__(fields, adata_iter, preprocess)
        # Default is the identity function.
        self._adata_iter = adata_iter
        self._preprocess = [AnnData2DataDefault.convert_to_array]
        # Add preprocessing of the addresses to last.
        # So that get_as_array works properly.
        self._preprocess = (
            preprocess if preprocess is not None else []
        ) + self._preprocess
        self.yields_edge_index = yields_edge_index

    def get_adj_matrix(self, adata, *args, **kwargs):
        """helper function to create adj matrix depending on the adata"""
        # Get adjacency matrices
        return get_adjacency_from_adata(adata, *args, **kwargs)

    def array_from_address(self, adata, address):
        """This version assumes the addresses are stored on adata.uns

        Args:
            adata (_type_): _description_
            address (_type_): _description_

        Returns:
            _type_: _description_
        """
        processed_address = adata.uns["processed_index"][address]
        return get_from_address(adata, processed_address)

    @staticmethod
    def convert_to_array(adata, fields):
        """Adds the addresses that map fields addresses with
        the addresses on the adata it exists on.
        """
        adata.uns["processed_index"] = dict()

        for _, addresses in fields.items():

            for address in addresses:
                last_attr = address.split("/")[-1]
                save_name = last_attr + "_processed"
                # TODO: Is adding a suffix a good idea?

                obj = get_from_address(adata, address)

                # if obj is categorical
                if obj.dtype.name == "category":
                    adata.obsm[save_name] = pd.get_dummies(obj).to_numpy()
                    adata.uns["processed_index"][address] = "obsm/" + save_name
                elif not np.issubdtype(obj.dtype, np.number):
                    adata.obsm[save_name] = obj.astype(np.float)
                    adata.uns["processed_index"][address] = "obsm/" + save_name
                elif sparse.issparse(obj):
                    adata.obsm[save_name] = np.array(obj.todense())
                    adata.uns["processed_index"][address] = "obsm/" + save_name

                # If no storing required
                else:
                    adata.uns["processed_index"][address] = address


class AnnData2DataByCategory(AnnData2DataDefault):
    def __init__(self, fields, category, preprocess=None, yields_edge_index=True):
        super().__init__(
            fields=fields,
            adata_iter=lambda x: AnnData2DataByCategory.adata_iter_category(
                x, category
            ),
            preprocess=preprocess,
            yields_edge_index=yields_edge_index,
        )

    @staticmethod
    def adata_iter_category(adata, c):
        cats = adata.obs[c].dtypes.categories
        for cat in cats:
            yield adata[adata.obs[c] == cat]
