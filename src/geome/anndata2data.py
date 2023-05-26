import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from scipy import sparse
from abc import ABC, abstractmethod
from .transforms import get_from_address, get_adjacency_from_adata
from typing import Optional, List, Callable, Any, Dict, Union


class AnnData2Data(ABC):
    def __init__(
        self,
        fields: Dict[str, List[str]],
        adata_iter: Optional[Callable[[Any], Any]] = None,
        preprocess: Optional[List[Callable[[Any, Dict[str, List[str]]], None]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an AnnData to PyTorch Data object converter.

        Args:
            fields: A dictionary that maps field names to a list of addresses.
                Each address specifies the path to a numpy array in the AnnData object
                (i.e., `fields[field_name] = ['attribute/key', 'attribute/key', ...]`).
                (e.g.,  'features':['obs/Cluster_preprocessed','obs/donor','obsm/design_matrix'],
                        'labels':['X']').
            adata_iter: A function that returns an iterable of AnnData objects.
                This function will be used to extract multiple sub-AnnData objects from a larger AnnData object.
                If set to None, the object will assume adata_iter returns only the input AnnData object.
            preprocess: A list of callables that take an AnnData object and the fields dictionary as inputs and
                perform data preprocessing steps on the AnnData object before conversion to PyTorch Data objects.
            *args: Any additional arguments to be passed to subclass constructors.
            **kwargs: Any additional keyword arguments to be passed to subclass constructors.
        """
        self._adata_iter = adata_iter
        self._preprocess = preprocess
        self.fields = fields

    @abstractmethod
    def get_adj_matrix(self, adata: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for computing adjacency matrix from AnnData object.

        Args:
            adata: AnnData object.

        Returns:
            Adjacency matrix.
        """
        pass

    def get_edge_index(self, adata: Any, adj_matrix: Any) -> Any:
        """
        Computes edge index tensor from adjacency matrix.

        Args:
            adata: AnnData object.
            adj_matrix: Adjacency matrix.

        Returns:
            Edge index tensor.
        """
        nodes1, nodes2 = adj_matrix.nonzero()
        return torch.vstack(
            [
                torch.from_numpy(nodes1).to(torch.long),
                torch.from_numpy(nodes2).to(torch.long),
            ]
        )

    @abstractmethod
    def array_from_address(self, adata: Any, address: str, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Abstract method for retrieving a numpy array from an AnnData object.

        Args:
            adata: AnnData object.
            address: Tuple of key and attribute for the numpy array.

        Returns:
            Numpy array.
        """
        pass

    def create_data_obj(self, adata: Any, adj_matrix: Any) -> Data:
        """
        Creates a PyTorch Data object from an AnnData object.

        Args:
            adata: AnnData object.
            adj_matrix: Adjacency matrix.

        Returns:
            PyTorch Data object.
        """
        obj = dict()
        if self.yields_edge_index:
            obj["edge_index"] = self.get_edge_index(adata, adj_matrix)

        for field, addresses in self.fields.items():
            arrs = []
            for address in addresses:
                arrs.append(self.array_from_address(adata, address))
            obj[field] = torch.from_numpy(np.concatenate(arrs, axis=-1)).to(torch.float)

        return Data(**obj)

    def __call__(self, adata: Any) -> List[Any]:
        """
        Convert an AnnData object to a list of PyTorch compatible data objects.

        Args:
            adata: The AnnData object to be converted.

        Returns:
            A list of PyTorch compatible data objects.
        """
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
    def __init__(
        self,
        fields: Dict[str, List[str]],
        adata_iter: Union[None, Callable[[Any], Any]] = None,
        preprocess: Union[None, List[Callable[[Any], Any]]] = None,
        yields_edge_index: bool = True,
    ) -> None:
        """
        Convert anndata object into a dictionary of arrays.

        Assumes adata.obsp["adjacency_matrix_connectivities"] exists
        if not it is computed with sq.gr.spatial_neighbors.
        Works for squidpy datasets and the datasets of this package.

        Example for fields argument:
            fields = {
                'features': ['obs/Cluster', 'obs/donor'],
                'labels': ['X']
            }

        Args:
            fields: A dictionary of field names and their addresses in the AnnData object.
            adata_iter: An iterator function that returns an AnnData object.
            preprocess: A list of functions to preprocess the input data.
                This class by default adds a preprocessing step.
                See the static method default_preprocess.
            yields_edge_index: Whether to return the edge index of the adjacency matrix.
        """
        super().__init__(fields, adata_iter, preprocess)
        # Default is the identity function.
        self._adata_iter = adata_iter
        self._preprocess = [AnnData2DataDefault.convert_to_array]
        # Add preprocessing of the addresses to last.
        # So that get_as_array works properly.
        self._preprocess = (preprocess if preprocess is not None else []) + self._preprocess
        self.yields_edge_index = yields_edge_index

    def get_adj_matrix(self, adata: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Helper function to create an adjacency matrix depending on the anndata object.

        Args:
            adata: An AnnData object.
            args: Additional arguments passed to the function get_adjacency_from_adata.
            kwargs: Additional keyword arguments passed to the function get_adjacency_from_adata.

        Returns:
            The adjacency matrix.
        """
        # Get adjacency matrices
        return get_adjacency_from_adata(adata, *args, **kwargs)

    def array_from_address(self, adata: Any, address: str) -> Union[np.ndarray, sparse.spmatrix]:
        """Return the array corresponding to the given address.

        This version assumes the addresses are stored on adata.uns.

        Args:
            adata: An AnnData object.
            address: The address of the field in the anndata object.

        Returns:
            A numpy array or a sparse matrix.
        """
        processed_address = adata.uns["processed_index"][address]
        return get_from_address(adata, processed_address)

    @staticmethod
    def convert_to_array(adata: Any, fields: Dict[str, List[str]]) -> None:
        """
        Store processed data in `obsm` field of `adata`.
        Store the new address for each processed data in `uns` field of `adata`.

        Parameters
        ----------
        adata : Any
            AnnData object.
        fields : Dict[str, List[str]]
            Dictionary of fields and their associated attributes.

        Returns
        -------
        None
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
    """
    A class to transform AnnData objects into Data objects by category.

    Parameters
    ----------
    fields : dict
        Dictionary of fields and their associated attributes.
    category : str
        Column in `obs` containing the categories.
    preprocess : callable, optional
        Function to preprocess data before transformation.
    yields_edge_index : bool, default=True
        Whether to yield edge index.

    Attributes
    ----------
    adata_iter : callable
        Function to iterate over `adata` by category.
    """

    def __init__(self, fields: dict, category: str, preprocess=None, yields_edge_index: bool = True):
        """
        Initializes the class.

        Parameters
        ----------
        fields : dict
            Dictionary of fields and their associated attributes.
        category : str
            Column in `obs` containing the categories.
        preprocess : callable, optional
            Function to preprocess data before transformation.
        yields_edge_index : bool, default=True
            Whether to yield edge index.
        """
        super().__init__(
            fields=fields,
            adata_iter=lambda x: AnnData2DataByCategory.adata_iter_category(x, category),
            preprocess=preprocess,
            yields_edge_index=yields_edge_index,
        )

    @staticmethod
    def adata_iter_category(adata: Any, c: str) -> Any:
        """
        Iterates over `adata` by category.

        Parameters
        ----------
        adata : Any
            AnnData object.
        c : str
            Column in `obs` containing the categories.

        Yields
        -------
        Any
            Data object by category.
        """
        cats = adata.obs[c].dtypes.categories
        for cat in cats:
            yield adata[adata.obs[c] == cat]
