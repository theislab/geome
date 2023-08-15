from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from anndata import AnnData
from torch_geometric.data import Data


class AnnData2Data(ABC):
    """
    Abstract class for transforming an iterable of AnnData to Pytorch Geometric Data object.

    This class provides a blueprint for converting AnnData objects, commonly used in single-cell
    genomics, into PyTorch Geometric Data objects suitable for graph-based machine learning.
    """

    def __init__(
        self,
        fields: Dict[str, List[str]],
        adata2iterable: Optional[Callable[[AnnData], Iterable[AnnData]]] = None,
        preprocess: Optional[Callable[[AnnData], AnnData]] = None,
        transform: Optional[Callable[[AnnData], AnnData]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the converter from AnnData to PyTorch Data object.

        Args:
            fields (Dict[str, List[str]]): Dictionary mapping field names to addresses in the AnnData object.
            adata2iterable (Callable, optional): Function returning an iterable of AnnData objects.
            preprocess (Callable, optional): Function for preprocessing the AnnData object before conversion.
            transform (Callable, optional): Function for transforming the AnnData object after preprocessing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._adata2iterable = adata2iterable
        self._preprocess = preprocess
        self.fields = fields
        self._transform = transform

    @abstractmethod
    def array_from_address(
        self, adata: AnnData, address: str, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Retrieve a numpy array from an AnnData object based on the provided address.

        Args:
            adata (AnnData): The AnnData object containing the data.
            address (str): Address for the numpy array.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Retrieved array.
        """
        pass

    def create_data_obj(self, adata: AnnData) -> Data:
        """
        Convert an AnnData object to a PyTorch Data object.

        Args:
            adata (AnnData): The AnnData object to be converted.

        Returns:
            Data: PyTorch Geometric Data object.
        """
        obj = {}
        if hasattr(self, "yields_edge_index") and self.yields_edge_index:
            obj["edge_index"] = self.get_edge_index(adata)

        for field, addresses in self.fields.items():
            arrs = [self.array_from_address(adata, address) for address in addresses]
            obj[field] = torch.from_numpy(np.concatenate(arrs, axis=-1)).to(torch.float)

        return Data(**obj)

    def __call__(self, adata: Union[AnnData, Iterable[AnnData]]) -> List[Data]:
        """
        Convert an AnnData object to a list of PyTorch compatible data objects.

        Args:
            adata (Union[AnnData, Iterable[AnnData]]): The AnnData object or iterable of AnnData objects.

        Returns:
            List[Data]: List of PyTorch Geometric Data objects.
        """
        dataset = []
        if self._preprocess:
            adata = self._preprocess(adata)
        adata_iter = self._adata2iterable(adata) if self._adata2iterable else [adata]

        for subadata in adata_iter:
            if self._transform:
                sub
