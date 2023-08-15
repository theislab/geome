from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from anndata import AnnData
from torch_geometric.data import Data


class AnnData2Data(ABC):
    """Abstract class that transforms an iterable of AnnData to Pytorch Geometric Data object."""

    def __init__(
        self,
        fields: Dict[str, List[str]],
        adata2iterable: Optional[Callable[[AnnData], Iterable[AnnData]]] = None,
        preprocess: Optional[Callable[[AnnData], AnnData]] = None,
        transform: Optional[Callable[[AnnData], AnnData]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes an AnnData to PyTorch Data object converter.

        Args:
        ----
        fields: Dictionary mapping field names to addresses in the AnnData object. Each address points to a numpy array.
        adata2iterable: Optional function that returns an iterable of sub-adata objects from a larger AnnData object.
                        If not provided, it's assumed that the input AnnData object is directly converted.
        preprocess: Optional function to preprocess the AnnData object before conversion.
        transform: Optional function to apply transformations on the AnnData object after preprocessing.
        *args: Additional positional arguments for subclasses.
        **kwargs: Additional keyword arguments for subclasses.
        """
        self._adata2iterable = adata2iterable
        self._preprocess = preprocess
        self.fields = fields
        self._transform = transform

    @abstractmethod
    def array_from_location(
        self, adata: Any, location: str, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Abstract method for retrieving a numpy array from an AnnData object.

        Args:
        ----
        adata: AnnData object.
        location: Location of the numpy array in the AnnData object.
        args: additional args
        kwargs: additional args

        Returns:
        -------
            Numpy array.
        """
        pass

    def create_data_obj(self, adata: AnnData) -> Data:
        """Creates a PyTorch Data object from an AnnData object.

        Args:
        ----
        adata: AnnData object.

        Returns:
        -------
        PyTorch Data object.
        """
        obj = {}
        if self.yields_edge_index:
            obj["edge_index"] = self.get_edge_index(adata)

        for field, addresses in self.fields.items():
            arrs = []
            for address in addresses:
                arrs.append(self.array_from_address(adata, address))
            obj[field] = torch.from_numpy(np.concatenate(arrs, axis=-1)).to(torch.float)

        return Data(**obj)

    def __call__(self, adata: Union[AnnData, Iterable[AnnData]]) -> List[Data]:
        """Convert an AnnData object to a list of PyTorch compatible data objects.

        Args:
        ----
        adata: The AnnData object to be converted.

        Returns:
        -------
        A list of PyTorch compatible data objects.
        """
        dataset = []
        # do the given preprocessing steps.
        if self._preprocess is not None:
            adata = self._preprocess(adata)
        # convert adata to iterable if it is specified
        adata_iter = adata
        if self._adata2iterable is not None:
            adata_iter = self._adata2iterable(adata)

        # iterate trough adata.
        for subadata in adata_iter:
            if self._transform is not None:
                subadata = self._transform(subadata)
            data = self.create_data_obj(subadata)
            dataset.append(data)

        return dataset
