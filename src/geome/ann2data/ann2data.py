from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import torch
from anndata import AnnData
from torch_geometric.data import Data


class Ann2Data(ABC):
    """Abstract class that transforms an iterable of AnnData to Pytorch Geometric Data objects."""

    def __init__(
        self,
        fields: dict[str, list[str]],
        adata2iterable: Callable[[AnnData], Iterable[AnnData]] | None = None,
        preprocess: Callable[[AnnData], AnnData] | None = None,
        transform: Callable[[AnnData], AnnData] | None = None,
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
    def merge_field(self, adata:AnnData, field: str, locations: list[str]) -> torch.Tensor:
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
        for field, locations in self.fields.items():
            obj[field] = self.merge_field(adata, field, locations)
        return Data(**obj)

    def __call__(self, adata: AnnData | Iterable[AnnData]) -> Iterable[Data]:
        """Convert an AnnData object to a PyTorch compatible data object.

        Args:
        ----
        adata: The AnnData object to be converted.

        Yields:
        ------
        PyTorch Geometric compatible data object.

        """
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
            yield self.create_data_obj(subadata)

    def to_list(self, adata: AnnData | Iterable[AnnData]) -> list[Data]:
        """Convert an AnnData object to a list of PyTorch compatible data objects.

        Args:
        ----
        adata: The AnnData object to be converted.

        Returns:
        -------
        A list of PyTorch Geometric compatible data objects.
        """
        return list(self(adata))
