from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from anndata import AnnData


class ToIterable(ABC):
    """Abstract class for creating an iterable of AnnData objects."""

    @abstractmethod
    def __call__(self, adata: AnnData) -> Iterable[AnnData]:
        """Abstract method for creating an iterable of AnnData objects."""
        pass
