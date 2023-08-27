from abc import ABC, abstractmethod

from anndata import AnnData


class Transform(ABC):
    """Abstract class for Transform.

    A Transform is a callable that takes an AnnData object and returns a transformed AnnData object.
    """

    @abstractmethod
    def __call__(self, adata: AnnData) -> AnnData:
        """Abstract method for transforming an AnnData object."""
        pass
