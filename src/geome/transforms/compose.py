from __future__ import annotations

from typing import Callable

from anndata import AnnData

from .transform import Transform


class Compose(Transform):
    """Composes several transforms together as a single transform."""

    def __init__(self, transforms: list[Callable[[AnnData], AnnData]]):
        self.transforms = transforms

    def __call__(self, adata: AnnData) -> AnnData:
        """Applies each transform in order."""
        for t in self.transforms:
            adata = t(adata)
        return adata
