from typing import (
    Callable,
    List,
)

from anndata import AnnData


class Compose:
    """Iterates over `adata` by category on the given axis (either obs(0) or var(1))."""

    def __init__(self, transforms: List[Callable[[AnnData], AnnData]]):
        self.transforms = transforms

    def __call__(self, adata: AnnData) -> AnnData:
        for t in self.transforms:
            adata = t(adata)
        return adata
