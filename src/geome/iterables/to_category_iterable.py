from typing import Literal

from anndata import AnnData

from .adata2iterable import AnnData2Iterable


class ToCategoryIterable(AnnData2Iterable):
    """Iterates over `adata` by category on the given axis (either obs(0) or var(1))."""

    def __init__(self, category: str, axis: Literal[0, 1, "obs", "var"] = "obs"):
        self.category = category
        if axis not in (0, 1, "obs", "var"):
            raise TypeError("axis needs to be one of obs, var, 0 or 1")
        if isinstance(axis, int):
            axis = ("obs", "var")[axis]
        self.axis = axis

    def __call__(self, adata: AnnData):
        cats_df = getattr(adata, self.axis)[self.category]
        cats = cats_df.dtypes.categories
        for cat in cats:
            yield adata[cats_df == cat]
