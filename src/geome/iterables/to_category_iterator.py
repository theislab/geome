from typing import Literal
from collections.abc import Iterator

from anndata import AnnData

from geome.utils import get_from_loc

from .base.to_iterable import ToIterable


class ToCategoryIterator(ToIterable):
    """Iterates over `adata` by category on the given axis (either obs(0) or var(1)).

    Preserves the categories in the resulting AnnData obs and var Series.
    """

    def __init__(self, category: str, axis: Literal[0, 1, "obs", "var"] = "obs", preserve_categories: bool = True):
        self.category = category
        if axis not in (0, 1, "obs", "var"):
            raise TypeError("axis needs to be one of obs, var, 0 or 1")
        if isinstance(axis, int):
            axis = ("obs", "var")[axis]
        self.axis = axis
        self.preserve_categories = preserve_categories

    def __call__(self, adata: AnnData) -> Iterator[AnnData]:
        """Iterates over `adata` by category on the given axis (either obs(0) or var(1)).

        Preserves the categories in the resulting AnnData obs and var Series if `preserve_categories` is True. 
        Returns an iterator, which means it can be only iterated once per call.

        Args:
        ----
        adata (AnnData): AnnData object to iterate over.

        Yields
        ------
        adata[adata.axis[category] == cat] for each cat in adata.axis[category].categories
        """
        cats_df = get_from_loc(adata, f"{self.axis}/{self.category}")
        cats = cats_df.dtypes.categories
        preserved_categories = {"obs": {}, "var": {}}
        if self.preserve_categories:
            for axis in ("obs", "var"):
                adata_axis = getattr(adata, axis)
                if adata_axis is not None:
                    for key in adata_axis.keys():
                        if adata_axis[key].dtype.name == "category":
                            preserved_categories[axis][key] = adata_axis[key].cat.categories

        for cat in cats:
            # TODO(syelman): is this wise? Maybe create copy only if preserve_categories is True?
            subadata = adata[cats_df == cat].copy()
            # if categories are preserved
            if self.preserve_categories:
                for axis in ("obs", "var"):
                    for key, val in preserved_categories[axis].items():
                        if key in getattr(subadata, axis):
                            subadata_axis = getattr(subadata, axis)
                            subadata_axis[key] = subadata_axis[key].cat.set_categories(val)
            yield subadata
