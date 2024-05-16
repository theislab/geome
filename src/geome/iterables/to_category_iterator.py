from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from anndata import AnnData

from geome.utils import get_from_loc

from .base.to_iterable import ToIterable


@dataclass
class ToCategoryIterator(ToIterable):
    """Iterates over `adata` by category on the given axis (either obs(0) or var(1)).

    Args:
    ----
    category (str): The category to iterate over.
    axis (int | str): The axis along which to iterate over the categories. Can be either 0, 1, "obs" or "var".
        0 or "obs" means the categories are in the observation axis.
        1 or "var" means the categories are in the variable axis.
    preserve_categories (bool | list[str]): Preserves the categories in the resulting AnnData obs and var Series if `preserve_categories` is True. If a list is provided, only the categories in the list will be preserved.
        The list should contain locations of in the following format: ['obs/key1', 'var/key2'].
    """

    category: str
    axis: Literal[0, 1, "obs", "var"] = "obs"
    preserve_categories: bool | list[str] = True

    def __post_init__(self):
        if self.axis not in (0, 1, "obs", "var"):
            raise TypeError("axis needs to be one of obs, var, 0 or 1")
        if isinstance(self.axis, int):
            self.axis = ("obs", "var")[self.axis]
        if isinstance(self.preserve_categories, list):
            for loc in self.preserve_categories:
                if loc.count("/") != 1 or "/" in loc[-1] + loc[0]:
                    raise ValueError("preserve_categories should be a list of strings with format 'axis/key'.")
                axis, key = loc.split("/")
                if axis not in ("obs", "var"):
                    raise TypeError("axis needs to be one of obs, var")

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
        if isinstance(self.preserve_categories, list):
            for loc in self.preserve_categories:
                axis, key = loc.split("/")
                preserved_categories[axis][key] = adata[axis][key].cat.categories
        elif self.preserve_categories:
            for axis in ("obs", "var"):
                adata_axis = getattr(adata, axis)
                for key in adata_axis.keys():
                    if adata_axis[key].dtype.name == "category":
                        preserved_categories[axis][key] = adata_axis[key].cat.categories
        for cat in cats:
            # TODO(syelman): is this wise? Maybe create copy only if preserve_categories is True?
            subadata = adata[cats_df == cat].copy()
            # if categories are preserved
            for axis in ("obs", "var"):
                for key, val in preserved_categories[axis].items():
                    if key in getattr(subadata, axis):
                        subadata_axis = getattr(subadata, axis)
                        subadata_axis[key] = subadata_axis[key].cat.set_categories(val)
            yield subadata
