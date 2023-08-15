from typing import Literal

from anndata import AnnData

from .adata2iterable import AnnData2Iterable
from geome.utils import get_from_loc


class ToCategoryIterable(AnnData2Iterable):
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

    def __call__(self, adata: AnnData):
        cats_df = get_from_loc(adata, f"{self.axis}/{self.category}")
        cats = cats_df.dtypes.categories
        obs_cats = {}
        var_cats = {}
        if self.preserve_categories:
            # TODO: maybe avoid duplicate code here later
            if adata.obs is not None:
                for key in adata.obs.keys():
                    if adata.obs[key].dtype.name == "category":
                        obs_cats[key] = adata.obs[key].cat.categories
            if adata.var is not None:
                for key in adata.var.keys():
                    if adata.var[key].dtype.name == "category":
                        var_cats[key] = adata.var[key].cat.categories

        for cat in cats:
            subadata = adata[cats_df == cat].copy()
            for k, v in obs_cats.items():
                subadata.obs[k] = subadata.obs[k].cat.set_categories(v)
            for k, v in var_cats.items():
                subadata.var[k] = subadata.var[k].cat.set_categories(v)
            yield subadata
