from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from anndata import AnnData

from .base.transform import Transform


@dataclass
class Categorize(Transform):
    """Converts the given list of observation columns in the AnnData object to categorical.

    Args:
    ----
    keys (str | list): The list of observation columns or a single observation column to convert to categorical.
    axis (int | str): The axis along which to convert the columns to categorical. Can be either 0, 1, "obs" or "var".
        0 or "obs" means the columns are in the observation axis.
        1 or "var" means the columns are in the variable axis.

    One should do this if they expect one-hot encoding to be performed on the given columns.
    """

    keys: str | list
    axis: Literal[0, 1, "obs", "var"] = "obs"

    def __post_init__(self):
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        if self.axis not in (0, 1, "obs", "var"):
            raise TypeError("axis needs to be one of obs, var, 0 or 1")
        if isinstance(self.axis, int):
            self.axis = ("obs", "var")[self.axis]

    def __call__(self, adata: AnnData):  # noqa: D102
        for key in self.keys:
            getattr(adata, self.axis)[key] = getattr(adata, self.axis)[key].astype("category")
        return adata
