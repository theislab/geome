from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData

from .base.transform import Transform


class Subset(Transform):
    """
    Create a subset of an AnnData object based on specified observation or feature values.

    Parameters
    ----------
    key_value : dict
        A dictionary where keys are observation or feature names, and values are lists of values to keep.
    axis : Literal[0, 1, "obs", "var"], optional
        The axis to subset on. It can be 0 or "obs" for observations and 1 or "var" for features.
        Default is "obs".
    copy : bool, optional
        If True, return a copy of the subsetted AnnData object instead of a view. Default is False.

    Example:
        import anndata as ad
        import numpy as np
        from geome.transforms import Subset

        # Create test data
        obs_data = {"cell_type": ["B cell", "T cell", "B cell", "T cell"]}
        var_data = {"gene": ["gene1", "gene2", "gene3", "gene4"]}
        adata = ad.AnnData(X=np.random.rand(4, 4), obs=obs_data, var=var_data)

        # Subset by observations
        adata_subset_obs = Subset(key_value={"cell_type": ["B cell"]}, axis="obs")(adata)
        print(adata_subset_obs)
        # View of AnnData object with n_obs x n_vars = 2 x 4
        #    obs: 'cell_type'
        #    var: 'gene'
    """

    def __init__(self, key_value: dict, axis: Literal[0, 1, "obs", "var"] = "obs", copy: bool = False):
        self.key_value = key_value
        if axis not in (0, 1, "obs", "var"):
            raise TypeError("axis needs to be one of obs, var, 0 or 1")
        if isinstance(axis, int):
            axis = ("obs", "var")[axis]
        self.axis = axis
        self.copy = copy

    def __call__(self, adata: AnnData):
        """Subset the AnnData object based on the provided key_value and axis."""
        subset_mask = self._generate_subset_mask(adata, self.axis)
        if self.axis == "obs":
            sub_adata = adata[subset_mask]
        elif self.axis == "var":
            sub_adata = adata[:, subset_mask]
        return sub_adata.copy() if self.copy else sub_adata

    def _generate_subset_mask(self, adata: AnnData, axis: Literal["obs", "var"]):
        """Generate a boolean mask for selecting observations or variables based on the specified axis."""
        data_attr = adata.obs if axis == "obs" else adata.var
        subset_mask = np.ones(data_attr.shape[0], dtype=bool)
        for key, values in self.key_value.items():
            subset_mask &= data_attr[key].isin(values)
        return subset_mask
