from typing import Literal

from anndata import AnnData

from .base.transform import Transform


class Subset(Transform):
    """Create a subset of adata on list of observation or features.

    Input:
        key_value   : dict{str(obs_name): list[str(Value1), str(Value2),...],...}
                        The dictionary with observation columns and list of values to keep for each observation.
    """

    def __init__(self, key_value: dict, axis: Literal[0, 1, "obs", "var"] = "obs"):
        self.key_value = key_value
        if axis not in (0, 1, "obs", "var"):
            raise TypeError("axis needs to be one of obs, var, 0 or 1")
        if isinstance(axis, int):
            axis = ("obs", "var")[axis]
        self.axis = axis

    def __call__(self, adata: AnnData):
        """Converts the given list of observation columns in the AnnData object to categorical.

        Args:
        ----
        adata: The AnnData object.
        obs_list (list[str]): The list of observation columns to convert to categorical.
        """
        if self.axis == "obs":
            subset_mask = self._generate_obs_subset_mask(adata)
            sub_adata = adata[subset_mask]
        elif self.axis == "var":
            subset_mask = self._generate_var_subset_mask(adata)
            sub_adata = adata[:, subset_mask]
        return sub_adata

    def _generate_obs_subset_mask(self, adata: AnnData):
        """Generate a boolean mask for selecting observations."""
        subset_mask = None
        for key, values in self.key_value.items():
            if subset_mask is None:
                subset_mask = adata.obs[key].isin(values)
            else:
                subset_mask &= adata.obs[key].isin(values)
        return subset_mask

    def _generate_var_subset_mask(self, adata: AnnData):
        """Generate a boolean mask for selecting variables."""
        subset_mask = None
        for key, values in self.key_value.items():
            if subset_mask is None:
                subset_mask = adata.var[key].isin(values)
            else:
                subset_mask &= adata.var[key].isin(values)
        return subset_mask
