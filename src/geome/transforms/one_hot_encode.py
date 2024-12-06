from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from anndata import AnnData

from .base.transform import Transform


@dataclass
class SaveOneHotEncodeLabels(Transform):
    """One-hot encode specified columns in an AnnData object and store them in the specified matrix slot.

    Args:
    -----
    keys (Union[str, List[str]]): Columns to be one-hot encoded.
    axis (Literal["obs", "var"]): Axis on which the columns are located. 'obs' for observation, 'var' for variables.
    key_added (str): Base key under which the one-hot encoded data and label mappings will be stored.

    Methods
    -------
    __call__(adata: AnnData) -> None:
        Converts the specified columns to one-hot encoded format and updates `adata` accordingly.
    """

    keys: str | list
    axis: Literal["obs", "var"]
    key_added: str

    def __post_init__(self):
        if isinstance(self.keys, str):
            self.keys = [self.keys]

    def __call__(self, adata: AnnData) -> None:
        """
        One-hot encode the specified columns and store the result in the AnnData object.

        Parameters
        ----------
        adata : AnnData
            The annotated data matrix to be updated with one-hot encoded data.

        Returns
        -------
        None
        """
        matrix_key = f"{self.axis}m"  # e.g., 'obsm' or 'varm'
        # encoded_data = {}
        label_mappings = {}
        encoded_data_list = []

        for key in self.keys:
            # Generate one-hot encoding
            categories = pd.get_dummies(getattr(adata, self.axis)[key])
            # encoded_data[key] = categories
            encoded_data_list.append(categories)
            # Store mapping of codes to labels
            label_mappings[key] = categories.columns.tolist()

        encoded_data_combined = pd.concat(encoded_data_list, axis=1)

        # Save the encoded data and mappings in the appropriate AnnData structure
        getattr(adata, matrix_key)[self.key_added] = pd.DataFrame(encoded_data_combined)
        adata.uns[f"{self.key_added}_mappings"] = label_mappings

        print(label_mappings)

        return adata
