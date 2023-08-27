from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from geome.utils import get_from_loc

from .transform import Transform


class ToArray(Transform):
    def __init__(self, fields: dict[str, list[str]]):
        self.fields = fields

    def __call__(self, adata: AnnData) -> AnnData:
        """Store processed data in `obsm` field of `adata`.

        Store the new address for each processed data in `uns` field of `adata`.

        Parameters
        ----------
        adata : AnnData
            AnnData object.
        fields : Dict[str, List[str]]
            Dictionary of fields and their associated attributes.

        Returns
        -------
        Processed adata
        """
        adata.uns["processed_index"] = {}

        for _, addresses in self.fields.items():
            for address in addresses:
                last_attr = address.split("/")[-1]
                save_name = last_attr + "_processed"
                # TODO: Is adding a suffix a good idea?

                obj = get_from_loc(adata, address)

                # if obj is categorical
                if obj.dtype.name == "category":
                    adata.obsm[save_name] = pd.get_dummies(obj).to_numpy()
                    adata.uns["processed_index"][address] = "obsm/" + save_name
                elif not np.issubdtype(obj.dtype, np.number):
                    adata.obsm[save_name] = obj.astype(np.float)
                    adata.uns["processed_index"][address] = "obsm/" + save_name
                elif sparse.issparse(obj):
                    adata.obsm[save_name] = np.array(obj.todense())
                    adata.uns["processed_index"][address] = "obsm/" + save_name

                # If no storing required
                else:
                    adata.uns["processed_index"][address] = address

        return adata
