from __future__ import annotations

from dataclasses import dataclass, field

import squidpy as sq
from anndata import AnnData

from .base.transform import Transform


@dataclass
class AddAdjMatrix(Transform):
    """Add the adjacency matrix to the AnnData object.

    Args:
    ----
    spatial_key (str): Key in anndata.AnnData.obsm where spatial coordinates are stored.
    key_added (str): The key to add the adjacency matrix to.
    func_args (dict): Additional arguments to pass to the `spatial_neighbors` function.

    Calls the `spatial_neighbors` function from `squidpy` to calculate the spatial connectivities matrix internally.
    See `here <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html>`_ for more information on the additional arguments.

    Modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_connectivities']`` - the spatial connectivities.
        - :attr:`anndata.AnnData.obsp` ``['{{key_added}}_distances']`` - the spatial distances.
        - :attr:`anndata.AnnData.uns`  ``['{{key_added}}']`` - :class:`dict` containing parameters.

    Attributes
    ----------
        spatial_key (str): The location where the adjacency matrix will be added.

    Methods
    -------
        __call__(adata: AnnData) -> AnnData:
            Adds the spatial connectivities matrix to the given location in the AnnData object.
    """

    spatial_key: str
    key_added: str
    func_args: dict = field(default_factory=dict)

    def __call__(self, adata: AnnData) -> AnnData:  # noqa: D102
        sq.gr.spatial_neighbors(
            adata,
            key_added=self.key_added,
            spatial_key=self.spatial_key,
            **self.func_args,
        )
        return adata
