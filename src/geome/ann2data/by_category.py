from __future__ import annotations

from typing import Callable

from anndata import AnnData

from geome.iterables import ToCategoryIterator

from .basic import Ann2DataBasic


class Ann2DataByCategory(Ann2DataBasic):
    """Convert anndata object into a dictionary of torch.tensors then create pyg.Data from them."""

    def __init__(
        self,
        fields: dict[str, list[str]],
        category: str,
        axis: int | str = "obs",
        preserve_categories: bool | list[str] = True,
        preprocess: list[Callable[[AnnData], AnnData]] | None = None,
        transform: list[Callable[[AnnData], AnnData]] | None = None,
    ):
        """Initializes the class.

        Args:
        ----
        fields : Dictionary of fields and their associated attributes.
        category : Column in `obs` containing the categories.
        fields: Dictionary that maps field names to their locations in the AnnData object.
        adj_matrix_loc: Location of the adjacency matrix within the AnnData object.
        adata2iter: Optional function that converts AnnData objects to iterable.
                    If not given will assume that an iterable is already provided.
        preprocess: List of functions to preprocess the AnnData object before conversion.
                    A default preprocessing step (AddAdjMatrix) is added if adj_matrix_loc is provided.
        transform: List of functions to transform the AnnData object after preprocessing.
        edge_index_key: Key for the edge index in the converted data. Defaults to 'edge_index'.
        category: key for the category in the AnnData object to iterate over.
        axis: axis to iterate over. Can be 'obs' or 'var'.
        preserve_categories: If True, preserves the categories in the resulting AnnData obs and var Series.
                            If a list is provided, only the categories in the list will be preserved.
        """
        super().__init__(
            fields=fields,
            adata2iter=ToCategoryIterator(category, axis=axis, preserve_categories=preserve_categories),
            preprocess=preprocess,
            transform=transform,
        )
