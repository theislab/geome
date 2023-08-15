from typing import Callable, Optional

from anndata import AnnData

from geome.iterables import ToCategoryIterable

from .default import AnnData2DataDefault


class AnnData2DataByCategory(AnnData2DataDefault):
    def __init__(
        self,
        fields: dict[str, list[str]],
        category: str,
        preprocess: Optional[list[Callable[[AnnData], AnnData]]] = None,
        transform: Optional[list[Callable[[AnnData], AnnData]]] = None,
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
        """
        super().__init__(
            fields=fields,
            adata2iter=ToCategoryIterable(category, axis='obs'),
            preprocess=preprocess,
            transform=transform,
        )
