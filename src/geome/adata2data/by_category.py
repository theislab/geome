from typing import Any, Callable, Union, Optional
from geome.iterables import ToCategoryIterable

from .default import AnnData2DataDefault
from anndata import AnnData


class AnnData2DataByCategory(AnnData2DataDefault):
    """A class to transform AnnData objects into Data objects by category.

    Parameters
    ----------
    fields : dict
        Dictionary of fields and their associated attributes.
    category : str
        Column in `obs` containing the categories.
    preprocess : callable, optional
        Function to preprocess data before transformation.
    transform: callable, optional
        Function to apply to each iterated anndata object
    yields_edge_index : bool, default=True
        Whether to yield edge index.

    Attributes
    ----------
    adata_iter : callable
        Function to iterate over `adata` by category.
    """

    def __init__(
        self,
        fields: dict[str, list[str]],
        category: str,
        adj_matrix_loc: str | None,
        preprocess: Optional[list[Callable[[AnnData], AnnData]]] = None,
        transform: Optional[list[Callable[[AnnData], AnnData]]] = None,
        edge_index_key: Optional[str] = 'edge_index',
        edge_weight_key: Optional[str] = None
    ):
        """Initializes the class.

        Parameters
        ----------
        fields : dict
            Dictionary of fields and their associated attributes.
        category : str
            Column in `obs` containing the categories.
        preprocess : callable, optional
            Function to preprocess data before transformation.
        yields_edge_index : bool, default=True
            Whether to yield edge index.
        """
        super().__init__(
            fields=fields,
            adata2iter=ToCategoryIterable(category, axis='obs'),
            preprocess=preprocess,
            transform=transform,
            adj_matrix_loc=adj_matrix_loc,
            edge_index_key=edge_index_key,
            edge_weight_key=edge_weight_key
        )
