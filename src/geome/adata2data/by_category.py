from geome.iterables import ToCategoryIterable

from .default import AnnData2DataDefault


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
        fields: dict,
        category: str,
        preprocess=None,
        transform=None,
        yields_edge_index: bool = True,
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
            yields_edge_index=yields_edge_index,
            transform=transform
        )
