import squidpy as sq
from anndata import AnnData

from geome.transforms.utils import check_adj_matrix_loc
from geome.utils import set_to_loc


class AddAdjMatrix:
    """Add the adjacency matrix to the AnnData object.

    Args:
    ----
    location (str): The location to add the adjacency matrix to.
                    Format should be 'attribute/key', e.g., 'obsp/adjacency_matrix'.
                    'X' can be used for the main matrix.
    overwrite (bool): Whether to overwrite the existing adjacency matrix.

    Attributes:
    ----------
        location (str): The location where the adjacency matrix will be added.

    Methods:
    -------
        __call__(adata: AnnData) -> AnnData:
            Adds the spatial connectivities matrix to the given location in the AnnData object.
    """

    def __init__(self, location: str, overwrite: bool = False):
        """Initialize the AddAdjMatrix class.

        Args:
        ----
        location (str): The location to add the adjacency matrix to.
        """
        check_adj_matrix_loc(location)
        self.location = location
        self.overwrite = overwrite

    def __call__(self, adata: AnnData) -> AnnData:
        """Add the spatial connectivities matrix to the given location.

        Args:
        ----
        adata (AnnData): The AnnData object.

        Returns:
        -------
            AnnData: The updated AnnData object with the added adjacency matrix.
        """
        adj_matrix = sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            key_added="spatial",
            copy=True,
        )[0]
        set_to_loc(adata, self.location, adj_matrix, self.overwrite)
        return adata
