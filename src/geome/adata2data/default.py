from typing import Any, Callable, Dict, List, Union

import numpy as np
from anndata import AnnData
from scipy import sparse

from geome.transforms import Compose, ToArray
from geome.utils import get_adjacency_from_adata, get_from_address

from .anndata2data import AnnData2Data


class AnnData2DataDefault(AnnData2Data):
    def __init__(
        self,
        fields: Dict[str, List[str]],
        adata2iter: Union[None, Callable[[AnnData], AnnData]] = None,
        preprocess: Union[None, List[Callable[[AnnData], AnnData]]] = None,
        transform: Union[None, List[Callable[[AnnData], AnnData]]] = None,
        yields_edge_index: bool = True,
    ) -> None:
        """Convert anndata object into a dictionary of arrays.

        Assumes adata.obsp["adjacency_matrix_connectivities"] exists
        if not it is computed with sq.gr.spatial_neighbors.
        Works for squidpy datasets and the datasets of this package.

        Example for fields argument:
            fields = {
                'features': ['obs/Cluster', 'obs/donor'],
                'labels': ['X']
            }

        Args:
        ----
        fields: A dictionary of field names and their addresses in the AnnData object.
        adata_iter: An iterator function that returns an AnnData object.
        preprocess: A list of functions to preprocess the input data.
            This class by default adds a preprocessing step.
            See the static method default_preprocess.
        yields_edge_index: Whether to return the edge index of the adjacency matrix.
        """
        super().__init__(fields, adata2iter, preprocess, transform)
        # Add preprocessing of the addresses to last.
        # So that get_as_array works properly.
        preprocess_list = [ToArray(fields)]
        if preprocess is not None:
            preprocess_list = [preprocess, *preprocess_list]
        self._preprocess = Compose(preprocess_list)
        self._transform = transform
        self._adata2iter = adata2iter
        self.yields_edge_index = yields_edge_index

    def get_adj_matrix(self, adata: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Helper function to create an adjacency matrix depending on the anndata object.

        Args:
        ----
        adata: An AnnData object.
        args: Additional arguments passed to the function get_adjacency_from_adata.
        kwargs: Additional keyword arguments passed to the function get_adjacency_from_adata.

        Returns:
        -------
            The adjacency matrix.
        """
        # Get adjacency matrices
        return get_adjacency_from_adata(adata, *args, **kwargs)

    def array_from_address(
        self, adata: Any, address: str
    ) -> Union[np.ndarray, sparse.spmatrix]:
        """Return the array corresponding to the given address.

        This version assumes the addresses are stored on adata.uns.

        Args:
        ----
        adata: An AnnData object.
        address: The address of the field in the anndata object.

        Returns:
        -------
            A numpy array or a sparse matrix.
        """
        processed_address = adata.uns["processed_index"][address]
        return get_from_address(adata, processed_address)
