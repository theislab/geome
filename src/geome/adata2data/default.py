from typing import Any, Callable, Optional

import numpy as np
from anndata import AnnData

from geome.transforms import AddAdjMatrix, AddEdgeIndex, AddEdgeWeight, Compose
from geome.utils import get_from_loc

from .anndata2data import AnnData2Data


class AnnData2DataDefault(AnnData2Data):
    def __init__(
        self,
        fields: dict[str, list[str]],
        adj_matrix_loc: str | None,
        adata2iter: Optional[Callable[[AnnData], AnnData]] = None,
        preprocess: Optional[list[Callable[[AnnData], AnnData]]] = None,
        transform: Optional[list[Callable[[AnnData], AnnData]]] = None,
        edge_index_key: Optional[str] = 'edge_index',
        edge_weight_key: Optional[str] = None
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
        preprocess_list = []
        if adj_matrix_loc is not None:
            preprocess_list.append(AddAdjMatrix(location=adj_matrix_loc, overwrite=False))
        if preprocess is not None:
            preprocess_list = [preprocess, *preprocess_list]
        transform_list = []
        if edge_index_key is not None:
            transform_list.append(AddEdgeIndex(adj_matrix_loc, edge_index_key, overwrite=False))
        if edge_weight_key is not None:
            transform_list.append(AddEdgeWeight(adj_matrix_loc, edge_weight_key, overwrite=False))
        if transform is not None:
            transform_list = [*transform_list, transform]
        self._preprocess = Compose(preprocess_list)
        self._transform = Compose(transform_list)
        self._adata2iter = adata2iter


    def array_from_address(
        self, adata: Any, location: str
    ) -> np.ndarray:
        """Return the processed array corresponding to the given location.

        This version assumes the addresses are stored on adata.uns.

        Args:
        ----
        adata: An AnnData object.
        location: The location of the field in the anndata object.

        Returns:
        -------
            A numpy array.
        """
        processed_location = adata.uns["processed_index"][location]
        return get_from_loc(adata, processed_location)
