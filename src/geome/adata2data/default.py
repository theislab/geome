from typing import Any, Callable, Dict, List, Optional

import numpy as np
from anndata import AnnData

from geome.transforms import Compose, AddAdjMatrix, AddEdgeIndex, AddEdgeWeight
from geome.utils import get_from_loc

from .anndata2data import AnnData2Data


class AnnData2DataDefault(AnnData2Data):
    """
    Default converter for transforming AnnData objects into PyTorch Geometric Data objects.

    This class provides a standard implementation for converting AnnData objects into PyTorch Geometric Data objects.
    It assumes certain default structures and attributes within the AnnData object.
    """

    def __init__(
        self,
        fields: Dict[str, List[str]],
        adj_matrix_loc: Optional[str],
        adata2iter: Optional[Callable[[AnnData], AnnData]] = None,
        preprocess: Optional[List[Callable[[AnnData], AnnData]]] = None,
        transform: Optional[List[Callable[[AnnData], AnnData]]] = None,
        edge_index_key: Optional[str] = 'edge_index',
        edge_weight_key: Optional[str] = None
    ) -> None:
        """
        Initialize the default converter.

        Args:
            fields (Dict[str, List[str]]): Dictionary mapping field names to addresses in the AnnData object.
            adj_matrix_loc (str, optional): Location of the adjacency matrix in the AnnData object.
            adata2iter (Callable, optional): Function returning an iterable of AnnData objects.
            preprocess (List[Callable], optional): List of functions for preprocessing the AnnData object.
            transform (List[Callable], optional): List of functions for transforming the AnnData object.
            edge_index_key (str, optional): Key for storing the edge index.
            edge_weight_key (str, optional): Key for storing the edge weight.
        """
        super().__init__(fields, adata2iter, preprocess, transform)
        preprocess_list = [AddAdjMatrix(location=adj_matrix_loc, overwrite=False)] if adj_matrix_loc else []
        transform_list = [AddEdgeIndex(adj_matrix_loc, edge_index_key, overwrite=False)] if edge_index_key else []
        if edge_weight_key:
            transform_list.append(AddEdgeWeight(adj_matrix_loc, edge_weight_key, overwrite=False))
        self._preprocess = Compose(preprocess_list) if preprocess_list else None
        self._transform = Compose(transform_list) if transform_list else None
        self._adata2iter = adata2iter

    def array_from_address(
        self, adata: AnnData, location: str
    ) -> np.ndarray:
        """
        Retrieve the processed array based on the provided location.

        Args:
            adata (AnnData): The AnnData object containing the data.
            location (str): Location of the field in the AnnData object.

        Returns:
            np.ndarray: Processed array.
        """
        processed_location = adata.uns["processed_index"][location]
        return get_from_loc(adata, processed_location)
