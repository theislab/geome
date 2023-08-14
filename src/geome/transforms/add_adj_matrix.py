
from anndata import AnnData

from geome.utils import get_adjacency_from_adata


class AddDesignMatrix:
    def __init__(self, xl_name: str, xc_name: str, output_name: str):
        self.xl_name = xl_name
        self.xc_name = xc_name
        self.output_name = output_name

    def __call__(self, adata: AnnData) -> AnnData:
        """Will add the spatial connectivities matrix to adata.obsp["adjacency_matrix_connectivities"].

        Args:
        ----
        adata: The AnnData object.

        """
        if "adjacency_matrix_connectivities" not in adata.obsp.keys():
            adata.obsp["adjacency_matrix_connectivities"] = get_adjacency_from_adata(adata)
        return adata