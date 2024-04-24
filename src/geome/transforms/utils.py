from geome.utils import check_loc


def check_adj_matrix_loc(location: str):
    """Checks the correctness of the location format for adjacency matrices.

    Args:
    ----
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.

    Raises
    ------
        ValueError: If the location format is incorrect.
    """
    check_loc(location)
    if location != "X":
        parts = location.split("/")
        if parts[0] not in ("obsp", "varp", "layers"):
            raise ValueError("Location attribute must be one of ('obsp', 'varp', 'X', 'layers')")
