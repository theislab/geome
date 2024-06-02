from typing import Any

from anndata import AnnData


def get_from_loc(adata: AnnData, location: str) -> Any:
    """Get a value from a specified location in the AnnData object.

    Args:
    ----
    adata (AnnData): The AnnData object.
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.

    Returns
    -------
        Any: The value at the specified location.

    Raises
    ------
        KeyError: If the specified location does not exist in the AnnData object.
    """
    if location == "X":
        return adata.X
    elif location == "obs_names":
        return adata.obs_names.to_numpy()
    elif location == "var_names":
        return adata.var_names.to_numpy()

    assert len(location.split("/")) == 2, f"Location must have only one delimiter {location}"
    axis, key = location.split("/")

    if key not in getattr(adata, axis, {}):
        raise KeyError(f"The specified key '{key}' does not exist in the '{axis}' of the AnnData object.")

    return getattr(adata, axis)[key]


def set_to_loc(adata: AnnData, location: str, value: Any, overwrite: bool = False):
    """Assign a value to a specified location in the AnnData object.

    Args:
    ----
    adata (AnnData): The AnnData object.
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.
    value (Any): The value to assign.
    overwrite (bool, optional): If set to True, will overwrite the existing value at the location. Defaults to False.

    Raises
    ------
        ValueError: If the specified location already exists and overwrite is set to False.
    """
    if location == "X":
        if not overwrite and hasattr(adata, "X"):
            raise ValueError("The location 'X' already has data. To overwrite, set the 'overwrite' parameter to True.")
        adata.X = value
    else:
        axis, key = location.split("/")

        if not overwrite and key in getattr(adata, axis, {}):
            raise ValueError(
                f"The location '{location}' already has data. To overwrite, set the 'overwrite' parameter to True."
            )

        getattr(adata, axis)[key] = value


def check_loc(location: str):
    """Checks the correctness of the location format.

    Args:
    ----
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.

    Raises
    ------
        ValueError: If the location format is incorrect.
    """
    if location != "X":
        parts = location.split("/")
        if len(parts) != 2:
            raise ValueError("Location must have only one delimiter '/'")
