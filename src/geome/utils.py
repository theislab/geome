from typing import Any

from anndata import AnnData


def get_from_loc(adata: AnnData, location: str) -> Any:
    """Get a value from a specified location in the AnnData object.

    Args:
    ----
    adata (AnnData): The AnnData object.
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.

    Returns:
    -------
        Any: The value at the specified location.

    Raises:
    ------
        KeyError: If the specified location does not exist in the AnnData object.
    """
    if location == "X":
        return adata.X
    assert len(location.split("/")) == 2, f"Location must have only one delimiter {location}"
    axis, key = location.split("/")

    if key not in getattr(adata, axis, {}):
        raise KeyError(
            f"The specified key '{key}' does not exist in the '{axis}' of the AnnData object."
        )

    return getattr(adata, axis)[key]


def set_to_loc(adata: AnnData, location: str, value: Any, override: bool = False):
    """Assign a value to a specified location in the AnnData object.

    Args:
    ----
    adata (AnnData): The AnnData object.
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.
    value (Any): The value to assign.
    override (bool, optional): If set to True, will override the existing value at the location. Defaults to False.

    Raises:
    ------
        ValueError: If the specified location already exists and override is set to False.
    """
    if location == "X":
        if not override and hasattr(adata, "X"):
            raise ValueError(
                "The location 'X' already has data. To override, set the 'override' parameter to True."
            )
        adata.X = value
    else:
        axis, key = location.split("/")

        if not override and key in getattr(adata, axis, {}):
            raise ValueError(
                f"The location '{location}' already has data. To override, set the 'override' parameter to True."
            )

        getattr(adata, axis)[key] = value


def check_loc(location: str):
    """Checks the correctness of the location format.

    Args:
    ----
    location (str): The location in the AnnData object. Format should be 'attribute/key' or 'X'.

    Raises:
    ------
        ValueError: If the location format is incorrect.
    """
    if location != "X":
        parts = location.split("/")
        if len(parts) != 2:
            raise ValueError("Location must have only one delimiter '/'")
