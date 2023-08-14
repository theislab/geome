from typing import Any, List

import numpy as np
import pandas as pd
import squidpy as sq


def categorize_obs(adata: Any, obs_list: List[str]) -> None:
    """Converts the given list of observation columns in the AnnData object to categorical.

    Args:
    ----
    adata: The AnnData object.
    obs_list (list[str]): The list of observation columns to convert to categorical.
    """
    for cat in obs_list:
        adata.obs[cat] = adata.obs[cat].astype("category")
