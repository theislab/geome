import anndata as ad
import numpy as np

from geome import transforms


def test_categorize():
    adata = ad.AnnData(np.random.rand(50, 2), obs={"cell_type": ["a"] * 25 + ["b"] * 25})
    # has dtype('O')
    assert adata.obs["cell_type"].dtype == "object"
    adata = transforms.Categorize(keys="cell_type")(adata)
    # has dtype('category')
    assert adata.obs["cell_type"].dtype == "category"
    # check if it does fail when called again
    adata = transforms.Categorize(keys="cell_type")(adata)


def test_categorize_list_arg():
    adata = ad.AnnData(np.random.rand(50, 2), obs={"cell_type": ["a"] * 25 + ["b"] * 25, "image_id": list("cd" * 25)})
    # has dtype('O')
    assert adata.obs["cell_type"].dtype == "object"
    assert adata.obs["image_id"].dtype == "object"
    adata = transforms.Categorize(keys=["cell_type", "image_id"])(adata)
    # has dtype('category')
    assert adata.obs["cell_type"].dtype == "category"
    assert adata.obs["image_id"].dtype == "category"
    # check if it does fail when called again
    adata = transforms.Categorize(keys=["cell_type", "image_id"])(adata)
