import anndata as ad
import numpy as np

from geome import iterables, transforms


def test_to_category_iterator():
    adata = ad.AnnData(
        X=np.random.rand(50, 2),
        obs={"cell_type": ["a"] * 25 + ["b"] * 25, "image_id": list("cd" * 25)},
    )
    adata = transforms.Categorize(keys=["cell_type", "image_id"])(adata)
    adatas = list(iterables.ToCategoryIterator(category="cell_type")(adata))
    assert len(adatas) == 2
    # assert
    for cat_adata in adatas:
        assert len(cat_adata) == 25
        assert cat_adata.obs["cell_type"].dtype == "category"
        assert cat_adata.obs["image_id"].dtype == "category"

    # assert if category counts are correct
    assert adatas[0].obs["image_id"].value_counts().to_dict() == {"c": 13, "d": 12}
    assert adatas[1].obs["image_id"].value_counts().to_dict() == {"c": 12, "d": 13}
