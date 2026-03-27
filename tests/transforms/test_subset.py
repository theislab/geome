import anndata as ad
import numpy as np
import pytest

from geome.transforms import Subset


# copy value true or false as a fixture
@pytest.fixture(params=[True, False])
def copy(request):
    return request.param


@pytest.fixture
def adata():
    # Create test data for obs and var
    obs_data = {
        "cell_type": ["B cell", "T cell", "B cell", "T cell"],
        "condition": ["healthy", "disease", "disease", "healthy"],
    }
    var_data = {"gene": ["gene1", "gene2", "gene3", "gene4"], "chromosome": ["chr1", "chr2", "chr1", "chr2"]}
    adata = ad.AnnData(X=np.random.rand(4, 4), obs=obs_data, var=var_data)
    return adata


@pytest.mark.parametrize(
    "axis,key_value,expected_shape,expected_values",
    [
        ("obs", {"cell_type": ["B cell"]}, (2, 4), {"cell_type": "B cell"}),
        ("obs", {"condition": ["healthy"]}, (2, 4), {"condition": "healthy"}),
        ("var", {"chromosome": ["chr1"]}, (4, 2), {"chromosome": "chr1"}),
        ("var", {"gene": ["gene1", "gene3"]}, (4, 2), {"gene": ["gene1", "gene3"]}),
    ],
)
def test_subset(adata, axis, key_value, expected_shape, expected_values, copy):
    # Apply the Subset transformation
    subset_transform = Subset(key_value, axis=axis, copy=copy)
    adata_subset = subset_transform(adata)

    assert adata_subset.is_view != copy, "The subsetted AnnData object is not a view as expected."

    # Assert the expected shape
    assert adata_subset.shape == expected_shape

    # Check the integrity of the subset data
    for key, expected_value in expected_values.items():
        if axis == "var":
            assert all(adata_subset.var[key] == expected_value)
        else:
            assert all(adata_subset.obs[key] == expected_value)

    # Check how X is changed
    if axis == "obs":
        mask = adata.obs[list(key_value.keys())[0]].isin(list(key_value.values())[0])
        assert np.all(adata_subset.X == adata.X[mask])
    else:
        mask = adata.var[list(key_value.keys())[0]].isin(list(key_value.values())[0])
        assert np.all(adata_subset.X == adata.X[:, mask])


def test_subset_multiple_keys(adata):
    # Apply the Subset transformation with multiple keys
    key_value = {"cell_type": ["B cell"], "condition": ["healthy"]}
    subset_transform = Subset(key_value, axis="obs")
    adata_subset = subset_transform(adata)

    # Assert the expected shape
    assert adata_subset.shape == (1, 4)

    # Check the integrity of the subset data
    assert all(adata_subset.obs["cell_type"] == "B cell")
    assert all(adata_subset.obs["condition"] == "healthy")

    # Check how X is changed
    mask = adata.obs["cell_type"].isin(key_value["cell_type"]) & adata.obs["condition"].isin(key_value["condition"])
    assert np.all(adata_subset.X == adata.X[mask])
