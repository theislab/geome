import pytest

import geome
import anndata as ad
import torch
import numpy as np
import squidpy as sq
from geome import transforms
from geome.anndata2data import AnnData2DataByCategory
from torch_geometric.utils.convert import from_scipy_sparse_matrix


def test_example():
    assert 1 == 1  # This test is designed to fail.

def create_test_adata():
    n_cells = 14 # only even values
    n_genes = 10

    counts = np.random.rand(n_cells, n_genes)
    coordinates =  np.random.rand(n_cells, n_genes)
    cell_types = np.random.randint(5, size=n_cells)
    #image_id = np.random.randint(2, size=n_cells)
    image_id = np.concatenate((np.zeros(int(n_cells/2)), np.ones(int(n_cells/2))))

    new_adata = ad.AnnData(counts, 
                obs={'cell_type': cell_types, 'image_id': image_id},
                obsm={"spatial": coordinates}, 
                dtype=np.int64)
    return new_adata

def load_pyg_object(adata: ad.AnnData, connectivities_key: str = None):
        
        fields = {
            'features':['obs/cell_type','obs/image_id'],
            'labels':['X']
        }
        
        category_to_iterate = 'image_id'
        
        preprocess = [
            lambda x,_: transforms.categorize_obs(x,['cell_type', 'image_id']),
        ]
        
        a2d = AnnData2DataByCategory(
            fields=fields,
            category=category_to_iterate,
            preprocess=preprocess,
            yields_edge_index=True, 
            connectivities_key=connectivities_key
        )
        
        return a2d(adata)
    
def map_values(original_list):
    # Create a mapping from unique elements to values from 0 to n
    unique_elements = list(set(original_list))
    mapping = {element: idx for idx, element in enumerate(unique_elements)}

    # Convert the original list to the list with values from 0 to n
    return [mapping[element] for element in original_list]
     
    
def test_connectivites_key():
    connectivity_key = 'adjacency'
    adata = create_test_adata()
    for obs_var in adata.obs:
         adata.obs[obs_var] = adata.obs[obs_var].astype('category')
    sq.gr.spatial_neighbors(adata, library_key = 'image_id', key_added = connectivity_key, coord_type = 'generic')
    pyg_datas = load_pyg_object(adata, connectivity_key)
    image_id = 0
    pyg_adj_m = pyg_datas[image_id].edge_index # 2d tensor object
    adata_adj_m = from_scipy_sparse_matrix(adata.obsp['adjacency_connectivities'][adata.obs['image_id']==image_id])[0]
    assert torch.equal(pyg_adj_m, adata_adj_m)
