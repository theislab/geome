import squidpy as sq
import torch
import pandas as pd
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data
from anndata import AnnData
import scipy
import torch.nn as nn


def adata2data(adata: AnnData, feature_names) -> Union[Data, Sequence[Data]]:
    """Function that takes in input an anndata object and returns a Pytorch Geometric (PyG) data object or a sequence thereof.
    Each data object represents a graph of an image stored in the anndata object.


    :param adata: Anndata object storing the images to be trained on
    :type adata: AnnData
    :param feature_names: The feature names to be used for training, extracted from anndata.obs
    :type feature_names: tuple
    :return: PyG data object or sequence thereof if more than one image is stored in the anndata object
    :rtype: Union[Data, Sequence[Data]]
    """

    dataset = []

    # Set cases for when one or more images are to be extracted from anndata
    if "library_id" in adata.obs.keys():
        library_ids = [library_id for library_id in adata.uns["spatial"].keys()]
        lib_indices = adata.obs["library_id"] == library_ids[0]
        for i in range(len(library_ids) - 1):
            lib_indices = pd.concat(
                [lib_indices, adata.obs["library_id"] == library_ids[i + 1]], axis=1
            )
        lib_indices.columns = library_ids
    else:
        lib_indices = pd.DataFrame(
            data=range(len(adata.obs)), columns=[""]
        )  # [range(len(adata.obs))]
    for library_id in lib_indices.columns:
        spatial_connectivities, _ = sq.gr.spatial_neighbors(
            adata[lib_indices[library_id]],
            coord_type="generic",
            key_added=library_id + "spatial",
            copy=True,
        )
        nodes1, nodes2 = spatial_connectivities.nonzero()
        edge_index = torch.vstack(
            [
                torch.from_numpy(nodes1).to(torch.long),
                torch.from_numpy(nodes2).to(torch.long),
            ]
        )
        if len(feature_names) > 1:
            cell_type = torch.from_numpy(
                pd.get_dummies(
                    adata.obs[feature_names[0]][lib_indices[library_id]]
                ).to_numpy()
            )
            domain = torch.from_numpy(
                pd.get_dummies(
                    adata.obs[feature_names[1]][lib_indices[library_id]]
                ).to_numpy()
            )
            features = torch.cat([cell_type, domain], dim=1)
        else:
            features = torch.from_numpy(
                pd.get_dummies(adata.obs[feature_names]).to_numpy()
            )

        X = adata.X[lib_indices[library_id]]
        if scipy.sparse.issparse(X):
            coo = X.tocoo()
            values = coo.data
            indices = np.vstack((coo.row, coo.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = coo.shape
            gene_expression = torch.sparse.FloatTensor(
                i, v, torch.Size(shape)
            ).to_dense()
        else:
            gene_expression = torch.from_numpy(adata.X)

        data = Data(
            edge_index=edge_index,
            y=gene_expression,
            x=features,
        )
        dataset.append(data)
    return dataset


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0.01)
