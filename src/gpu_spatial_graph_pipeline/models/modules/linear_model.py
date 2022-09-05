"""
Linear baseline model.
"""
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
from torch_geometric.utils import to_dense_adj

# Linear NCEM spatial and nonspatial models as defined in https://www.biorxiv.org/content/10.1101/2021.07.11.451750v1


class LinearNonspatial(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Inputs:
            in_channels - Dimension of input features
            out_channels - Dimension of the output features.
        """
        super().__init__()

        if not isinstance(in_channels, int):
            in_channels = in_channels[0] + in_channels[1]

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        Inputs:
            x - Input features per node
        """

        return self.linear(x)


class LinearSpatial(nn.Module):
    def __init__(self, in_channels=(8, 3), out_channels=36):
        """
        Inputs:
            in_channels - Dimension of input features. (number of cell types and domains e.g. image or patient)
            out_channels - Dimension of the output features. (number of genes)
        """
        super().__init__()

        self.mult_features = True
        self.num_genes = out_channels

        # Consider separate cases where cell type is the only feature vs when there are more features (e.g. domain)
        if isinstance(in_channels, int):
            self.mult_features = False
            self.num_cell_types = in_channels
            self.linear = nn.Linear(self.num_cell_types, self.num_genes)
        else:
            self.num_cell_types = in_channels[0]
            self.num_domains = in_channels[1]
            self.num_features = in_channels[0] + in_channels[0] ** 2 + in_channels[1]
            self.linear = nn.Linear(self.num_features, self.num_genes)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - input edge indices. Shape 2 x 2*(No. edges)
        """

        X_cell_type = x[:, 0 : self.num_cell_types + 1]

        if self.mult_features:
            X_domain = x[:, self.num_cell_types + 1 :]

        num_obs = x.shape[0]

        # Define adjacency matrix
        adj_matrix = torch.eye(num_obs) + torch.squeeze(to_dense_adj(edge_index))  # NxN

        # Compute discrete target cell interactions
        Xs = torch.matmul(adj_matrix, X_cell_type)
        Xt = Xs > 0

        # Compute interaction matrix
        Xts = torch.empty(num_obs, self.num_cell_types**2)
        i = 0
        for col1 in range(self.num_cell_types):
            for col2 in range(self.num_cell_types):
                Xts[:, i] = x[:, col1] * Xt[:, col2]
                i += 1

        # Define design matrix
        if self.mult_features:
            Xd = torch.cat([X_cell_type, Xts, X_domain], dim=1)  # Nx(L+L^2+C)
        else:
            Xd = torch.cat([X_cell_type, Xts], dim=1)  # Nx(L+L^2)

        return self.linear(Xd)
