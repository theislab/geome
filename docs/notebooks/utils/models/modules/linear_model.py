"""
Linear baseline model.
"""
import torch
import torch.nn as nn

# Linear NCEM spatial and nonspatial models as defined in https://www.biorxiv.org/content/10.1101/2021.07.11.451750v1


class LinearNonSpatial(nn.Module):
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
    def __init__(self, in_channels=(9, 2), out_channels=36):
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


    def forward(self, x):
        """
        Inputs:
            x - Input features per node
        """

        return self.linear(x)
