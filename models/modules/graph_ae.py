"""
Graph VAE module
g(A,X) from the paper
"""
import torch
from torch_geometric.nn import SAGEConv


# taken from https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial6/Tutorial6.ipynb
class GraphAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super(GraphAE, self).__init__()
        self.conv1 = SAGEConv(in_channels, latent_dim * 2)
        self.conv2 = SAGEConv(latent_dim * 2, latent_dim)
        self.conv3 = SAGEConv(latent_dim, latent_dim * 2 )
        self.conv4 = SAGEConv(latent_dim * 2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index)
        return x
