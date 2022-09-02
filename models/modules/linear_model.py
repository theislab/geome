"""
Linear baseline model.
"""

import torch.nn as nn
import torch
import torch_geometric.nn as geom_nn
from torch_geometric.utils import to_dense_adj

#TODO: generalize in_channels, out_channels to work for any dataset 

class LinearNonspatial(nn.Module):

    def __init__(self, in_channels=8, out_channels=36):
        """
        Inputs:
            in_channels - Dimension of input features
            out_channels - Dimension of the output features.
        """
        super().__init__()

        self.linear=nn.Linear(in_channels, out_channels)


    def forward(self, x):
        """
        Inputs:
            x - Input features per node
        """

        return self.linear(x)

class LinearSpatial(nn.Module):

    def __init__(self, in_channels=8, out_channels=36):
        """
        Inputs:
            in_channels - Dimension of input features. (number of cell types and domains e.g. image or patient)
            out_channels - Dimension of the output features. (number of genes)
        """
        super().__init__()
        
        
        #self.linear=nn.Linear(in_channels, out_channels)

        #self.linear=geom_nn.GCNConv(in_channels, out_channels)


    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - input edge indices. Shape 2 x 2*(No. edges)
        """
        num_obs=x.shape[0]
        num_features=x.shape[1]

        adj_matrix=torch.eye(num_obs)+torch.squeeze(to_dense_adj(edge_index)) #NxN 
        Xs=torch.matmul(adj_matrix,x)
        Xt=Xs>0


        #TODO: Define number of distinct batch (domain) assignments

        
        Xts=torch.empty(num_obs,num_features**2)
        i=0
        for col1 in range(num_features):
            for col2 in range(num_features):
                Xts[:,i]=x[:,col1]*Xt[:,col2]
                i+=1

        print(Xts) #Should be NxL^2

        Xd=torch.cat([x,Xts],dim=1) #Nx(L+L^2)

        print(Xd.shape)

        #TODO: Spatial model as done in paper. Outer product implies in_channels dependent on N 
        #adj_matrix=to_dense_adj(edge_index) #NxN x: NxL
        #Xs=adj_matrix*x
        #mask=adj_matrix*x>0
        #Xs=Xs[mask] #NxL
        #Xts=torch.outer(x,Xs) #NxL^2 or NxN ?
        #Xd=torch.cat(x,Xts) #Nx(L+L^2)?
        #out=self.linear(Xd)

        return self.linear(Xd)