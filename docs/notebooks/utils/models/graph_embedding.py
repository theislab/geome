"""
Graph VAE module
g(A,X) from the paper
"""
import torch
from geome.models.modules.graph_ae import GraphAE
#from models.modules.graph_ae import GraphAE
import pytorch_lightning as pl
from torch_geometric.data import Batch


class GraphEmbedding(pl.LightningModule):
    def __init__(self,
                 num_features: int = 36,
                 latent_dim: int = 30,
                 **kwargs
                 ):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(kwargs)

        self.model = GraphAE(num_features, num_features, latent_dim)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        return x

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),
                                 lr=self.hparams["lr"],
                                 weight_decay=self.hparams["weight_decay"])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 'min', factor=0.2, patience=20, min_lr=5e-5)
        optim = {"optimizer": optim,
                 "lr_scheduler": sch, "monitor": "train_loss"}
        return optim

    def general_step(self, batch, batch_idx, mode):
        batch_size = 1
        if type(batch) == list:
            batch_size = len(batch)
            batch = Batch.from_data_list(batch)
        x = self.forward(batch)
        recon_loss = self.loss_fn(x, batch.x) / batch_size
        return recon_loss, batch_size

    def training_step(self, data_list, batch_idx):
        loss, batch_size = self.general_step(data_list, batch_idx, "train")
        self.log('train_loss', loss, batch_size=batch_size)
        return loss

    def validation_step(self, data_list, batch_idx):
        loss, batch_size = self.general_step(data_list, batch_idx, "val")
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True)

    def test_step(self, data_list, batch_idx):
        loss, batch_size = self.general_step(data_list, batch_idx, "test")
        self.log('test_loss', loss, batch_size=batch_size, prog_bar=True)