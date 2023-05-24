"""
NonLinearNCEM module
"""

import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import r2_score
from .modules.gnn_model import GNNModel
from .modules.mlp_model import MLPModel
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)



class NonLinearNCEM(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(model_kwargs)

        self.encoder = GNNModel(
            in_channels=self.hparams.in_channels,
            hidden_dims=self.hparams.encoder_hidden_dims,
            out_channels=self.hparams.latent_dim,
        )

        self.decoder_sigma = MLPModel(
            in_channels=self.hparams.latent_dim,
            out_channels=self.hparams.out_channels,
            hidden_dims=self.hparams.decoder_hidden_dims,
        )
        self.decoder_mu = MLPModel(
            in_channels=self.hparams.latent_dim,
            out_channels=self.hparams.out_channels,
            hidden_dims=self.hparams.decoder_hidden_dims,
        )

        self.loss_module = nn.GaussianNLLLoss(eps=1e-5)

        self.encoder.apply(init_weights)
        self.decoder_mu.apply(init_weights)
        self.decoder_sigma.apply(init_weights)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index)
        sigma = self.decoder_sigma(x)
        mu = self.decoder_mu(x)
        return mu, sigma

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def training_step(self, batch, _):

        self.batch_size = batch.batch_size

        mu, sigma = self.forward(batch)
        loss = self.loss_module(
            mu[: self.batch_size], batch.y[: self.batch_size], sigma[: self.batch_size]
        )
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, _):

        self.batch_size = batch.batch_size

        mu, sigma = self.forward(batch)
        val_loss = self.loss_module(
            mu[: self.batch_size], batch.y[: self.batch_size], sigma[: self.batch_size]
        )
        val_r2_score = r2_score(
            batch.y.cpu()[: self.batch_size], mu.cpu()[: self.batch_size]
        )
        self.log(
            "val_r2_score", val_r2_score, prog_bar=True, batch_size=self.batch_size
        )
        self.log("val_loss", val_loss, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, batch, _):
        self.batch_size = batch.batch_size

        mu, sigma = self.forward(batch)
        test_loss = self.loss_module(
            mu[: self.batch_size], batch.y[: self.batch_size], sigma[: self.batch_size]
        )
        test_r2_score = r2_score(
            batch.y.cpu()[: self.batch_size], mu.cpu()[: self.batch_size]
        )
        self.log(
            "test_r2_score", test_r2_score, prog_bar=True, batch_size=self.batch_size
        )
        self.log("test_loss", test_loss, prog_bar=True, batch_size=self.batch_size)
