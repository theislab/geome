"""
LinearNCEM module
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from sklearn.metrics import r2_score
import numpy as np
from torch_geometric.data import Batch


class LinearNCEM(pl.LightningModule):
    def __init__(self, use_node_scale=False, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters(model_kwargs)

        self.use_node_scale = use_node_scale

        self.model_sigma = nn.Linear(
            in_features=self.hparams.in_channels,
            out_features=self.hparams.out_channels,
        )
        self.model_mu = nn.Linear(
            in_features=self.hparams.in_channels,
            out_features=self.hparams.out_channels,
        )

        self.loss_module = nn.GaussianNLLLoss(eps=1e-5)

    def forward(self, data):
        x = data.x
        mu = self.model_mu(x)
        sigma = torch.exp(self.model_sigma(x))

        # scale by sf
        if self.use_node_scale:
            sf = torch.unsqueeze(data.sf, 1)  # Nx1
            mu = sf * mu
            sigma = sf * sigma

        # clip output
        bound = 60.0
        mu = torch.clamp(mu, min=-np.exp(bound), max=np.exp(bound))
        sigma = torch.clamp(sigma, min=-bound, max=bound)
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
