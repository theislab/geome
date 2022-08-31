import pytorch_lightning as pl
from typing import Callable, Optional, Sequence
from torch_geometric.loader import RandomNodeSampler, DataLoader
from torch_geometric.data import Data
from anndata import AnnData


class GraphAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: AnnData = None,
        adata2data_fn: Callable[[AnnData], Sequence[Data]] = None,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.adata2data_fn = adata2data_fn
        self.adata = adata

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement each case
        # TODO: Splitting
        stage = "train" if not stage else stage

        self.data = self.adata2data_fn(self.adata)

        if len(self.data) == 1:
            self.data = self.data[0]
            self._train_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )
            self._val_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )
            self._test_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )
            self._predict_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )

        else:
            if stage == "train":
                self._train_dataloader = DataLoader(
                    self.data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
            else:
                self._val_dataloader = DataLoader(
                    self.data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                self._test_dataloader = DataLoader(
                    self.data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                self._predict_dataloader = DataLoader(
                    self.data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def predict_dataloader(self):
        return self._predict_dataloader


# TODO: Param check in __init__
