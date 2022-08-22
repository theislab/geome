import pytorch_lightning as pl
from typing import Callable, Optional
from torch_geometric.loader import RandomNodeSampler
from torch_geometric.data import Data
from anndata import AnnData


class NodeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        adata: AnnData = None,
        adata2data_fn: Callable[[AnnData], Data] = None,
        batch_size: int = 1,
        num_workers: int = 1
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.adata2data_fn = adata2data_fn
        self.adata = adata

    def setup(self, stage: Optional[str] = None):
        # TODO: implement other stages
        self.data = self.adata2data_fn(self.adata)

    def train_dataloader(self):
        return RandomNodeSampler(self.data, num_parts=self.batch_size,
                                 num_workers=self.num_workers
                                 )

    def val_dataloader(self):
        return RandomNodeSampler(
            self.data, num_parts=self.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return RandomNodeSampler(self.data, num_parts=self.batch_size,
                                 num_workers=self.num_workers)

    def predict_dataloader(self):
        return RandomNodeSampler(self.data, num_parts=self.batch_size,
                                 num_workers=self.num_workers)

# TODO: Param check in __init__
