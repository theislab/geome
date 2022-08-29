import pytorch_lightning as pl
from typing import Callable, Optional
from torch_geometric.loader import RandomNodeSampler, DataLoader
from torch_geometric.data import Data
from anndata import AnnData


class NodeDataModule(pl.LightningDataModule):

    def __init__(
        self,
        adata: AnnData = None,
        feature_name: str=None,
        adata2data_fn: Callable[[AnnData], Data] = None,
        batch_size: int = 1,
        num_workers: int = 1
    
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.adata2data_fn = adata2data_fn
        self.adata = adata
        self.feature_name = feature_name

    def setup(self, stage: Optional[str] = None):
        # TODO: implement other stages
        self.data = self.adata2data_fn(self.adata, self.feature_name)

        if len(self.data)==1:
            self.data= self.data[0]
            self.dataloader=RandomNodeSampler(self.data, num_parts=self.batch_size,
                                 num_workers=self.num_workers
                                 )
        else:
            self.dataloader=DataLoader(self.data, batch_size=self.batch_size, shuffle=True,
                                 num_workers=self.num_workers
                                 ) 


    def train_dataloader(self):
        
        return self.dataloader

    def val_dataloader(self):
        return self.dataloader

    def test_dataloader(self):
        return self.dataloader

    def predict_dataloader(self):
        return self.dataloader

# TODO: Param check in __init__
