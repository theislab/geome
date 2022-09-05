import pytorch_lightning as pl
from typing import Callable, Optional, Sequence
from torch_geometric.loader import RandomNodeSampler, DataLoader, DataListLoader
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit # AddTrainValTestMask
from anndata import AnnData

VALID_STAGE = {"fit", "test", None}
VALID_SPLIT = {"nodewise", "graphwise"}

class GraphAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: AnnData = None,
        feature_names: tuple = None,
        adata2data_fn: Callable[[AnnData], Sequence[Data]] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        learning_type: Optional[str] = "nodewise"
    ):  

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.adata2data_fn = adata2data_fn
        self.adata = adata
        self.feature_names=feature_names
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        self.learning_type = learning_type

    
    def nodewise_setup(self, stage: Optional[str]):

        self.data = self.data[0]
        self.transform = RandomNodeSplit(
            split = "train_rest",
            num_val=self.data.num_nodes*0.05,
            num_test=self.data.num_nodes*0.1 )

        self.data = self.transform(self.data)

        if stage == "fit" or stage is None:
            self._train_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )
            self._val_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )
        if stage == "test" or stage is None:
            self._test_dataloader = RandomNodeSampler(
                self.data, num_parts=self.batch_size,
                num_workers=self.num_workers
            )


    def graphwise_setup(self, stage: Optional[str]):
        #self.data.shuffle()

        num_val = int(len(self.data)*0.05 + 1)
        num_test = int(len(self.data)*0.01 +1)

        if stage == "fit" or stage is None:
            self._val_data_loader = self.data[:num_val]
            self._train_data_loader = self.data[num_val + num_test:]
        if stage =="test" or stage is None:
            self._test_data_loader = self.data[num_val:num_val + num_test]

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement each case
        # TODO: Splitting
        # stage = "train" if not stage else stage

        self.data = self.adata2data_fn(self.adata, self.feature_names)
        if stage not in VALID_STAGE:
            raise ValueError("Stage must be one of %r." % VALID_STAGE)

        if self.learning_type == "graphwise":
            if len(self.data) <= 3:
                raise RuntimeError("Not enough graphs in data to do graphwise learning")
            self.graphwise_setup(stage)

        else:
            if len(self.data) != 1:
                raise RuntimeError("Currently not good for nodewise learning, should be changed")
            self.nodewise_setup(stage)


    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    #def predict_dataloader(self):
     #   return self._predict_dataloader


# TODO: Param check in __init__
