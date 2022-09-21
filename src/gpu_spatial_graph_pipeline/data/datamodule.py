import pytorch_lightning as pl
from typing import Callable, Optional, Sequence, Union
from torch_geometric.loader import NeighborLoader, RandomNodeSampler
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RandomNodeSplit  # AddTrainValTestMask
from anndata import AnnData

VALID_STAGE = {"fit", "test", None}
VALID_SPLIT = {"nodewise", "graphwise"}


class GraphAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: AnnData = None,
        adata2data_fn: Callable[[AnnData], Union[Sequence[Data], Batch]] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        learning_type: Optional[str] = "nodewise",
    ):

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.adata2data_fn = adata2data_fn
        self.adata = adata
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        self.learning_type = learning_type

    def _nodewise_setup(self, stage: Optional[str]):

        self.data = Batch.from_data_list(self.data)


        self.transform = RandomNodeSplit(
            split="train_rest",
            num_val=max(int(self.data.num_nodes * 0.1), 1),
            num_test=max(int(self.data.num_nodes * 0.05), 1),
        )

        self.data = self.transform(self.data)

        if stage == "fit" or stage is None:
            self._train_dataloader = NeighborLoader(
                self.data,
                num_neighbors=[-1],
                batch_size=self.batch_size,
                input_nodes=self.data.train_mask,
                shuffle=True,
                num_workers=self.num_workers
            )
            self._val_dataloader = NeighborLoader(
                self.data,
                num_neighbors=[-1],
                batch_size=self.batch_size,
                input_nodes=self.data.val_mask,
                shuffle=False,
                num_workers=self.num_workers
            )

        if stage == "test" or stage is None:
            self._test_dataloader = NeighborLoader(
                self.data,
                num_neighbors=[-1],
                batch_size=self.batch_size,
                input_nodes=self.data.test_mask,
                shuffle=False,
                num_workers=self.num_workers
            )

    def _graphwise_setup(self, stage: Optional[str]):
        # self.data.shuffle()

        self.data_old=self.data
        self.data=[]

        for data in self.data_old:
            data = Data(
                edge_index=data.edge_index,
                y=data.y,
                x=data.x,
                Xd=data.Xd,
                batch_size=self.batch_size
            )
        
            self.data.append(data)
        num_val = int(len(self.data) * 0.05 + 1)
        num_test = int(len(self.data) * 0.01 + 1)

        if stage == "fit" or stage is None:
            self._val_dataloader = self.data[:num_val]
            self._train_dataloader = self.data[num_val + num_test :]
        if stage == "test" or stage is None:
            self._test_dataloader = self.data[num_val : num_val + num_test]

    def setup(self, stage: Optional[str] = None):
        # TODO: Implement each case
        # TODO: Splitting
        # stage = "train" if not stage else stage

        self.data = self.adata2data_fn(self.adata)
        
        if stage not in VALID_STAGE:
            raise ValueError("Stage must be one of %r." % VALID_STAGE)

        if self.learning_type == "graphwise":
            if len(self.data) <= 3:
                raise RuntimeError("Not enough graphs in data to do graphwise learning")
            self._graphwise_setup(stage)

        else:
            self._nodewise_setup(stage)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


# TODO: Param check in __init__
