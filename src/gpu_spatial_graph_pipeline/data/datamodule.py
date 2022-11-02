import pytorch_lightning as pl
from typing import Callable, Literal, Optional, Sequence, Union
from torch_geometric.loader import NeighborLoader, DataListLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RandomNodeSplit
from anndata import AnnData

VALID_STAGE = {"fit", "test", None}
VALID_SPLIT = {"node", "graph"}


class GraphAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata: AnnData = None,
        adata2data_fn: Callable[[AnnData], Union[Sequence[Data], Batch]] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        learning_type: Literal["node", "graph"] = "node",
    ):
        """Manages loading and sampling schemes before loading to GPU.

        Args:
            adata (AnnData, optional): _description_. Defaults to None.
            adata2data_fn (Callable[[AnnData], Union[Sequence[Data], Batch]], optional): _description_. Defaults to None.
            batch_size (int, optional): _description_. Defaults to 1.
            num_workers (int, optional): _description_. Defaults to 1.
            learning_type (Literal[&quot;node&quot;, &quot;graph&quot;], optional):
                If graph is selected batch_size means number of graphs and the adata2data_fn
                is expected to give a list of Data.
                If node is selected batch_size means the number of nodes
                and adata2data_fn is
                expected to give a list of Data objects
                with edge_index attribute.

                 Defaults to "nodewise".


        Raises:
            ValueError: _description_
        """
        # TODO: Fill the docstring

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
            self._train_dataloader = self._spatial_node_loader(
                input_nodes=self.data.train_mask, shuffle=True
            )
            self._val_dataloader = self._spatial_node_loader(
                input_nodes=self.data.val_mask,
            )

        if stage == "test" or stage is None:
            self._test_dataloader = self._spatial_node_loader(
                input_nodes=self.data.test_mask,
            )

    def _graphwise_setup(self, stage: Optional[str]):

        num_val = int(len(self.data) * 0.05 + 1)
        num_test = int(len(self.data) * 0.01 + 1)

        if stage == "fit" or stage is None:
            self._train_dataloader = self._graph_loader(
                data=self.data[num_val + num_test :],
                shuffle=True,
            )
            self._val_dataloader = self._graph_loader(data=self.data[:num_val])
        if stage == "test" or stage is None:
            self._test_dataloader = self._graph_loader(
                data=self.data[num_val : num_val + num_test]
            )

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

    def _graph_loader(self, data, shuffle=False, **kwargs):
        """Loads from the list of data

        Args:
            shuffle (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        return DataListLoader(
            dataset=data,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kwargs
        )

    def _spatial_node_loader(self, input_nodes, shuffle=False, **kwargs):
        return NeighborLoader(
            self.data,
            num_neighbors=[-1],
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
            num_workers=self.num_workers,
            **kwargs
        )
