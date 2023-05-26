import pytorch_lightning as pl
from typing import Literal, Optional, Sequence, List
from torch_geometric.loader import NeighborLoader, DataListLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import RandomNodeSplit

VALID_STAGE = {"fit", "test", None}
VALID_SPLIT = {"node", "graph"}

# TODO: Fix dataloader


class GraphAnnDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datas: Optional[Sequence[Data]] = None,
        batch_size: int = 1,
        num_workers: int = 1,
        learning_type: Literal["node", "graph"] = "node",
    ):
        """
        Manages loading and sampling schemes before loading to GPU.

        Args:
            datas (Sequence[Data], optional): The data to be loaded. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 1.
            num_workers (int, optional): The number of workers. Defaults to 1.
            learning_type (Literal["node", "graph"], optional): The type of learning to be performed.
                If "graph" is selected, `batch_size` means the number of graphs and `datas` is expected to be a list of Data.
                If "node" is selected, `batch_size` means the number of nodes and `datas` is expected to be a list of Data objects
                with an edge_index attribute. Defaults to "node".

        Raises:
            ValueError: If `learning_type` is not one of {"node", "graph"}.
        """
        # TODO: Fill the docstring

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = datas
        if learning_type not in VALID_SPLIT:
            raise ValueError("Learning type must be one of %r." % VALID_SPLIT)
        self.learning_type = learning_type
        self.first_time = True

    def _nodewise_setup(self, stage: Optional[str]) -> None:
        """
        Sets up the data loaders for node-wise learning.

        Args:
            stage (Optional[str]): The stage of training to set up the data loader for. Defaults to None.

        Returns:
            None
        """
        if self.first_time:
            self.data = Batch.from_data_list(self.data)
            self.transform = RandomNodeSplit(
                split="train_rest",
                num_val=max(int(self.data.num_nodes * 0.1), 1),
                num_test=max(int(self.data.num_nodes * 0.05), 1),
            )

            self.data = self.transform(self.data)
            self.first_time = False

        if stage == "fit" or stage is None:
            self._train_dataloader = self._spatial_node_loader(input_nodes=self.data.train_mask, shuffle=True)
            self._val_dataloader = self._spatial_node_loader(
                input_nodes=self.data.val_mask,
            )
        if stage == "test" or stage is None:
            self._test_dataloader = self._spatial_node_loader(
                input_nodes=self.data.test_mask,
            )

    def _graphwise_setup(self, stage: Optional[str]) -> None:
        """
        Sets up the data loaders for graph-wise learning.

        Args:
            stage (Optional[str]): The stage of training to set up the data loader for. Defaults to None.

        Returns:
            None
        """
        num_val = int(len(self.data) * 0.05 + 1)
        num_test = int(len(self.data) * 0.01 + 1)

        if stage == "fit" or stage is None:
            self._train_dataloader = self._graph_loader(
                data=self.data[num_val + num_test :],
                shuffle=True,
            )
            self._val_dataloader = self._graph_loader(data=self.data[:num_val])
        if stage == "test" or stage is None:
            self._test_dataloader = self._graph_loader(data=self.data[num_val : num_val + num_test])

    def setup(self, stage: Optional[str] = None):
        """
        Setup function to be called at the beginning of training, validation or testing

        Args:
            stage (str, optional): the stage of the training, either 'train', 'val' or 'test'. Defaults to None.
        """
        # TODO: Implement each case
        # TODO: Splitting
        # stage = "train" if not stage else stage

        if stage not in VALID_STAGE:
            raise ValueError("Stage must be one of %r." % VALID_STAGE)

        if self.learning_type == "graph":
            if len(self.data) <= 3:
                raise RuntimeError("Not enough graphs in data to do graph-wise learning")
            self._graphwise_setup(stage)

        else:
            self._nodewise_setup(stage)

    def train_dataloader(self):
        """
        Returns the training dataloader
        """
        return self._train_dataloader

    def val_dataloader(self):
        """
        Returns the validation dataloader
        """
        return self._val_dataloader

    def test_dataloader(self):
        """
        Returns the test dataloader
        """
        return self._test_dataloader

    def _graph_loader(self, data: List, shuffle: bool = False, **kwargs) -> DataListLoader:
        """
        Loads the data in the form of graphs

        Args:
            data (List): list of data to be loaded
            shuffle (bool, optional): whether to shuffle the data. Defaults to False.

        Returns:
            DataListLoader: the graph dataloader
        """
        return DataListLoader(
            dataset=data, shuffle=shuffle, batch_size=self.batch_size, num_workers=self.num_workers, **kwargs
        )

    def _spatial_node_loader(self, input_nodes: List, shuffle: bool = False, **kwargs) -> NeighborLoader:
        """
        Loads the data in the form of nodes

        Args:
            input_nodes (List): the input nodes
            shuffle (bool, optional): whether to shuffle the data. Defaults to False.

        Returns:
            NeighborLoader: the node dataloader
        """
        return NeighborLoader(
            self.data,
            num_neighbors=[-1],
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=shuffle,
            num_workers=self.num_workers,
            **kwargs,
        )
