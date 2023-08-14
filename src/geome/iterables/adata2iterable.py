from abc import ABC, abstractmethod
from typing import Union, Iterable
from anndata import AnnData

class AnnData2Iterable(ABC):

    @abstractmethod
    def __call__(self, adata: AnnData)-> Iterable[AnnData]:
        pass
