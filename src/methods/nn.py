import math
import faiss
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
from torch import Tensor
from sklearn.neighbors import NearestNeighbors
from src.shared.registry import registry

Array = Union[Tensor, np.ndarray]

def to_numpy(X : Array) -> np.ndarray:

    if isinstance(X, Tensor):
        X = X.detach().cpu().numpy()

    return np.ascontiguousarray(X, dtype=np.float32)


class NNRetriever(ABC):

    def update_kwargs(kwargs : Dict[str, Any], others : Dict[str, Any]) -> Dict[str, Any]:
        return {
            **kwargs,
            **others
        }

    def __init__(self,
        k : int
    ) -> None:
        self.k = k
        self.X = None
        self.fitted = False

    @abstractmethod
    def _fit(self) -> None:
        ...

    @abstractmethod
    def _search(self, Q : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def fit(self, X : Array) -> 'NNRetriever':

        self.X = to_numpy(X)

        self._fit()
        self.fitted = True

        return self

    def nn(self, Q : Optional[Array] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        k nearest neighbors (among the fitted X) of every row of Q.
        Q can be a Tensor or a numpy array; if None, Q = the fitted X (each row is its own first neighbor).

        Returns (scores, indices), both of shape (len(Q), k).
        scores are cosine similarities: higher = closer.
        """

        if not self.fitted:
            raise RuntimeError("call fit(X) before nn()")

        Q = self.X if Q is None else to_numpy(Q)

        return self._search(Q)


NNRetrieverRegistry = registry(NNRetriever)


@NNRetrieverRegistry.register("sklearn")
class SklearnNNRetriever(NNRetriever):

    def _fit(self) -> None:

        self.index = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.index.fit(self.X)

    def _search(self, Q : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        distances, indices = self.index.kneighbors(Q, n_neighbors=self.k)

        return 1.0 - distances, indices


@NNRetrieverRegistry.register("faiss")
class FaissNNRetriever(NNRetriever):
    """
    Inner product search: X should be L2-normalized so that scores are cosine similarities.
    Below exact_threshold points, an exact IndexFlatIP is used instead of the approximate IVF-PQ index.
    For IVF-PQ, D must be divisible by search_m.
    """

    exact_threshold : int = 10_000

    def __init__(self,
        k : int,
        search_nlist : Optional[int] = None,
        search_m : int = 16,
        search_nbits : int = 8,
        search_nprobe : int = 10
    ) -> None:
        super().__init__(k=k)
        self.search_nlist = search_nlist
        self.search_m = search_m
        self.search_nbits = search_nbits
        self.search_nprobe = search_nprobe

    def make_faiss_index(self, X : np.ndarray) -> 'faiss.IndexIVFPQ':

        N, D = X.shape
        search_nlist = self.search_nlist or int(math.sqrt(N))

        quantizer = faiss.IndexFlatIP(D)

        index = faiss.IndexIVFPQ(
            quantizer,
            D,
            search_nlist,
            self.search_m,
            self.search_nbits,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.train(X)
        index.add(X)
        index.nprobe = self.search_nprobe

        # keep the quantizer alive as long as the index (faiss does not own it from Python)
        self.quantizer = quantizer

        return index

    def make_exact_index(self, X : np.ndarray) -> 'faiss.IndexFlatIP':

        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)

        return index

    def is_exact(self) -> bool:
        return self.X.shape[0] < self.exact_threshold

    def _fit(self) -> None:

        if self.is_exact():
            self.index = self.make_exact_index(self.X)
        else:
            self.index = self.make_faiss_index(self.X)

    def _search(self, Q : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        scores, indices = self.index.search(Q, self.k)

        return scores, indices


if __name__ == '__main__':

    X = torch.nn.functional.normalize(torch.randn(5000, 384), dim=-1)

    for name in ["sklearn", "faiss"]:
        retriever = NNRetrieverRegistry.create(name, k=5)
        scores, indices = retriever.fit(X).nn()
        print(name, scores.shape, indices.shape, indices[0])