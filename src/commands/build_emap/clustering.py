import numpy as np
import torch
from abc import ABC, abstractmethod
from src.helpers.torch_utils import create_similarity_graph, create_knn_graph
from sknetwork.clustering import Leiden, Louvain
from sknetwork.data import from_edge_list
from scipy.sparse import csr_matrix
from torch import Tensor
from scipy.sparse import csr_matrix
from typing import Literal, Union, Dict, Type, Tuple, Callable, Optional

COMMUNITY_DETECTION_ALG = Literal['leiden', 'louvain']
GRAPH_TYPE = Literal['similarity', 'nn']
MODULARITY = Literal['Dugue', 'Newman', 'Potts']

class ClusteringMethod(ABC):

    def __init__(self, min_cluster_size : int) -> None:
        super().__init__()
        self.min_cluster_size = min_cluster_size

    @abstractmethod
    def base_cluster(self, X : Tensor) -> Tensor:
        ...

    def cluster(self, X : Tensor) -> Tensor:

        clusters = self.base_cluster(X)

        uniques, counts = clusters.unique(return_counts=True)
        discarded = uniques[counts < self.min_cluster_size]
        clusters[torch.isin(clusters, discarded)] = -1

        retained = uniques[counts >= self.min_cluster_size].tolist()

        mapping = { 
            v : i
            for i, v in enumerate(retained)
        }

        mapping[-1] = -1

        clusters = torch.tensor([mapping[i] for i in clusters.tolist()]).long()

        return clusters

class CommunityDetectionBasedClustering(ClusteringMethod):

    __ALG_DICT__ : Dict[COMMUNITY_DETECTION_ALG, Union[Type[Leiden], Type[Louvain]]] = {
        "leiden" : Leiden,
        "louvain" : Louvain
    }

    __GRAPH_TYPE_DICT__ : Dict[GRAPH_TYPE, Callable] = {
        "similarity" : create_similarity_graph,
        "nn" : create_knn_graph
    }

    def __init__(self,
        alg : COMMUNITY_DETECTION_ALG,
        graph_type : GRAPH_TYPE,
        threshold_or_num_neighbors : Union[int, float],
        modularity : MODULARITY,
        resolution : float,
        weighted : bool,
        min_cluster_size : int
    ) -> None:
        super().__init__(min_cluster_size=min_cluster_size)

        self.alg : COMMUNITY_DETECTION_ALG = alg
        self.graph_type : GRAPH_TYPE = graph_type
        self.threshold_or_num_neighbors = threshold_or_num_neighbors
        self.modularity : MODULARITY = modularity
        self.resolution = resolution
        self.weighted = weighted

        self._graph_fn = self.__GRAPH_TYPE_DICT__[self.graph_type]

        self._graph : Optional[Tuple[Tensor, Tensor]] = None
        self._instance : Optional[Union[Leiden, Louvain]] = None
        self._probas : Optional[Tensor] = None

    @property
    def algorithm(self) -> Union[Type[Leiden], Type[Louvain]]:
        return self.__ALG_DICT__[self.alg]

    def create_graph(self, X : Tensor, batch_size : int = 1024) -> Tuple[Tensor, Tensor]:

        return self._graph_fn(
            X, 
            self.threshold_or_num_neighbors, 
            batch_size
        )

    def detect_communities(self,
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
    ) -> Tuple[Tensor, Leiden | Louvain]:

        edge_index = edge_index.detach().cpu().numpy()
        edge_weight = edge_weight.detach().cpu().numpy()

        if not self.weighted:
            edge_weight = np.ones_like(edge_weight)

        # Build sparse adjacency matrix
        _data = (edge_weight, (edge_index[0], edge_index[1]))

        adjacency = csr_matrix(
            _data,
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )

        # Make graph undirected
        adjacency = adjacency.maximum(adjacency.T)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()

        # Leiden
        alg = self.algorithm(
            resolution=self.resolution,
            modularity=self.modularity
        )

        probas = alg.fit_predict(adjacency)

        return torch.from_numpy(probas), alg

    def base_cluster(self, X : Tensor) -> Tensor:

        X = X / X.norm(p=2, dim=-1, keepdim=True)
        edge_index, edge_weight = self.create_graph(X=X, batch_size=1024)

        probas, _instance = self.detect_communities(
            edge_index=edge_index, 
            edge_weight=edge_weight, 
            num_nodes=X.size(0)
        )

        self._instance = _instance
        self._probas = probas
        self._graph = (edge_index, edge_weight)

        return probas# .argmax(dim=-1)
