import torch
import numpy as np
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.storage import AttrType 
from torch_geometric.typing import OptTensor
from torch_geometric import utils as gnn_utils
from typing import Optional, Any
from collections import defaultdict

class ArticlesGraph(Data):

    def __init__(self, 
        x : OptTensor = None, 
        edge_index : OptTensor = None,
        edge_type : OptTensor = None,
        y : OptTensor = None, 
    ):
        super().__init__(x, edge_index, y = y)

        self.article_id = torch.arange(x.size(0)) if x is not None else None
        self.edge_id = torch.arange(edge_index.size(-1)) if edge_index is not None else None
        self.edge_type = edge_type

        # print(self.is_node_attr('article_id'))
        # print(self.is_edge_attr('edge_id'))
        # print(self.is_edge_attr('edge_type'))

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)
    
    @property
    def num_features(self) -> int:
        return self.x.size(-1)
    
    @property
    def num_relations(self) -> int:
        return len(self.edge_type.unique())
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.size(-1)
    
    @property
    def num_labels(self) -> int:
        return self.y.size(-1)

class KnowledgeGraph(Data):

    def __init__(self, 
        x : OptTensor = None,
        virtual : OptTensor = None,
        e : OptTensor = None,
        r : OptTensor = None,
        edge_index : OptTensor = None,
        edge_type : OptTensor = None,
        y : OptTensor = None,
        entity_names : Optional[list[str]] = None,
    ) -> None:
        super().__init__(x, edge_index, e, y)

        self.entity_id = torch.arange(x.size(0)) if x is not None else None
        self.virtual = virtual
        self.triple_id = torch.arange(edge_index.size(-1)) if edge_index is not None else None
        self.r = r
        self.edge_type = edge_type
        self.entity_names = entity_names

        # print(self.is_node_attr('entity_names'))

    def subgraph(self, subset : Tensor):
        graph = super().subgraph(subset)
        # print(len(graph.entity_names), subset.shape, subset.max())
        # graph.entity_names = np.array(graph.entity_names)[subset.cpu().numpy()].tolist() if graph.entity_names is not None else None
        return graph

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)
    
    @property
    def num_features(self) -> int:
        return self.x.size(-1)
    
    @property
    def num_relations(self) -> int:
        return self.r.size(0)
    
    @property
    def num_edges(self) -> int:
        return self.edge_index.size(-1)
    
    @property
    def num_labels(self) -> int:
        return self.y.size(-1)

class ArticleAsKG:

    def __init__(self, 
        articles : ArticlesGraph,
        kg : KnowledgeGraph,
        edge_article : Tensor, 
        metadata : Any = None          
    ) -> None:
        
        self.articles = articles
        self.kg = kg
        self.edge_article = edge_article
        self.metadata = metadata

        ### Compute Entity<->Article
        triple_id, article_id = edge_article
        row, col = kg.edge_index[:,triple_id]

        self.entity_article = gnn_utils.coalesce(torch.cat([
            torch.stack([row, article_id], dim = 0),
            torch.stack([col, article_id], dim = 0)
        ],dim = -1)).to(self.edge_article.device)

    def to(self, device : str) -> 'ArticleAsKG':

        self.articles = self.articles.to(device)
        self.kg = self.kg.to(device)
        self.edge_article = self.edge_article.to(device)
        self.entity_article = self.entity_article.to(device)

        return self
    
    def cpu(self) -> 'ArticleAsKG':
        return self.to('cpu')
    
    def cuda(self) -> 'ArticleAsKG':
        return self.to('cuda')

    def subgraph_articles(self, ids : Tensor) -> 'ArticleAsKG':
    
        ### Get Articles Subset
        articles = self.articles.subgraph(subset = ids)

        ### Keep Only Triples and entities belonging to the article subset
        edge_article = self.edge_article
        triple_ids, article_ids = edge_article
        mask = torch.isin(article_ids, articles.article_id)
        triple_ids = torch.unique(triple_ids[mask])
        entity_ids = torch.unique(self.kg.edge_index[:,triple_ids])

        ### Get KG Subset & Remove Triples not extracted from the articles
        kg = self.kg.subgraph(entity_ids)
        mask = torch.isin(kg.triple_id, triple_ids)

        kg.edge_attr = kg.edge_attr[mask,:]
        kg.edge_index = kg.edge_index[:,mask]
        kg.edge_type = kg.edge_type[mask]
        kg.triple_id = kg.triple_id[mask]

        ### Get Subset Of Edge<->Article
        mask = torch.isin(edge_article[0], kg.triple_id) & torch.isin(edge_article[1], articles.article_id)
        edge_article = edge_article[:,mask]

        ### Map Edge<->Article and Entity<->Article

        ### 1- Create Maps
        articles_map = {
            id_.item() : i
            for i,id_ in enumerate(articles.article_id)
        }

        edge_map = {
            id_.item() : i
            for i,id_ in enumerate(kg.triple_id)
        }

        ### 2- Map Edge<->Article
        edge_article = torch.tensor([
            [edge_map[e],articles_map[a]]
            for e, a in edge_article.t().tolist()
        ]).t().long()

        ### 3- Map Triple Id
        kg.triple_id = torch.tensor([edge_map[e.item()] for e in kg.triple_id])
        kg.entity_id = torch.arange(kg.num_nodes)
        articles.article_id = torch.tensor([articles_map[a.item()] for a in articles.article_id])
        articles.edge_id = torch.arange(articles.num_edges)

        return ArticleAsKG(
            articles = articles,
            kg = kg,
            edge_article = edge_article,
            metadata = self.metadata
        )
    
    def subgraph_entities(self, entity_ids : Tensor) -> 'ArticleAsKG':
    
        ### Get KG Subset
        kg = self.kg.subgraph(entity_ids)

        ### Filter out Articles with 0 nodes
        edges, articles = self.edge_article
        article_ids = torch.unique(articles[torch.isin(edges, kg.triple_id)])

        ### Filter out Edge<->Article
        mask = torch.isin(edges, kg.triple_id) & torch.isin(articles, article_ids)
        edge_article = self.edge_article[:,mask]

        articles = self.articles.subgraph(article_ids)

        ### Map Edge<->Article and Entity<->Article

        ### 1- Create Maps
        articles_map = {
            id_.item() : i
            for i,id_ in enumerate(articles.article_id)
        }

        edge_map = {
            id_.item() : i
            for i,id_ in enumerate(kg.triple_id)
        }

        ### 2- Map Edge<->Article
        edge_article = edge_article.t().tolist()
        edge_article = torch.tensor([
            [edge_map[e],articles_map[a]]
            for e, a in edge_article
        ]).t().long()

        ### 3- Map Triple Id
        kg.triple_id = torch.tensor([edge_map[e.item()] for e in kg.triple_id])
        kg.entity_id = torch.arange(kg.num_nodes)
        articles.article_id = torch.tensor([articles_map[a.item()] for a in articles.article_id])
        articles.edge_id = torch.arange(articles.num_edges)

        return ArticleAsKG(
            articles = articles,
            kg = kg,
            edge_article = edge_article,
            metadata = self.metadata
        )
        
    
    def describe(self) -> None:

        print(f"Number of articles : {self.articles.num_nodes}")
        print(f"Number of edges in articles graph : {self.articles.num_edges}")
        print(f"Number of entities : {self.kg.num_nodes}")
        print(f"Number of Relations : {self.kg.num_relations}")
        print(f"Number of triples : {self.kg.num_edges}")
        print(f"Number of edges between articles and triples : {self.edge_article.size(-1)}")
        print(f"Number of edges between articles and entities : {self.entity_article.size(-1)} \n\n")