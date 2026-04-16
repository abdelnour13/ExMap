import torch
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torch_geometric.sampler import NeighborSampler
from typing import Dict, Generator
from src import helpers
from src.gnn.sampler.datasets import ArticlesGraph, KnowledgeGraph, ArticleAsKG

class HybridSampler:
    
    def __init__(self,
        articles : ArticlesGraph,
        kg : KnowledgeGraph,
        edge_article : Tensor,
        articles_loader_args : Dict,
        kg_sampler_args : Dict
    ) -> None:
        
        self.articles = articles
        self.kg = kg
        self.articles_loader_args = articles_loader_args
        self.kg_sampler_args = kg_sampler_args
        self.edge_article = edge_article
        
        self.articles_loader = NeighborLoader(
            data = articles,
            **articles_loader_args
        )

        self.kg_sampler = NeighborSampler(
            data = kg,
            **kg_sampler_args
        )

    def __len__(self) -> int:
        return len(self.articles_loader)
    
    def __iter__(self) -> Generator[ArticleAsKG, None, None]:

        articles : ArticlesGraph = None
        
        ### Get Subsets
        for articles in self.articles_loader:
                        
            ### Get the edges and that belong 
            ### to the set of sampled articles
            edge_ids, article_ids = self.edge_article
            edges = torch.unique(edge_ids[torch.isin(article_ids, articles.article_id)])
            nodes = torch.unique(self.kg.edge_index[:,edges])
            
            ### Get KG subgraph
            kg = self.kg.edge_subgraph(edges)
            kg = kg.subgraph(nodes)

            ### Map Edge<->Articles
            articles_map = torch.zeros(size = (articles.article_id.max() + 1, )).long()
            articles_map[articles.article_id] = torch.arange(articles.article_id.size(0))

            edges_map = torch.zeros(size = (kg.triple_id.max() + 1, )).long()
            edges_map[kg.triple_id] = torch.arange(kg.triple_id.size(0))


            edge_ids, article_ids = self.edge_article
            mask = torch.isin(edge_ids, kg.triple_id) & torch.isin(article_ids, articles.article_id)
            edge_ids, article_ids = edge_ids[mask], article_ids[mask]

            article_ids = articles_map[article_ids]
            edge_ids = edges_map[edge_ids]

            edge_article = torch.stack([edge_ids, article_ids], dim = 0).to(self.edge_article.device).long()

            yield ArticleAsKG(
                articles = articles,
                kg = kg,
                edge_article = edge_article
            )

if __name__ == '__main__':

    A_Nx, A_Ne, A_Nr = 100, 300, 2
    A_D = 32

    A = helpers.create_toy_graph(
        num_nodes = A_Nx,
        num_edges = A_Ne,
        num_types = A_Nr,
        node_features = A_D,
        edge_features = A_D,
        type_features = A_D
    )

    A = ArticlesGraph(
        x = A["x"],
        edge_index = A["edge_index"],
        edge_type = A["edge_type"],
    )

    KG_Nx, KG_Ne, KG_Nr = 1000, 2000, 8
    KG_D = 32

    KG = helpers.create_toy_graph(
        num_nodes = KG_Nx,
        num_edges = KG_Ne,
        num_types = KG_Nr,
        node_features = KG_D,
        edge_features = KG_D,
        type_features = KG_D
    )

    KG = KnowledgeGraph(
        x = KG["x"],
        e = KG["e"],
        r = KG["r"],
        edge_index = KG["edge_index"],
        edge_type = KG["edge_type"],
    )

    edge_article = torch.stack([
        torch.arange(KG_Ne),
        torch.randint(0, A_Nx, size = (KG_Ne,))
    ],dim = 0)

    sampler = HybridSampler(
        articles = A,
        kg = KG,
        edge_article = edge_article,
        articles_loader_args = {
            "batch_size" : 4,
            "num_neighbors" : (4,4)
        },
        kg_sampler_args = {
            "num_neighbors" : (4,)
        }
    )

    batch : ArticleAsKG = next(iter(sampler))

    print(batch.articles)
    print(batch.kg)
    print(batch.edge_article.size())

    print(batch.edge_article[0].max().item(), batch.edge_article[1].max().item())

    print(len(batch.edge_article[0].unique()), len(batch.edge_article[1].unique()))

    from tqdm.auto import tqdm

    for epoch in range(2):
        for batch in tqdm(sampler, total = len(sampler)):
            pass