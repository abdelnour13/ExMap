import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric import nn as gnn, utils as gnn_utils
from torch_geometric.data import HeteroData
from typing import Literal, Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
from pydantic import BaseModel
from src.gnn.layers import RGATConv, TransGNN, MLP, Activation
from src.gnn.losses import (
    ClipLossContext,
    DiversityLossContext,
    ConsistencyLossContext,
)
from warnings import warn
from torch_scatter import scatter_sum, scatter

class StochasticAttnPooling(nn.Module):

    def __init__(self, rate : float, input_channels : int, temp : float = 3.0):
        super().__init__()

        self.rate = rate
        self.input_channels = input_channels

        self.register_buffer('temp', torch.scalar_tensor(temp))

        self.gate = nn.Linear(2 * input_channels, 1) # Entities alone ir entities-articles interactions ?

    def forward(self, 
        x_e : Tensor, 
        x_a : Tensor,
        edge_index : Tensor, 
    ) -> Tensor:
        
        row, col = edge_index # (entities-articles)

        scores = self.gate(torch.cat([x_e[row], x_a[col]], dim=-1))
        scores = gnn_utils.softmax(scores/self.get_buffer('temp'), col, num_nodes=x_a.shape[0])

        if self.training:

            X, M = gnn_utils.to_dense_batch(  # (N,M,D)
                x=x_e,
                batch=col
            )

            scores, _ = gnn_utils.to_dense_batch(  # (N,M)
                x=scores,
                batch=col
            )
            
            scores = scores[:,:,0]

            new_scores = []

            for _, entities_scores, mask in zip(X, scores, M):

                # Or uniformly random ?
                indices = torch.multinomial(
                    (1 - entities_scores) * mask.float(), 
                    num_samples=max(1, int(self.rate * mask.float().sum().item()))
                )
                    
                entities_scores[indices] = 0.0
                new_scores.append(entities_scores)

            new_scores = torch.stack(new_scores, dim=0).softmax(dim=-1)
            new_scores = new_scores[:,:,None]

            x_articles = torch.sum(X * new_scores, dim=1)

        else:

            x_articles = scatter_sum(
                src=x_e[row] * scores,
                index=col,
                reduce='sim',
                dim=0,
                dim_size=x_a.size(0)
            )

        return x_articles

class StdGNN(nn.Module):

    def __init__(
        self,
        in_channels: Tuple[int, int],
        out_channels: int,
        variant: Literal["conv", "attn"] = "conv",
        norm : bool = True,
        activation : str = "relu",
        **kwargs
    ) -> None:
        
        super().__init__()

        self.norm = norm
        self.activation = activation
        self.variant = variant
        self.gate = nn.Linear(in_features=out_channels, out_features=1)
        self.root = nn.Linear(in_features=in_channels[1],out_features=out_channels)

        if variant == "conv":

            self.conv = gnn.SAGEConv(
                in_channels=in_channels, 
                out_channels=out_channels, 
                root_weight=False,
                **kwargs
            )

        else:  # 'attn'

            gat_kwargs = { 
                "add_self_loops": False, 
                "residual" : False, 
                **kwargs
            }

            self.conv = gnn.GATConv(
                in_channels=in_channels, 
                out_channels=out_channels, 
                **gat_kwargs
            )

        self.bn = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.act = Activation(activation)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor
    ) -> Tensor:
        messages = self.conv(x, edge_index)
        x = self.root(x[1])
        sigma =  torch.clip(F.sigmoid(self.gate(messages)), 0.3, 0.7)
        x = sigma * x + (1 - sigma) * messages
        return self.act(self.bn(x))
    
class RGNN(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        variant: Literal["conv", "attn", "mlp"] = "conv",
        norm : bool = False,
        activation : str = "relu",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.variant = variant
        self.num_relations = num_relations

        self.use_mlp = kwargs.pop("use_mlp", False)

        if variant == "conv":

            self.conv = gnn.RGCNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                num_relations=num_relations,
                **kwargs
            )

        else:  # 'attn'

            self.conv = RGATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                num_relations=num_relations,
                **kwargs
            )

        if self.use_mlp:

            self.conv = nn.Linear(
                in_channels,
                out_channels,
                **kwargs
            )

        self.bn = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.act = Activation(activation)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:

        if self.use_mlp:
            return self.act(self.bn(self.conv(x)))

        return self.act(self.bn(self.conv(x, edge_index, edge_type)))
    
class HetConv(nn.Module):

    __trans_gnn__ = {
        "use_edges_info": True,
        "kg_scoring_fn": "TransE",
        "combine": "cat",
        "num_heads": 1,
        "bias": True,
        "version": "v2",
    }

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: Tuple[int, int],  # (kg, articles)
        variant: Literal["conv", "attn"] = "conv",
        activation: str = "relu",
        norm: bool = False,
        use_entities_articles_links : bool = True,
        kgnn_args: Optional[Dict] = None,
        rgnn_args: Optional[Dict] = None,
        gnn_args: Optional[Dict] = None,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.use_entities_articles_links = use_entities_articles_links
        self.kgnn_args = kgnn_args or {}
        self.rgnn_args = rgnn_args or {}
        self.gnn_args = gnn_args or {}
        self.use_mlp = num_relations[1] == 0

        # KG processing module
        self.trans_gnn = TransGNN(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations[0],
            variant=variant,
            normalize=norm,
            activation=activation,
            **{**self.__trans_gnn__, **self.kgnn_args}
        )

        # Article relation processing with RGNN
        if self.use_mlp:

            self.rgnn = MLP(
                input_dim=in_channels,
                channels = [in_channels],
                output_dim=out_channels,
                hidden_activation=activation,
                output_activation=activation,
                norm=norm
            )

        else:

            self.rgnn = RGNN(
                in_channels=in_channels,
                out_channels=out_channels,
                num_relations=num_relations[1],
                variant=variant,
                norm=norm,
                activation=activation,
                **self.rgnn_args
            )

        # Standard GNN for neighbor aggregation
        if self.use_entities_articles_links:

            self.gnn = StdGNN(
                in_channels = (out_channels, out_channels),
                out_channels = out_channels,
                variant = variant,
                norm=norm,
                activation=activation,
                **self.gnn_args
            )

    def forward(
        self,
        kg: Tuple[Tensor, Tensor, Tensor],
        articles: Tuple[Tensor, Tensor, Tensor],
        entity_article: Tensor,
        node_score : Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]: 

        kg_x, kg_e, kg_A = kg
        art_x, art_e, art_A = articles

        ### Trans GNN Forward
        kg_x, kg_e, _ = self.trans_gnn.forward(x=kg_x, edge_index=kg_A, edge_attr=kg_e, node_score=node_score)

        ### R-GNN Forward
        if self.use_mlp:
            art_x = self.rgnn(art_x)
        else:
            art_x = self.rgnn.forward(art_x, art_A, art_e)

        # Update Articles Embeddings With Entities embeddings
        if self.use_entities_articles_links:
            art_x = self.gnn.forward(x=(kg_x, art_x), edge_index=entity_article)

        return kg_x, kg_e, art_x
    
class HetGNNConfig(BaseModel):
    channels : Optional[List[int]] = None
    projection_dim : Optional[int] = None
    variant: Literal["conv", "attn"] = "conv"
    readout : Literal['mean','max','attn'] = 'max' 
    activation: str = "relu"
    norm: bool = False
    use_entities_articles_links : bool = True
    kgnn_args: Optional[Dict] = None
    rgnn_args: Optional[Dict] = None
    gnn_args: Optional[Dict] = None   
    jk : bool = True
    dropout_rate : float = 0.0
    n_prototypes : Optional[int] = None  

@dataclass
class ViewResults:

    entities_embeddings : Tensor
    articles_embeddings : Tensor

    entities_prototypes : Optional[Tensor] = None
    articles_prototypes : Optional[Tensor] = None

    entities_predictions : Optional[Dict[str, Tensor]] = None
    articles_predictions : Optional[Dict[str, Tensor]] = None

    attn_weights : Optional[Tensor] = None

@dataclass
class HetModelReturn:
    entities_view : ViewResults
    articles_view : ViewResults
    C : Optional[Tensor] = None

    def context(self, batch : HeteroData, batch_size : int) -> Dict[str, Any]:

        return {
            "clip_loss" : ClipLossContext(
                za=self.articles_view.articles_prototypes,
                ze=self.entities_view.articles_prototypes
            ),
            "diversity_loss" : DiversityLossContext(
                h=self.C
            ),
            "consistency_loss" : ConsistencyLossContext(
                xa=batch['Article'].x_passage,
                xe=batch['Entity'].x,
                a2t=self.articles_view.articles_prototypes,
                e2t=self.entities_view.entities_prototypes
            )
        }

class HetModel(nn.Module):

    def __init__(self, 
        in_channels : int,
        num_relations: Tuple[int, int],
        projection_dim : int,
        num_labels : Optional[Dict[str, int]] = None,
        channels : Optional[List[int]] = None,
        variant: Literal["conv", "attn"] = "conv",
        activation: str = "relu",
        use_entities_articles_links : bool = True,
        norm: bool = False,
        kgnn_args: Optional[Dict] = None,
        rgnn_args: Optional[Dict] = None,
        gnn_args: Optional[Dict] = None,   
        jk : bool = True,
        readout : Literal['mean','max','attn'] = 'max',
        dropout_rate : float = 0.0,      
        n_prototypes : Optional[int] = None   
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.num_relations = num_relations
        self.projection_dim = projection_dim
        self.channels = [in_channels, *(channels or [in_channels // 2] * 2)]
        self.jk = jk
        self.readout = readout
        self.use_entities_articles_links = use_entities_articles_links
        self.n_prototypes = n_prototypes

        self.layers = nn.ModuleList()

        ### Encoder
        for in_dim, out_dim in zip(self.channels[:-1], self.channels[1:]):

            layer = HetConv(
                in_channels = in_dim,
                out_channels = out_dim,
                num_relations = num_relations,
                variant = variant,
                activation = activation,
                norm = norm,
                kgnn_args = kgnn_args,
                rgnn_args = rgnn_args,
                gnn_args = gnn_args,
                use_entities_articles_links=use_entities_articles_links
            )

            self.layers.append(layer)

        ## JK 
        self.kg_jk = gnn.JumpingKnowledge(mode = "max")
        self.art_jk = gnn.JumpingKnowledge(mode = "max")

        ## Final Passing after JK
        if self.use_entities_articles_links:

            self.gnn = StdGNN(
                in_channels = (self.channels[-1],self.channels[-1]),
                out_channels = self.channels[-1],
                variant = variant,
                norm = norm,
                activation=activation
            )

        ### Projection Layers
        self.proj_x = nn.Linear(self.channels[-1], projection_dim)
        self.proj_art = nn.Linear(self.channels[-1], projection_dim)

        ### Pooling
        self.pool = None

        if readout == 'attn':
            self.entities_attn_weights = nn.Linear(projection_dim, 1)
            self.articles_attn_weights = nn.Linear(projection_dim, 1)

        ### Centroids
        if n_prototypes is not None:

            self.C = nn.Sequential(
                Activation(activation),
                nn.Linear(in_features=projection_dim, out_features=n_prototypes, bias=False),
                nn.Sigmoid()
            )

        ### Prediction Heads
        self.heads = nn.ModuleDict()

        if num_labels is not None:

            for label_name, label_dim in self.num_labels.items():

                self.heads.add_module(label_name, nn.Sequential(
                    nn.Dropout(p = dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Linear(projection_dim, label_dim)
                ))

    def reduce(
        self,
        x : Tensor,
        edge_index : Tensor,
        num_nodes : int,
        weights : Optional[Tensor] = None,
        reduce : Literal['mean', 'max', 'attn'] = 'mean'
    ) -> Tensor:

        if weights is not None and reduce != 'attn':
            warn(message=f"Attention weights provide but reduce = '{reduce}'")

        if weights is None and reduce == 'attn':
            raise Exception(f"reduce = '{reduce}'. But attnetion weights not provided")

        row, col = edge_index

        if reduce != 'attn':

            x = scatter(
                src=x[row,:],
                index=col,
                dim=0,
                dim_size=num_nodes,
                reduce=reduce
            )

            return x

        weights = gnn_utils.softmax(
            src=weights[row],
            index=col,
            num_nodes=num_nodes,
            dim=0
        )

        x = scatter(
            src=weights * x[row,:],
            index=col,
            dim=0,
            dim_size=num_nodes,
            reduce='sum'
        )

        return x        

    @staticmethod
    def from_config(
        in_channels : int,
        num_labels : Dict[str, int],
        num_relations : Tuple[int, int],
        config : HetGNNConfig                
    ) -> 'HetModel':
        
        return HetModel(
            in_channels = in_channels,
            num_labels = num_labels,
            num_relations = num_relations,
            **config.model_dump()
        )
    
    def encode(self, batch : HeteroData) -> Tuple[Tensor, Tensor, Tensor]:

        kg_x, kg_e, kg_A = (
            batch['Entity'].x, 
            batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr, 
            batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        )

        art_x, art_e, art_A = (
            batch['Article'].x, 
            batch['Article','relate-to','Article'].edge_type, 
            batch['Article', 'relate-to', 'Article'].edge_index
        )

        kg_x_jk = []
        art_x_jk = []
        entity_article = batch['Entity','APPEARIN', 'Article'].edge_index

        ### Encoder
        node_score = batch['Entity'].get('node_score')

        for layer in self.layers:

            kg_x, kg_e, art_x = layer(
                (kg_x, kg_e, kg_A),
                (art_x, art_e, art_A),
                entity_article,
                node_score=node_score
            )

            kg_x_jk.append(kg_x)
            art_x_jk.append(art_x)

        ### Jumping Knowledge
        if self.jk:

            kg_x = self.kg_jk.forward(kg_x_jk)
            art_x = self.art_jk.forward(art_x_jk)

            if self.use_entities_articles_links:

                ### Final Message Passing After KG
                art_x = self.gnn.forward(x=(kg_x, art_x), edge_index=entity_article[[1,0],:])

        ### Projection Layer
        kg_x = self.proj_x(kg_x)
        art_x = self.proj_art(art_x)

        return kg_x, art_x
    
    def predict(self, x : Tensor) -> Dict[str, Tensor]:
        
        return {
            label_name : m(x)
            for label_name, m in self.heads.items()
        }

    def forward(self, 
        batch : HeteroData,
        entity_article : Optional[Tensor] = None # 2xN : entity_id, article_id
    ) -> HetModelReturn:

        ### Encoder        
        kg_x_v1, art_x_v2 = self.encode(batch)
        
        ### Pooling
        entity_article = batch['Entity','APPEARIN', 'Article'].edge_index

        entities_attn_weights = None
        articles_attn_weights = None

        if self.readout == 'attn':
            entities_attn_weights = self.entities_attn_weights(kg_x_v1)
            articles_attn_weights = self.articles_attn_weights(art_x_v2)

        art_x_v1 = self.reduce(
            x=kg_x_v1,
            edge_index=entity_article,
            num_nodes=batch['Article'].x.size(0),
            weights=entities_attn_weights,
            reduce=self.readout
        )

        kg_x_v2 = self.reduce(
            x=art_x_v2,
            edge_index=entity_article[[1,0],:],
            num_nodes=batch['Entity'].x.size(0),
            weights=articles_attn_weights,
            reduce=self.readout
        )

        ### Prototypes
        kg_x_v1_prot = None
        kg_x_v2_prot = None
        art_x_v1_prot = None
        art_x_v2_prot = None
        
        if self.n_prototypes is not None:

            kg_x_v1_prot = self.C(kg_x_v1)
            art_x_v2_prot = self.C(art_x_v2)

            art_x_v1_prot = self.reduce(
                x=kg_x_v1_prot,
                edge_index=entity_article,
                num_nodes=batch['Article'].x.size(0),
                weights=entities_attn_weights,
                reduce=self.readout
            )
    
            kg_x_v2_prot = self.reduce(
                x=art_x_v2_prot,
                edge_index=entity_article[[1,0],:],
                num_nodes=batch['Entity'].x.size(0),
                weights=articles_attn_weights,
                reduce=self.readout
            )

        ### Predictions
        kg_x_v1_pred = None
        kg_x_v2_pred = None
        art_x_v1_pred = None
        art_x_v2_pred = None

        if (self.num_labels is not None) and (len(self.num_labels) > 0):
            kg_x_v1_pred = self.predict(kg_x_v1)
            kg_x_v2_pred = self.predict(kg_x_v2)
            art_x_v1_pred = self.predict(art_x_v1)
            art_x_v2_pred = self.predict(art_x_v2)

        return HetModelReturn(
            entities_view=ViewResults(
                entities_embeddings=kg_x_v1,
                articles_embeddings=art_x_v1,
                entities_prototypes=kg_x_v1_prot,
                articles_prototypes=art_x_v1_prot,
                entities_predictions=kg_x_v1_pred,
                articles_predictions=art_x_v1_pred,
                attn_weights=entities_attn_weights
            ),
            articles_view=ViewResults(
                entities_embeddings=kg_x_v2,
                articles_embeddings=art_x_v2,
                entities_prototypes=kg_x_v2_prot,
                articles_prototypes=art_x_v2_prot,
                entities_predictions=kg_x_v2_pred,
                articles_predictions=art_x_v2_pred,
                attn_weights=articles_attn_weights
            ),
            C=self.C[1].weight
        )