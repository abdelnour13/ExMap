import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torch_geometric.data import HeteroData
from typing import Literal, Optional, Dict, Tuple, List
from dataclasses import dataclass
from torch_scatter import scatter
from pydantic import BaseModel
from src.gnn.layers import RGATConv, TransGNN, MLP, Activation

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
        sigma =  torch.clip(F.sigmoid(self.gate(messages)), 0.2, 0.8)
        x = sigma * x + (1 - sigma) * messages
        return self.act(self.bn(x))

class RGNN(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        variant: Literal["conv", "attn"] = "conv",
        norm : bool = False,
        activation : str = "relu",
        **kwargs
    ) -> None:
        super().__init__()
        
        self.variant = variant
        self.num_relations = num_relations

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

        self.bn = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.act = Activation(activation)

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
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
        kgnn_args: Optional[Dict] = None,
        rgnn_args: Optional[Dict] = None,
        gnn_args: Optional[Dict] = None,
        biderectional : bool = False,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.kgnn_args = kgnn_args or {}
        self.rgnn_args = rgnn_args or {}
        self.gnn_args = gnn_args or {}
        self.biderectional = biderectional
        self.use_mlp = self.rgnn_args.pop("use_mlp") if "use_mlp" in self.rgnn_args else False

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
        self.gnn_1 = StdGNN(
            in_channels = (out_channels, out_channels),
            out_channels = out_channels,
            variant = variant,
            norm=norm,
            activation=activation,
            **self.gnn_args
        )

        if biderectional:

            self.gnn_2 = StdGNN(
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
    ) -> Tuple[Tensor, Tensor, Tensor]: 

        kg_x, kg_e, kg_A = kg
        art_x, art_e, art_A = articles

        ### Trans GNN Forward
        kg_x, kg_e, _ = self.trans_gnn.forward(x=kg_x, edge_index=kg_A, edge_attr=kg_e)

        ### R-GNN Forward
        if self.use_mlp:
            art_x = self.rgnn(art_x)
        else:
            art_x = self.rgnn.forward(art_x, art_A, art_e)

        ### GNN Forward
        new_kg_x = self.gnn_1.forward(x=(art_x, kg_x), edge_index=entity_article[[1,0],:])

        if self.biderectional:
            art_x = self.gnn_2.forward(x=(kg_x, art_x), edge_index=entity_article)

        kg_x = new_kg_x

        return kg_x, kg_e, art_x
    
class HetGNNConfig(BaseModel):
    channels : Optional[List[int]] = None
    variant: Literal["conv", "attn"] = "conv"
    readout : Literal['mean','max','attn'] = 'max' 
    activation: str = "relu"
    ssl : bool = False
    norm: bool = False
    kgnn_args: Optional[Dict] = None
    rgnn_args: Optional[Dict] = None
    gnn_args: Optional[Dict] = None   
    jk : bool = True
    dropout_rate : float = 0.0
    num_clusters : int = 5  

@dataclass
class HetModelReturn:

    ### Guidance Output
    articles_predictions : Dict[str, Tensor]
    entities_predictions : Dict[str, Tensor]

    ### Embeddings
    entities_embeddings : Tensor
    articles_embeddings : Tensor

    ### SSL
    y_hat : Optional[Tensor]
    
    
class HetModel(nn.Module):

    def __init__(self, 
        in_channels : int,
        num_labels : Dict[str, int],
        num_relations: Tuple[int, int],
        channels : Optional[List[int]] = None,
        variant: Literal["conv", "attn"] = "conv",
        activation: str = "relu",
        norm: bool = False,
        kgnn_args: Optional[Dict] = None,
        rgnn_args: Optional[Dict] = None,
        gnn_args: Optional[Dict] = None,   
        jk : bool = True,
        readout : Literal['mean','max','attn'] = 'max',
        dropout_rate : float = 0.0,
        ssl : bool = False,
        cluster : bool = False,
        num_clusters : int = 5          
    ) -> None:
        super().__init__()

        self.num_relations = num_relations
        self.channels = [in_channels, *(channels or [in_channels // 2] * 2)]
        self.jk = jk
        self.readout = readout
        self.ssl = ssl
        self.cluster = cluster
        self.num_clusters = num_clusters

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
            )

            self.layers.append(layer)

        ## JK 
        self.kg_jk = gnn.JumpingKnowledge(mode = "max")
        self.edge_jk = gnn.JumpingKnowledge(mode = "max")
        self.art_jk = gnn.JumpingKnowledge(mode = "max")

        ## Final Passing after JK
        self.gnn = StdGNN(
            in_channels = (self.channels[-1],self.channels[-1]),
            out_channels = self.channels[-1],
            variant = variant,
            norm = norm,
            activation=activation
        )

        ### Projection Layers
        self.proj_e = nn.Linear(self.channels[-1], self.channels[-1])
        self.proj_x = nn.Linear(self.channels[-1], self.channels[-1])
        self.proj_art = nn.Linear(self.channels[-1], self.channels[-1])

        ### Pooling
        self.pool = None

        if readout == 'mean':
            self.readout_fn = gnn.MeanAggregation()
        elif readout == 'max':
            self.readout_fn = gnn.MaxAggregation()
        else:
            self.readout_fn = gnn.AttentionalAggregation(gate_nn = nn.Linear(self.channels[-1], 1))

        ### Prediction Heads
        self.heads = nn.ModuleDict()

        for label_name, label_dim in num_labels.items():

            self.heads.add_module(label_name, nn.Sequential(
                nn.Dropout(p = dropout_rate) if dropout_rate > 0 else nn.Identity(),
                nn.Linear(self.channels[-1], label_dim)
            ))

        ### SSL entity <-> article
        self.ssl_head = nn.Sequential(
            nn.Linear(2 * self.channels[-1], self.channels[-1]),
            nn.BatchNorm1d(self.channels[-1]),
            Activation(activation),
            nn.Linear(self.channels[-1], 1)
        )

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

        kg_x, kg_e, kg_A = batch['Entity'].x, batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr, batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        art_x, art_e, art_A = batch['Article'].x, batch['Article','relate-to','Article'].edge_type, batch['Article', 'relate-to', 'Article'].edge_index

        kg_x_jk = []
        art_x_jk = []
        edge_x_jk = []
        entity_article = batch['Entity','APPEARIN', 'Article'].edge_index

        ### Encoder
        for layer in self.layers:

            kg_x, kg_e, art_x = layer(
                (kg_x, kg_e, kg_A),
                (art_x, art_e, art_A),
                entity_article
            )

            kg_x_jk.append(kg_x)
            art_x_jk.append(art_x)
            edge_x_jk.append(kg_e)

        ### Jumping Knowledge
        if self.jk:

            kg_x = self.kg_jk.forward(kg_x_jk)
            art_x = self.art_jk.forward(art_x_jk)
            kg_e = self.edge_jk.forward(edge_x_jk)

            ### Final Message Passing After KG
            kg_x = self.gnn.forward(x=(art_x, kg_x), edge_index=entity_article[[1,0],:])

        ### Projection Layer
        kg_x = self.proj_x(kg_x)
        kg_e = self.proj_e(kg_e)
        art_x = self.proj_art(art_x)

        return kg_x, kg_e, art_x
    
    def article_embddings(self, 
        x : Tensor,
        entity_article : Tensor,
        num_articles : int                      
    ) -> Tensor:
        
        return self.readout_fn(
            x[entity_article[0]], 
            index = entity_article[1], 
            dim_size = num_articles,
            dim = 0
        )
    
    def relations(self, edge_attr : Tensor, edge_type : Tensor) -> Tensor:

        r = scatter(
            src = edge_attr,
            index = edge_type,
            dim = 0,
            dim_size = self.num_relations[0],
            reduce = "mean"
        )

        return self.proj_r(r)
    
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
        kg_x, _, art_x = self.encode(batch)
        
        ### Pooling
        entity_article = batch['Entity','APPEARIN', 'Article'].edge_index
        art_x_p = self.article_embddings(kg_x, entity_article, batch['Article'].x.size(0))

        ### Prediction
        entities_predictions = self.predict(kg_x)
        articles_predictions = self.predict(art_x_p)

        ### SSL
        y_hat = None
        
        if self.ssl:

            if entity_article is None:
                raise Exception("entity_article can not be null when ssl is true.")
            
            entity_ids, articles_ids = entity_article
            
            x = torch.cat([kg_x[entity_ids], art_x[articles_ids]], dim=-1)
            y_hat = self.ssl_head(x)

        return HetModelReturn(
            articles_predictions=articles_predictions,
            entities_predictions=entities_predictions,
            entities_embeddings=kg_x,
            articles_embeddings=art_x,
            y_hat=y_hat
        )