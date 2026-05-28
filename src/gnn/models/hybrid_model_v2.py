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
        sigma =  torch.clip(F.sigmoid(self.gate(messages)), 0.3, 0.7)
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

        # Update Articles Embeddings With Entities embeddings
        if self.use_entities_articles_links:
            art_x = self.gnn.forward(x=(kg_x, art_x), edge_index=entity_article)

        return kg_x, kg_e, art_x
    
class HetGNNConfig(BaseModel):
    channels : Optional[List[int]] = None
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

@dataclass
class HetModelReturn:

    ### Embeddings : View 1 : Entities Encoder
    entities_embeddings_v1 : Tensor
    articles_embeddings_v1 : Tensor # By aggregating entities embeddings

    ### SSL : View 2 : Articles Encoder
    entities_embeddings_v2 : Tensor # By aggregating articles embeddings
    articles_embeddings_v2 : Tensor

    ### Guidance Output
    articles_predictions_v1 : Optional[Dict[str, Tensor]] = None
    entities_predictions_v1 : Optional[Dict[str, Tensor]] = None

    articles_predictions_v2 : Optional[Dict[str, Tensor]] = None
    entities_predictions_v2 : Optional[Dict[str, Tensor]] = None

class HetModel(nn.Module):

    def __init__(self, 
        in_channels : int,
        num_relations: Tuple[int, int],
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
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.num_relations = num_relations
        self.channels = [in_channels, *(channels or [in_channels // 2] * 2)]
        self.jk = jk
        self.readout = readout
        self.use_entities_articles_links = use_entities_articles_links

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
        self.proj_x = nn.Linear(self.channels[-1], self.channels[-1])
        self.proj_art = nn.Linear(self.channels[-1], self.channels[-1])

        ### Pooling
        self.pool = None

        if readout == 'mean':
            self.entities_readout_fn = gnn.MeanAggregation()
            self.articles_readout_fn = gnn.MeanAggregation()
        elif readout == 'max':
            self.entities_readout_fn = gnn.MaxAggregation()
            self.articles_readout_fn = gnn.MaxAggregation()
        else:
            self.entities_readout_fn = gnn.AttentionalAggregation(gate_nn = nn.Linear(self.channels[-1], 1))
            self.articles_readout_fn = gnn.AttentionalAggregation(gate_nn = nn.Linear(self.channels[-1], 1))

        ### Prediction Heads
        self.heads = nn.ModuleDict()

        if num_labels is not None:

            for label_name, label_dim in self.num_labels.items():

                self.heads.add_module(label_name, nn.Sequential(
                    nn.Dropout(p = dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    nn.Linear(self.channels[-1], label_dim)
                ))

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
    
    def article_embddings(self, 
        x : Tensor,
        entity_article : Tensor,
        num_articles : int                      
    ) -> Tensor:
        
        return self.entities_readout_fn(
            x[entity_article[0]], 
            index = entity_article[1], 
            dim_size = num_articles,
            dim = 0
        )
    
    def entities_embddings(self, 
        x : Tensor,
        entity_article : Tensor,
        num_entities : int                      
    ) -> Tensor:
        
        return self.entities_readout_fn(
            x[entity_article[1]], 
            index = entity_article[0], 
            dim_size = num_entities,
            dim = 0
        )
    
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

        art_x_v1 = self.article_embddings(kg_x_v1, entity_article, batch['Article'].x.size(0))
        kg_x_v2 = self.entities_embddings(art_x_v2, entity_article, batch['Entity'].x.size(0))

        ### Prediction
        entities_predictions_v1 = None
        articles_predictions_v2 = None
        entities_predictions_v1 = None
        entities_predictions_v2 = None

        if self.num_labels is not None:

            entities_predictions_v1 = self.predict(kg_x_v1)
            entities_predictions_v2 = self.predict(kg_x_v2)

            articles_predictions_v1 = self.predict(art_x_v1)
            articles_predictions_v2 = self.predict(art_x_v2)

        return HetModelReturn(
            articles_predictions_v1=articles_predictions_v1,
            articles_predictions_v2=articles_predictions_v2,
            entities_predictions_v1=entities_predictions_v1,
            entities_predictions_v2=entities_predictions_v2,
            entities_embeddings_v1=kg_x_v1,
            entities_embeddings_v2=kg_x_v2,
            articles_embeddings_v1=art_x_v1,
            articles_embeddings_v2=art_x_v2 
        )