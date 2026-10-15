from torch import nn, Tensor
from typing import Literal, Optional, Dict, List, Tuple
from src.gnn.layers import Reduction, MultiTaskHead, Activation
from src.helpers import torch_utils
from torch_geometric.data import HeteroData
from dataclasses import dataclass
from src.gnn.losses import (
    ClipLossContext,
    DiversityLossContext,
    ConsistencyLossContext,
    SharedTopicsLossContext
)
from pydantic import BaseModel
from src.gnn.topic_models.output import BaseLossContext
from src.gnn.layers.encoder import RGNNEncoder, TransEncoder


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
class ConstrastiveModelOutput:
    entities_view : ViewResults
    articles_view : ViewResults
    C : Optional[Tensor] = None

    def context(self, batch : HeteroData, batch_size : int) -> Dict[str, BaseLossContext]:

        return {
            "clip_loss" : ClipLossContext(
                za=self.articles_view.articles_prototypes,
                ze=self.entities_view.articles_prototypes
            ),
            "shared_topics_loss" : SharedTopicsLossContext(
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

class ConstrastiveModelConfig(BaseModel):
    out_channels : Optional[int] = None
    hidden_channels : List[int]  = None
    variant : Literal['conv', 'attn'] = 'conv'
    activation : str = 'relu'
    norm : bool = True
    readout : Literal['mean','max','attn'] = 'mean'
    n_prototypes : int = 128
    shortcut: bool = False
    shortcut_dim: Optional[int] = None
    jk: bool = True
    doc_encoder_type : Literal['rgnn', 'mlp'] = 'mlp'

class ConstrastiveModel(nn.Module):

    def __init__(self,
        in_channels : int,
        out_channels : Optional[int],
        hidden_channels : List[int],
        num_relations : Tuple[int, int],
        variant : Literal['conv', 'attn'],
        activation : str,
        norm : bool,
        readout : Literal['mean','max','attn'],
        n_prototypes : int,
        num_labels : Optional[Dict[str, int]],
        shortcut: bool,
        shortcut_dim: int | None,
        jk: bool,
        doc_encoder_type : Literal['RGNN', 'MLP']
    ) -> None:

        super().__init__()

        out_channels = out_channels or hidden_channels[-1]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.readout = readout
        self.n_prototypes = n_prototypes
        self.num_labels = num_labels
        self.shortcut = shortcut
        self.shortcut_dim = shortcut_dim
        self.jk = jk
        self.doc_encoder_type = doc_encoder_type

        self.kg_encoder = TransEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_relations=num_relations[0],
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk
        )

        self.doc_encoder = RGNNEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_relations=num_relations[1],
            activation=activation,
            agg=('mean' if variant == 'conv' else 'attn'),
            norm=norm,
            root=True,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(doc_encoder_type == 'mlp')
        )

        self.C = nn.Sequential(*[
            Activation(activation),
            nn.Linear(in_features=out_channels, out_features=n_prototypes, bias=False),
            nn.Sigmoid()
        ])

        self.kg_readout = self.create_readout()
        self.doc_readout = self.create_readout()

        if num_labels is not None:

            self.head = MultiTaskHead(
                in_features=out_channels,
                dropout_rate=0.0,
                num_labels=num_labels
            )

    @staticmethod
    def from_config(
        in_channels : int,
        num_labels : Dict[str, int],
        num_relations : Tuple[int, int],
        config : ConstrastiveModelConfig                
    ) -> 'ConstrastiveModel':
        
        return ConstrastiveModel(
            in_channels = in_channels,
            num_labels = num_labels,
            num_relations = num_relations,
            **config.model_dump()
        )

    def create_readout(self) -> Reduction:

        return Reduction(
            kind=self.readout,
            weights_nn=nn.Linear(in_features=self.out_channels, out_features=1) if self.readout == 'attn' else None
        )

    def forward(self, batch : HeteroData) -> ConstrastiveModelOutput:

        KG = {
            'x' : batch['Entity'].x, 
            'edge_attr' : batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr, 
            'edge_index' : batch['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        }

        DOC_G = {
            'x' : batch['Article'].x, 
            'edge_type' : batch['Article','relate-to','Article'].edge_type, 
            'edge_index' : batch['Article', 'relate-to', 'Article'].edge_index
        }

        ED = batch['Entity', 'APPEARIN', 'Article'].edge_index

        ### Encoder
        entities_view__entities_embeddings = self.kg_encoder(KG)
        articles_view__articles_embeddings = self.doc_encoder(DOC_G)

        ### Readout
        entities_view__articles_embeddings, entities_weights, _ = self.kg_readout.forward(
            x=entities_view__entities_embeddings,
            edge_index=ED,
            num_nodes=DOC_G['x'].size(0)
        ).astuple()

        articles_view__entities_embeddings, articles_weights, _ = self.doc_readout.forward(
            x=articles_view__articles_embeddings,
            edge_index=ED[[1,0],:],
            num_nodes=KG['x'].size(0)
        ).astuple()

        ### Prediction
        entities_view__entities_predictions = None
        entities_view__articles_predictions = None
        articles_view__entities_predictions = None
        articles_view__articles_predictions = None

        if self.num_labels is not None:
            entities_view__entities_predictions = self.head(entities_view__entities_embeddings)
            entities_view__articles_predictions = self.head(entities_view__articles_embeddings)
            articles_view__entities_predictions = self.head(articles_view__entities_embeddings)
            articles_view__articles_predictions = self.head(articles_view__articles_embeddings)

        ### Compute Prototypes
        entities_view__entities_prototypes = self.C(entities_view__entities_embeddings)
        articles_view__articles_prototypes = self.C(articles_view__articles_embeddings)

        entities_view__articles_prototypes, _, _ = self.kg_readout.forward(
            x=entities_view__entities_prototypes,
            edge_index=ED,
            num_nodes=DOC_G['x'].size(0),
            weights=entities_weights
        ).astuple()

        articles_view__entities_prototypes, _, _ = self.doc_readout.forward(
            x=articles_view__articles_prototypes,
            edge_index=ED[[1,0],:],
            num_nodes=KG['x'].size(0),
            weights=articles_weights
        ).astuple()

        return ConstrastiveModelOutput(
            entities_view=ViewResults(
                entities_embeddings=entities_view__entities_embeddings,
                articles_embeddings=entities_view__articles_embeddings,
                entities_prototypes=entities_view__entities_prototypes,
                articles_prototypes=entities_view__articles_prototypes,
                entities_predictions=entities_view__entities_predictions,
                articles_predictions=entities_view__articles_predictions,
                attn_weights=entities_weights
            ),
            articles_view=ViewResults(
                articles_embeddings=articles_view__articles_embeddings,
                entities_embeddings=articles_view__entities_embeddings,
                articles_prototypes=articles_view__articles_prototypes,
                entities_prototypes=articles_view__entities_prototypes,
                entities_predictions=articles_view__entities_predictions,
                articles_predictions=articles_view__articles_predictions,
                attn_weights=articles_weights
            ),
            C=self.C[1].weight
        )

if __name__ == '__main__':

    G = torch_utils.create_toy_graph(
        num_nodes=128,
        num_edges=410,
        num_types=4,
        node_features=128,
        edge_features=128,
        type_features=128
    )

    from itertools import product

    combinations = product(
        [True, False],
        ['mean', 'attn'],
        [True, False],
        [True, False]
    )

    columns = [
        'expect_edges_info',
        'agg',
        'root',
        'shortcut'
    ]

    for comb in map(lambda x : dict(zip(columns,x)), combinations):

        print(comb)

        encoder = RGNNEncoder(
            in_channels = 128,
            out_channels = 10,
            hidden_channels = [64, 64],
            activation = "relu",
            norm = True,
            shortcut_dim = None,
            jk = True,
            **comb
        )

        output = encoder.forward({
            "x" : G['x'],
            "edge_index" : G['edge_index'],
            "edge_attr" : G['r'][G['edge_type']]
        })

        print(output.shape, end="\n\n")
    