import torch
from torch import nn, Tensor
from src import helpers
from src.graph_dl.pooling.cluster_pool import create_pooling_layer, POOLING_KIND, Pool, ClusterPoolOutput
from src.graph_dl.models.base_model import TopicModel, Context, IterationContext
from src.graph_dl.layers.trans_gnn import TransGNN
from src.graph_dl.layers.encoder import _Linear
from src.graph_dl.layers.activation import Activation
from src.graph_dl import losses
from torch_geometric.data import HeteroData
from src.graph_dl.utils import AdjMatrix, EdgeIndex
from typing import List, Literal, Optional, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field

ASSIGN_NN = Literal['gnn', 'lin']

class _Wrap(nn.Module):

    def __init__(self, nn : nn.Module):
        super().__init__()
        self.nn = nn

    def forward(self, x : Tensor, edge_index : Tensor, edge_attr : Optional[Tensor]) -> Tensor:
        out = self.nn(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return out['x']

_Layer = _Linear('x')

@TopicModel.register
class GraphUNet(TopicModel):

    __trans_args__ = {
        'kg_scoring_fn' : 'TransE',
        'variant' : 'conv',
        'combine' : 'cat',
        'bias' : True
    }

    class Config(BaseModel):
        hidden_channels : List[int] = Field(default_factory=list)
        out_channels : int = 256
        use_edges_info : bool = False
        activation : Optional[str] = 'relu'
        norm : bool = True
        pooling_layer : POOLING_KIND = 'dmon'
        norm_pool : bool = False
        num_clusters : List[int] = Field(default_factory=lambda x : [256])
        assign_nn : ASSIGN_NN = 'gnn'

    @dataclass
    class Output:
        x_hidden : Tensor
        x_hat : Tensor
        S : List[Tensor]
        aux_loss : Tensor
        
        def context(self, batch : HeteroData, batch_size : int) -> Dict[str, losses.BaseLossContext]:

            return {
                "rec_loss" : losses.CosineRecLossContext(
                    x1=self.x_hat,
                    y1=batch['Entity'].x
                ),
                "aux_loss" : losses.Identity(value=self.aux_loss)
            }

    def __init__(self, 
        in_channels : int,
        hidden_channels : List[int],
        out_channels : int,
        use_edges_info : bool,
        activation : Optional[str],
        norm : bool,
        pooling_layer : POOLING_KIND,
        norm_pool : bool,
        num_clusters : List[int],
        assign_nn : ASSIGN_NN,
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_edges_info = use_edges_info
        self.activation = activation
        self.norm = norm
        self.pooling_layer = pooling_layer
        self.norm_pool = norm_pool
        self.num_clusters = num_clusters
        self.assign_nn = assign_nn

        self.encoder_layers : List[TransGNN] = nn.ModuleList(self.create_encoder_layers())
        self.decoder_layers : List[TransGNN] = nn.ModuleList(self.create_decoder_layers())
        self.pooling_layers : List[Pool] = nn.ModuleList(self.create_pooling_layers())

        self.proj = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            Activation(activation),
            nn.Linear(out_channels, out_channels)
        )

    @staticmethod
    def from_config(
        config : Config, 
        ctx : Context, 
        labels : Dict[str, int] = None
    ) -> 'GraphUNet':

        return GraphUNet(
            in_channels = ctx.num_features[0],
            **config.model_dump()
        )

    def create_encoder_layers(self) -> List[TransGNN]:

        channels = [self.in_channels, *self.hidden_channels, self.out_channels]

        return [
            TransGNN(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=self.activation,
                norm=self.norm,
                use_edges_info=self.use_edges_info,
                **self.__class__.__trans_args__
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:]))
        ]

    def create_decoder_layers(self) -> List[TransGNN]:
    
        channels = [self.out_channels, *self.hidden_channels[::-1], self.out_channels]

        return [
            TransGNN(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=self.activation,
                norm=self.norm,
                use_edges_info=self.use_edges_info,
                **self.__class__.__trans_args__
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:]))
        ]

    def create_assign_nn(self,
        in_channels : int,
        out_channels : int, 
        use_edges_info : bool
    ) -> nn.Module:

        if self.assign_nn == 'gnn':

            m = TransGNN(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None,
                norm=False,
                use_edges_info=use_edges_info,
                **self.__class__.__trans_args__
            )

        else:

            m = _Layer(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None,
                norm=False
            )

        return _Wrap(nn=m)

    def create_pooling_layers(self) -> List[Pool]:
    
        channels = [self.in_channels, *self.hidden_channels, self.out_channels]

        return [
            create_pooling_layer(
                kind=self.pooling_layer,
                in_channels=out_channels,
                norm=self.norm_pool,
                nn=self.create_assign_nn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_edges_info=self.use_edges_info
                )
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels[1:], self.num_clusters))
        ]

    def encode(self, 
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix   
    ) -> List[ClusterPoolOutput]:

        outputs = []

        for encoder_layer, pooling_layer in zip(self.encoder_layers, self.pooling_layers):

            out = encoder_layer.forward(
                x=x,
                edge_index=EE.edge_index.edge_index,
                edge_attr=None
            )
            
            out = pooling_layer.pool(
                x=out['x'],
                EE=EE,
                EA=EA
            )

            x= out.x
            EE = out.EE
            EA = out.EA

            outputs.append(out)

        # out = self.encoder_layers[-1].forward(
        #     x=out.x,
        #     edge_index=out.EE.edge_index.edge_index,
        #     edge_attr=None
        # )

        return outputs

    def decode(self, 
        x : Tensor,
        EE : AdjMatrix,
        EA : AdjMatrix,
        outs : List[ClusterPoolOutput]
    ) -> List[ClusterPoolOutput]:

        outputs = []

        for encoder_layer, pooling_layer, cluster_output in zip(self.decoder_layers, self.pooling_layers, outs[::-1]):

            out = pooling_layer.unpool(
                x=x,
                EE=EE,
                EA=EA,
                out=cluster_output
            )

            gnn_out = encoder_layer.forward(
                x=out.x,
                edge_index=EE.edge_index.edge_index,
                edge_attr=None
            )
    
            x  = gnn_out['x']
            out.x = x
            EE = out.EE
            EA = out.EA

            outputs.append(out)

        outputs[-1].x = self.proj(outputs[-1].x)

        return outputs

    def forward(self, 
        data : HeteroData,
        ctx : Optional[IterationContext] = None
    ) -> None:

        x = data['Entity'].x

        EE = AdjMatrix(
            _edge_index=EdgeIndex(
                row=data['Entity', 'HASRELATIONWITH', 'Entity'].edge_index[0],
                col=data['Entity', 'HASRELATIONWITH', 'Entity'].edge_index[1],
                num_nodes=(x.size(0), x.size(0))
            )
        )

        EA = AdjMatrix(
            _edge_index=EdgeIndex(
                row=data['Entity', 'APPEARIN', 'Article'].edge_index[0],
                col=data['Entity', 'APPEARIN', 'Article'].edge_index[1],
                num_nodes=(x.size(0), data['Article'].x.size(0))
            )
        )

        pooling_outputs = self.encode(
            x=x, 
            EE=EE, 
            EA=EA
        )

        unpooling_outputs = self.decode(
            x=pooling_outputs[-1].x, 
            EE=pooling_outputs[-1].EE, 
            EA=pooling_outputs[-1].EA, 
            outs=pooling_outputs
        )

        return self.Output(
            x_hidden=pooling_outputs[-1].x,
            x_hat=unpooling_outputs[-1].x,
            S=[o.S for o in pooling_outputs],
            aux_loss=sum(out.aux_loss for out in pooling_outputs)
        )


if __name__ == '__main__':

    data = helpers.create_toy_graph(
        num_entities=256,
        num_articles=25,
        num_features=64,
        ee=int((256*256)*0.02),
        ea=int((256*25)*0.04)
    )

    layer = GraphUNet(
        in_channels=data['xe'].size(-1),
        hidden_channels=[data['xe'].size(-1) // 2],
        out_channels=data['xe'].size(-1) // 3,
        use_edges_info=False,
        activation='relu',
        norm=False,
        pooling_layer='dmon',
        norm_pool=False,
        num_clusters=[data['xe'].size(0) // 2, data['xe'].size(0) // 4],
        assign_nn='gnn'
    )

    outputs = layer.encode(
        x=data['xe'],
        EE=AdjMatrix(
            _edge_index=EdgeIndex(
                data['EE'][0],
                data['EE'][1],
                num_nodes=(256,256)
            )
        ),
        EA=AdjMatrix(
            _edge_index=EdgeIndex(
                data['EA'][0],
                data['EA'][1],
                num_nodes=(256,25)
            )
        ),
    )

    for output in outputs:

        print(output.x.size())
        # print(output.S.size())
        # print(output.EE.dense.size())
        # print(output.EA.dense.size())
        # print(output.aux_loss.size() if output.aux_loss is not None else None)

    unpooling_outputs = layer.decode(x=outputs[-1].x, EE=outputs[-1].EE, EA=outputs[-1].EA, outs=outputs)

    print('Unpooling')

    for output in unpooling_outputs:
    
        print(output.x.size())
        # print(output.S.size())
        # print(output.EE.dense.size())
        # print(output.EA.dense.size())
        # print(output.aux_loss.size() if output.aux_loss is not None else None)
