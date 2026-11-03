import torch
from torch import nn, Tensor
from src import helpers
from src.graph_dl.pooling.cluster_pool import create_pooling_layer, POOLING_KIND, Pool, ClusterPoolOutput, UNPOOL_KIND
from src.graph_dl.models.base_model import TopicModel, Context, IterationContext
from src.graph_dl.layers.trans_gnn import TransGNN
from src.graph_dl.layers.encoder import _Linear, TransEncoder
from src.graph_dl.layers.reduction import Reduction
from src.graph_dl.layers.mlp import MLP, NORM_TYPE
from src.graph_dl import losses
from torch_geometric.data import HeteroData
from src.graph_dl.utils import AdjMatrix, EdgeIndex
from typing import List, Literal, Optional, Dict, Any
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

class GraphUNetBlock(nn.Module):

    def __init__(self, 
        in_channels : int,
        out_channels : int,
        norm : Optional[NORM_TYPE],
        use_edges_info : bool,
        update_edges : bool,
        activation : str,
        gnn_depth : int,
        mlp_depth : int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.use_edges_info = use_edges_info
        self.update_edges = update_edges
        self.activation = activation
        self.mlp_depth = mlp_depth

        self.gnn = TransEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=[out_channels] * (gnn_depth),
            variant='conv',
            activation=activation,
            norm=norm,
            shortcut=True,
            jk=True,
            mlp=False,
            use_edges_info=use_edges_info,
            update_edges=self.update_edges
        )

        self.mlp = MLP(
            input_dim=out_channels,
            channels=[out_channels] * (mlp_depth - 1),
            output_dim=out_channels,
            hidden_activation=activation,
            norm=self.norm,
            output_activation=None
        ) if mlp_depth > 1 else nn.Identity()

    def forward(self, x : Tensor, edge_index : Tensor, edge_attr : Optional[Tensor]) -> Tensor:
        out = self.gnn.forward({ 'x' : x, 'edge_index' : edge_index, 'edge_attr' : edge_attr })
        x = self.mlp.forward(out)
        return x

@TopicModel.register
class GraphUNet(TopicModel):

    class Config(BaseModel):
        hidden_channels : List[int] = Field(default_factory=list)
        gnn_depth : int = 2
        mlp_depth : int = 0
        out_channels : int = 256
        use_edges_info : bool = False
        activation : Optional[str] = 'relu'
        norm : Optional[NORM_TYPE] = 'batchnorm'
        pooling_layer : POOLING_KIND = 'dmon'
        norm_pool : bool = False
        num_clusters : List[int] = Field(default_factory=lambda x : [256])
        assign_nn : ASSIGN_NN = 'gnn'
        unpooling_type : UNPOOL_KIND = 'matmul'

    @dataclass
    class Output:
        xd_hat : Tensor
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
                "doc_rec_loss" : losses.CosineRecLossContext(
                    x1=batch['Article'].x,
                    y1=self.xd_hat,
                ),
                "aux_loss" : losses.Identity(value=self.aux_loss)
            }

    def __init__(self, 
        in_channels : int,
        hidden_channels : List[int],
        out_channels : int,
        use_edges_info : bool,
        activation : Optional[str],
        norm : Optional[NORM_TYPE],
        pooling_layer : POOLING_KIND,
        norm_pool : bool,
        num_clusters : List[int],
        assign_nn : ASSIGN_NN,
        gnn_depth : int,
        mlp_depth : int,
        unpooling_type : UNPOOL_KIND
    ) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.use_edges_info = use_edges_info
        self.activation = activation
        self.norm = norm
        self.pooling_layer : POOLING_KIND = pooling_layer
        self.norm_pool = norm_pool
        self.num_clusters = num_clusters
        self.assign_nn = assign_nn
        self.gnn_depth = gnn_depth
        self.mlp_depth = mlp_depth
        self.unpooling_type : UNPOOL_KIND = unpooling_type

        self.encoder_layers : List[GraphUNetBlock] = nn.ModuleList(self.create_encoder_layers())
        self.decoder_layers : List[GraphUNetBlock] = nn.ModuleList(self.create_decoder_layers())
        self.pooling_layers : List[Pool] = nn.ModuleList(self.create_pooling_layers())

        self.reduce = Reduction(
            kind='attn',
            weights_nn=nn.Sequential(
                nn.Linear(in_features=out_channels, out_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(in_features=out_channels, out_features=1)
            )
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

    def create_encoder_layers(self) -> List[GraphUNetBlock]:

        channels = [self.in_channels, *self.hidden_channels, self.out_channels]

        return [
            GraphUNetBlock(
               in_channels=in_channels,
               out_channels=out_channels,
               norm=self.norm,
               use_edges_info=self.use_edges_info,
               activation=self.activation,
               gnn_depth=self.gnn_depth,
               mlp_depth=self.mlp_depth,
               update_edges=True
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:]))
        ]

    def create_decoder_layers(self) -> List[TransGNN]:
    
        channels = [self.out_channels, *self.hidden_channels[::-1], self.out_channels]

        return [
            GraphUNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=self.activation,
                norm=self.norm,
                use_edges_info=False,
                gnn_depth=self.gnn_depth,
                mlp_depth=self.mlp_depth,
                update_edges=(self.pooling_layer == 'vq')
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
                use_edges_info=False,
                **TransEncoder.__trans_gnn__
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
                in_channels=in_channels,
                norm=self.norm_pool,
                nn=self.create_assign_nn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    use_edges_info=self.use_edges_info
                ) if self.pooling_layer != 'vq' else None,
                num_prototypes=out_channels,
                unpool_kind=self.unpooling_type
            )
            for i, (in_channels, out_channels) in enumerate(zip(channels[1:], self.num_clusters))
        ]

    def encode(self, 
        x : Tensor,
        edge_attr : Optional[Tensor],
        EE : AdjMatrix,
        EA : AdjMatrix   
    ) -> List[ClusterPoolOutput]:

        outputs = []

        for encoder_layer, pooling_layer in zip(self.encoder_layers, self.pooling_layers):

            out = encoder_layer.forward(
                x=x,
                edge_index=EE,
                edge_attr=edge_attr
            )
            
            out = pooling_layer.pool(
                x=out,
                EE=EE,
                EA=EA
            )

            x = out.x
            EE = out.EE
            EA = out.EA

            outputs.append(out)

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

            x = encoder_layer.forward(
                x=out.x,
                edge_index=out.EE,
                edge_attr=None
            )
    
            out.x = x
            EE = out.EE
            EA = out.EA

            outputs.append(out)

        return outputs

    def forward(self, 
        data : HeteroData,
        ctx : Optional[IterationContext] = None
    ) -> None:

        x = data['Entity'].x

        ED = data['Entity', 'APPEARIN', 'Article'].edge_index
        N = data['Article'].x.size(0)

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
                num_nodes=(x.size(0), N)
            )
        )

        pooling_outputs = self.encode(
            x=x, 
            edge_attr=data['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr,
            EE=EE, 
            EA=EA
        )

        unpooling_outputs = self.decode(
            x=pooling_outputs[-1].x, 
            EE=pooling_outputs[-1].EE, 
            EA=pooling_outputs[-1].EA, 
            outs=pooling_outputs
        )

        xd_hat, weights, _ = self.reduce.forward(
            x=unpooling_outputs[-1].x,
            edge_index=ED,
            num_nodes=N
        ).astuple()

        return self.Output(
            xd_hat=xd_hat,
            x_hidden=pooling_outputs[-1].x,
            x_hat=unpooling_outputs[-1].x,
            S=[o.S for o in pooling_outputs],
            aux_loss=sum(out.aux_loss for out in pooling_outputs if out.aux_loss is not None)
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
