import torch
import torch_scatter
from torch import nn, Tensor
from typing import Literal, Optional, TypedDict
from torch_geometric import utils as gnn_utils
from src.graph_dl.layers.activation import Activation
from src import helpers

class TransGNNOutput(TypedDict):
    x : Tensor
    edge_attr : Tensor

KGScoringFn = Literal['TransE', 'RotatE', 'DistMult']

class KGScoring(nn.Module):

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        raise NotImplementedError
    
    def inv_rel(self, rel : Tensor) -> Tensor:
        NotImplementedError
    
    def inv_translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        return self.translate(x, self.inv_rel(rel))
    
    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        raise NotImplementedError
    
    def forward(self,
        x : Tensor,
        y : Tensor,
        rel : Tensor
    ) -> Tensor:
        x = self.translate(x, rel)
        return self.score(x, y)
    
class TransE(KGScoring):

    def __init__(self,
        p_norm : float = 2.0             
    ) -> None:
        super().__init__()
        self.p_norm = p_norm

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        return x + rel

    def inv_rel(self, rel : Tensor) -> Tensor:
        return - rel

    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        return torch.norm(x - y, p = self.p_norm, dim = 1)
    
class RotatE(KGScoring):

    def __init__(self,
        p_norm : float = 2.0
    ) -> None:
        super().__init__()
        self.p_norm = p_norm
        

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        
        x_im, x_re = torch.chunk(x, 2, dim = -1)
        rel_im, rel_re = torch.chunk(rel, 2, dim = -1)
        
        y_re = x_re * rel_re - x_im * rel_im
        y_im = x_re * rel_im + x_im * rel_re

        return torch.cat([y_im, y_re], dim = -1)
    
    def inv_rel(self, rel : Tensor) -> Tensor:
        rel_im, rel_re = torch.chunk(rel, 2, dim = -1)
        return torch.cat([-rel_im, rel_re], dim = -1)
    
    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        return torch.norm(x - y, p = self.p_norm, dim = 1)
    
class DistMult(KGScoring):

    def __init__(self) -> None:
        super().__init__()

    def translate(self,
        x : Tensor,
        rel : Tensor
    ) -> Tensor:
        return x * rel

    def inv_rel(self, rel : Tensor) -> Tensor:
        return rel.flip(dims = (-1,))

    def score(self,
        x : Tensor,
        y : Tensor
    ) -> Tensor:
        scores = torch.bmm(x.unsqueeze(1), y.unsqueeze(-1))
        scores = torch.squeeze(scores)
        return scores

class TransGNN(nn.Module):

    def __init__(self, 
        in_channels : int,
        out_channels : int,
        use_edges_info : bool = True,
        kg_scoring_fn : KGScoringFn = 'TransE',
        variant : Literal['attn','conv'] = 'conv',
        combine : Literal['cat', 'sum'] = 'cat',
        bias : bool = False,
        activation : Optional[str]  = 'relu',
        norm : bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_edges_info = use_edges_info
        self.kg_scoring_fn = kg_scoring_fn
        self.variant = variant
        self.combine = combine
        self.bias = bias
        self.activation = activation
        self.norm = norm

        self.kg_scoring = self.create_scoring_fn(name = kg_scoring_fn)

        ### Nodes
        if combine == 'sum':
            self.W_node = nn.Linear(in_channels, out_channels, bias = bias)
            self.res = nn.Linear(in_channels, out_channels, bias = bias)
        else:
            self.W_node = nn.Linear(2 * in_channels, out_channels, bias = bias)

        ### Edges
        m =  2 + int(use_edges_info)
        self.W_edge = nn.Linear(m * in_channels, out_channels, bias = bias)

        ### Activation
        self.act = Activation(activation)

        ### Normalization
        self.W_norm = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.E_norm = nn.BatchNorm1d(out_channels) if norm else nn.Identity()

        if variant == 'attn':

            self.attn = nn.Sequential(
                nn.Linear(m * in_channels, in_channels, bias = False),
                nn.LeakyReLU(negative_slope = 0.2),
                nn.Linear(in_channels, 1, bias = False),
            )

    @staticmethod
    def create_scoring_fn(
        name : KGScoringFn,
        **kwargs
    ) -> KGScoring:
        
        if name == 'TransE':
            return TransE(**kwargs)
        elif name == 'RotatE':
            return RotatE(**kwargs)
        elif name == 'DistMult':
            return DistMult(**kwargs)
        else:
            raise ValueError(f"Unknown scoring function {name}")

    def calculate_attn_weights(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Tensor= None,
    ) -> Optional[Tensor]:

        if self.variant != 'attn':
            return None
        
        row, col = edge_index
        h, t = x[row], x[col]

        if self.use_edges_info:
            r = edge_attr
            r_inv = self.kg_scoring.inv_rel(r)
            attn_weights1 = self.attn(torch.cat([h, t, r], dim = -1))
            attn_weights2 = self.attn(torch.cat([t, h, r_inv], dim = -1))
        else:
            attn_weights1 = self.attn(torch.cat([h, t], dim = -1))
            attn_weights2 = self.attn(torch.cat([t, h], dim = -1))

        attn_weights = gnn_utils.softmax(
            src = torch.cat([attn_weights1, attn_weights2],dim=0),
            index = torch.cat([row, col]),
            num_nodes = x.size(0),
            dim = 0
        )

        return attn_weights
    
    def message_and_aggregate(self, 
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None,     
        attn_weights : Optional[Tensor] = None                   
    ) -> Tensor:
        
        row, col = edge_index

        ### Get head tail and relation
        h, t = x[row], x[col]

        if self.use_edges_info:
            r = edge_attr
            out = self.kg_scoring.translate(h, r) if self.use_edges_info else h
            in_ = self.kg_scoring.inv_translate(t, r) if self.use_edges_info else t
        else:
            out = h
            in_ = t

        ### Messages
        src = torch.cat([in_,out], dim = 0)

        if attn_weights is not None:
            src = src * attn_weights

        messages = torch_scatter.scatter(
            src = src,
            index = torch.cat([row, col], dim=0),
            dim = 0,
            dim_size = x.size(0),
            reduce = ('mean' if attn_weights is None else 'sum')
        )

        return messages
    
    def node_update(self,
        x : Tensor,
        messages : Tensor         
    ) -> Tensor:
        
        if self.combine == 'sum':
            x = self.W_node(messages) + self.res(x)
        else:
            x = self.W_node(torch.cat([x, messages],dim = -1))

        x = self.W_norm(x)
        x = self.act(x)

        return x
    
    def edge_update(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None                
    ) -> Tensor:
        
        row, col = edge_index
        h, t = x[row], x[col]
        r = edge_attr
        
        if self.use_edges_info:
            e = self.W_edge(torch.cat([h, t, r],dim = -1))
        else:
            e = self.W_edge(torch.cat([h,t],dim = -1))
        
        e = self.E_norm(e)
        e = self.act(e)

        return e
    
    def forward(self,
        x : Tensor,
        edge_index : Tensor,
        edge_attr : Optional[Tensor] = None            
    ) -> TransGNNOutput:

        assert (edge_attr is not None) or (not self.use_edges_info)
        
        attn_weights = self.calculate_attn_weights(
            x,
            edge_index,
            edge_attr
        )

        messages = self.message_and_aggregate(
            x,
            edge_index,
            edge_attr,
            attn_weights
        )

        edge_attr = self.edge_update(
            x,
            edge_index,
            edge_attr
        )

        x = self.node_update(x, messages)

        return TransGNNOutput(x=x,edge_attr=edge_attr)
    
if __name__ == '__main__':

    G = helpers.create_toy_graph(
        num_nodes = 10,
        num_edges = 20,
        num_types = 3,
        node_features = 16,
        edge_features = 16,
        type_features = 16
    )

    for combine in ['cat','sum']:

        for variant in ['conv', 'attn']:

            for use_edges_info in [True, False]:

                print(f" Combine = {combine}, Variant = {variant} :")

                model = TransGNN(
                    in_channels = 16,
                    out_channels = 8,
                    combine = combine,
                    variant=variant,
                    use_edges_info=use_edges_info
                )

                out = model.forward(
                    x = G["x"],
                    edge_index = G["edge_index"],
                    edge_attr = G["e"]
                )

                print(f"x.size() = {out['x'].size()}")
                print(f"e.size() = {out['edge_attr'].size()}")

                print("\n")