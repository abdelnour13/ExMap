import torch
import torch_scatter
from torch import nn, Tensor
from typing import Literal, Optional, TypedDict
from torch_geometric import utils as gnn_utils
from src.graph_learning.layers.activation import Activation
from src.graph_learning.layers.norm import Norm, NORM_TYPE
from src.graph_learning.utils import AdjMatrix, SparseTensor, EdgeIndex
from src import helpers

class TransGNNOutput(TypedDict):
    x : Tensor
    edge_attr : Optional[Tensor]

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
        norm : Optional[NORM_TYPE] = None,
        update_edges : bool = True
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
        self.update_edges = update_edges

        self.kg_scoring = self.create_scoring_fn(name = kg_scoring_fn)

        ### Nodes
        if combine == 'sum':
            self.W_node = nn.Linear(in_channels, out_channels, bias = bias)
            self.res = nn.Linear(in_channels, out_channels, bias = bias)
        else:
            self.W_node = nn.Linear(2 * in_channels, out_channels, bias = bias)

        self.W_norm = Norm(type=norm, num_features=out_channels)

        ### Edges
        if self.update_edges:
            m =  2 + int(use_edges_info)
            self.W_edge = nn.Linear(m * in_channels, out_channels, bias = bias)
            self.E_norm = Norm(type=norm, num_features=out_channels)

        ### Activation
        self.act = Activation(activation)


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

    def calculate_attn_weights(
        self,
        x: Tensor,
        adj: AdjMatrix,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:

        if self.variant != 'attn':
            return None

        adj = adj.get_default()

        if isinstance(adj, Tensor):
            return self._dense_calculate_attn_weights(x, adj, edge_attr)
        elif isinstance(adj, SparseTensor):
            return self._sparse_calculate_attn_weights(x, adj, edge_attr)
        elif isinstance(adj, EdgeIndex):
            return self._edge_index_calculate_attn_weights(x, adj, edge_attr)

        raise TypeError(f"Unsupported adjacency representation: {type(adj).__name__}")


    def _dense_calculate_attn_weights(
        self,
        x: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError(
            "Attention weights are not available for the dense adjacency representation; "
            "use an AdjMatrix backed by a SparseTensor or an EdgeIndex."
        )


    def _sparse_calculate_attn_weights(
        self,
        x: Tensor,
        adj: SparseTensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        # Reads the SparseTensor's own COO storage (no conversion to another representation).
        # NOTE: entries are in SparseTensor order (sorted by row, then col),
        # so edge_attr must follow that same order.
        row, col, _ = adj.coo()
        return self._row_col_calculate_attn_weights(x, row, col, edge_attr)


    def _edge_index_calculate_attn_weights(
        self,
        x: Tensor,
        adj: EdgeIndex,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        return self._row_col_calculate_attn_weights(x, adj.row, adj.col, edge_attr)


    def _row_col_calculate_attn_weights(
        self,
        x: Tensor,
        row: Tensor,
        col: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        
        h, t = x[row], x[col]

        if self.use_edges_info:

            if edge_attr is None:
                raise ValueError("edge_attr is required when use_edges_info=True.")
            
            r = edge_attr
            r_inv = self.kg_scoring.inv_rel(r)
            attn_weights1 = self.attn(torch.cat([h, t, r], dim=-1))
            attn_weights2 = self.attn(torch.cat([t, h, r_inv], dim=-1))

        else:
            attn_weights1 = self.attn(torch.cat([h, t], dim=-1))
            attn_weights2 = self.attn(torch.cat([t, h], dim=-1))

        return gnn_utils.softmax(
            src=torch.cat([attn_weights1, attn_weights2], dim=0),
            index=torch.cat([row, col]),
            num_nodes=x.size(0),
            dim=0,
        )
    
    def message_and_aggregate(
        self,
        x: Tensor,
        adj: AdjMatrix,
        edge_attr: Optional[Tensor] = None,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:

        adj = adj.get_default()

        if isinstance(adj, Tensor):
            return self._dense_message_and_aggregate(x, adj, edge_attr, attn_weights)
        elif isinstance(adj, SparseTensor):
            return self._sparse_message_and_aggregate(x, adj, edge_attr, attn_weights)
        elif isinstance(adj, EdgeIndex):
            return self._edge_index_message_and_aggregate(x, adj, edge_attr, attn_weights)

        raise TypeError(f"Unsupported adjacency representation: {type(adj).__name__}")


    def _dense_message_and_aggregate(
        self,
        x: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:

        if attn_weights is not None:
            raise ValueError(
                "attn_weights cannot be used with the dense adjacency representation."
            )
        assert not self.use_edges_info, (
            "use_edges_info is not supported with the dense adjacency representation."
        )

        adj = adj.to(x.dtype)  # no-op if already the right dtype

        # Node i receives:
        #   in_ messages:  sum_j A[i, j] * x_j  ->  A   @ x
        #   out messages:  sum_j A[j, i] * x_j  ->  A^T @ x   (A.t() is a view, no copy)
        num = adj @ x
        num = num + adj.t() @ x

        # Sum of weights per node: row sums (as head) + column sums (as tail)
        den = adj.sum(dim=1, keepdim=True)
        den += adj.sum(dim=0).unsqueeze(1)
        den = den.masked_fill(den == 0, 1)

        return num / den  # weighted mean, 0 for isolated nodes


    def _sparse_message_and_aggregate(
        self,
        x: Tensor,
        adj: SparseTensor,
        edge_attr: Optional[Tensor] = None,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:
        # Native COO storage of the SparseTensor (sorted by row, then col):
        # edge_attr and attn_weights must follow this order.
        row, col, value = adj.coo()
        return self._row_col_message_and_aggregate(
            x, row, col, value, edge_attr, attn_weights
        )


    def _edge_index_message_and_aggregate(
        self,
        x: Tensor,
        adj: EdgeIndex,
        edge_attr: Optional[Tensor] = None,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:
        return self._row_col_message_and_aggregate(
            x, adj.row, adj.col, adj.value, edge_attr, attn_weights
        )


    def _row_col_message_and_aggregate(
        self,
        x: Tensor,
        row: Tensor,
        col: Tensor,
        value: Optional[Tensor] = None,
        edge_attr: Optional[Tensor] = None,
        attn_weights: Optional[Tensor] = None,
    ) -> Tensor:

        ### Head, tail and relation
        h, t = x[row], x[col]

        if self.use_edges_info:
            if edge_attr is None:
                raise ValueError("edge_attr is required when use_edges_info=True.")
            r = edge_attr
            out = self.kg_scoring.translate(h, r)
            in_ = self.kg_scoring.inv_translate(t, r)
        else:
            out = h
            in_ = t

        ### Messages: in_ goes to row, out goes to col
        src = torch.cat([in_, out], dim=0)
        index = torch.cat([row, col], dim=0)
        num_nodes = x.size(0)

        ### Computed attention weights: already softmax-normalised -> weighted sum
        if attn_weights is not None:

            return torch_scatter.scatter(
                src=src * attn_weights,
                index=index,
                dim=0,
                dim_size=num_nodes,
                reduce='sum',
            )

        ### No edge weights: one where there is an edge -> plain mean
        if value is None:

            messages = torch_scatter.scatter(
                src=src,
                index=index,
                dim=0,
                dim_size=num_nodes,
                reduce='mean',
            )

            if not torch.isfinite(messages).all():
                raise Exception("Messages are non-finite")

            return messages

        ### Adjacency weights -> weighted mean
        w = torch.cat([value, value], dim=0).to(src.dtype).view(-1, 1)

        num = torch_scatter.scatter(src * w, index, dim=0, dim_size=num_nodes, reduce='sum')
        den = torch_scatter.scatter(w, index, dim=0, dim_size=num_nodes, reduce='sum')

        return num / den.masked_fill(den == 0, 1)
    
    def _node_update(self,
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

    def node_update(self, x, messages : Tensor):

        if self.combine == 'sum':
            z = self.W_node(messages) + self.res(x)
        else:
            z = self.W_node(torch.cat([x, messages], dim=-1))

            if not torch.isfinite(z).all():
                raise RuntimeError(f"""
                    Non-finite after W_node / residual | {messages.isfinite().all()} | {x.isfinite().all()} |
                    {self.W_node.bias.isfinite().all()} | {self.W_node.weight.isfinite().all()} | {self.W_node.weight.min()} |
                    {self.W_node.weight.max()}
                """)

        z = self.W_norm(z)
        if not torch.isfinite(z).all():
            raise RuntimeError("Non-finite after W_norm")

        z = self.act(z)
        if not torch.isfinite(z).all():
            raise RuntimeError("Non-finite after activation")

        return z
    
    def edge_update(
        self,
        x: Tensor,
        adj: AdjMatrix,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:

        adj = adj.get_default()

        if isinstance(adj, Tensor):
            return self._dense_edge_update(x, adj, edge_attr)
        elif isinstance(adj, SparseTensor):
            return self._sparse_edge_update(x, adj, edge_attr)
        elif isinstance(adj, EdgeIndex):
            return self._edge_index_edge_update(x, adj, edge_attr)

        raise TypeError(f"Unsupported adjacency representation: {type(adj).__name__}")


    def _dense_edge_update(
        self,
        x: Tensor,
        adj: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        assert not self.update_edges, (
            "update_edges is not supported with the dense adjacency representation."
        )
        return None


    def _sparse_edge_update(
        self,
        x: Tensor,
        adj: SparseTensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        
        if not self.update_edges:
            return None
        # Native COO storage (sorted by row, then col): edge_attr must follow this order,
        # and the returned edge features are in this order too.
        row, col, _ = adj.coo()
        return self._row_col_edge_update(x, row, col, edge_attr)


    def _edge_index_edge_update(
        self,
        x: Tensor,
        adj: EdgeIndex,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if not self.update_edges:
            return None
        return self._row_col_edge_update(x, adj.row, adj.col, edge_attr)


    def _row_col_edge_update(
        self,
        x: Tensor,
        row: Tensor,
        col: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:

        h, t = x[row], x[col]

        if self.use_edges_info:

            if edge_attr is None:

                raise ValueError("edge_attr is required when use_edges_info=True.")
            
            e = self.W_edge(torch.cat([h, t, edge_attr], dim=-1))

        else:
            e = self.W_edge(torch.cat([h, t], dim=-1))

        e = self.E_norm(e)
        e = self.act(e)

        return e
    
    def forward(self,
        x : Tensor,
        edge_index : AdjMatrix | Tensor,
        edge_attr : Optional[Tensor] = None            
    ) -> TransGNNOutput:

        adj = edge_index

        if isinstance(adj, Tensor):

            row, col = adj
            N = x.size(0)

            adj = AdjMatrix(
                _edge_index=EdgeIndex(
                    row=row,
                    col=col,
                    num_nodes=(N, N)
                )
            )

        assert (edge_attr is not None) or (not self.use_edges_info)
        
        attn_weights = self.calculate_attn_weights(
            x,
            adj,
            edge_attr
        )

        messages = self.message_and_aggregate(
            x,
            adj,
            edge_attr,
            attn_weights
        )

        if not torch.isfinite(messages).all():
            raise Exception("Messages are non-finite")

        edge_attr = self.edge_update(
            x,
            adj,
            edge_attr
        )

        x = self.node_update(x, messages)

        if x.isnan().sum():
            print(f"{type(edge_index.get_default())}")
            print(f"{edge_index.edge_index.value is None}")

        return TransGNNOutput(x=x,edge_attr=edge_attr)
    
if __name__ == '__main__':

    G = helpers.create_toy_graph(
        num_entities=50,
        num_articles=10,
        num_features=16,
        ee=100,
        ea=40
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
                    x = G["xe"],
                    edge_index = G["EE"],
                    edge_attr = G["edge_attr"]
                )

                print(f"x.size() = {out['x'].size()}")
                print(f"e.size() = {out['edge_attr'].size()}")

                print("\n")