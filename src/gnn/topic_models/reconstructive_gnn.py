import torch
from pydantic import BaseModel, BeforeValidator
from src.gnn.layers.encoder import TransEncoder
from src.gnn.layers.reduction import Reduction, READOUT
from torch import nn, Tensor
from typing import Literal, List, Optional, Tuple, Dict, Annotated
from torch_geometric.data import HeteroData
from src.gnn import ops, losses
from src.gnn.topic_models.output import BaseLossContext
from dataclasses import dataclass

ENCODER_TYPE = Literal['gnn', 'mlp']
MODE = Literal['one', 'multi', 'topk']
READOUT_MODE = Literal['pre', 'post']

@dataclass
class ReconstructiveModelOutput:
    entities_embeddings : Tensor
    articles_embeddings : Tensor
    entities_prototypes : Tensor
    attn_weights : Tensor
    h : Tensor
    h_quan : Tensor
    codebook_loss : Tensor
    commitement_loss : Tensor
    temp : Tensor
    codebook : Tensor

    def context(self, batch : HeteroData, batch_size : int) -> Dict[str, BaseLossContext]:

        return {
            'consine_rec_loss' : losses.CosineRecLossContext(
                a=batch['Article'].x,
                b=self.articles_embeddings
            ),
            'codebook_loss' : losses.Identity(value=self.codebook_loss),
            'commitement_loss' : losses.Identity(value={
                "loss"  :self.commitement_loss,
                "temp" : self.temp
            }),
        }

class ReconstructiveModelConfig(BaseModel):
    encoder_type : ENCODER_TYPE = 'mlp'
    decoder_type : ENCODER_TYPE = 'gnn'
    hidden_channels : Optional[List[int]] = None
    out_channels : int = 256
    decoder_output_channel : Optional[int] = None
    variant : Literal['conv', 'attn'] = 'conv'
    activation : str = 'relu'
    norm : bool = True
    shortcut: bool = False
    shortcut_dim: Optional[int]  = None
    jk: bool = True
    num_prototypes : int = 512
    ste_mode : MODE = 'one'
    readout : READOUT = 'mean'
    temps : Annotated[Tuple[float, float], BeforeValidator(tuple)] = (2.0, 0.3)

class ReconstructiveModel(nn.Module):

    def __init__(
        self,
        encoder_type : ENCODER_TYPE,
        decoder_type : ENCODER_TYPE,
        in_channels  : int,
        total_iterations : int,
        hidden_channels : List[int],
        out_channels : int,
        decoder_output_channel : Optional[int],
        num_relations : int,
        variant : Literal['conv', 'attn'],
        activation : str,
        norm : bool,
        shortcut: bool,
        shortcut_dim: Optional[int] ,
        jk: bool ,
        num_prototypes : int,
        ste_mode : Optional[MODE],
        readout : READOUT,
        temps : Tuple[float, float],
    ) -> None:

        super().__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.variant = variant
        self.activation = activation
        self.norm = norm
        self.shortcut = shortcut
        self.shortcut_dim = shortcut_dim
        self.jk = jk
        self.num_prototypes = num_prototypes
        self.ste_mode = ste_mode
        self.decoder_output_channel = decoder_output_channel
        self.readout = readout
        self.total_iterations = total_iterations
        self.temps = temps

        self.encoder = TransEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_relations=num_relations,
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(encoder_type == 'mlp')
        )

        self.decoder = TransEncoder(
            in_channels=out_channels,
            out_channels=decoder_output_channel or in_channels,
            hidden_channels=hidden_channels[::-1],
            num_relations=num_relations,
            variant=variant,
            activation=activation,
            norm=norm,
            shortcut=shortcut,
            shortcut_dim=shortcut_dim,
            jk=jk,
            mlp=(decoder_type == 'mlp'),
            use_edges_info=False
        )

        self.C = nn.Parameter(torch.randn(num_prototypes, out_channels), requires_grad=True)

        self.register_buffer(
            "ema_count",
            torch.zeros(num_prototypes)
        )

        self.register_buffer(
            "ema_sum",
            torch.zeros(num_prototypes, out_channels)
        )

        self.reduce = Reduction(
            kind=readout,
            weights_nn=nn.Linear(
                in_features=(decoder_output_channel or in_channels),
                out_features=1
            )
        )

        s = self.temps[0]
        self.temp = nn.Parameter(torch.scalar_tensor(s), requires_grad=False)
        self.iter = 0
        self.decay = 0.99

    @torch.no_grad()
    def update_centers(self, z, assignments):

        if not self.training:
            return

        indices = assignments.argmax(dim=-1)

        count = torch.bincount(
            indices,
            minlength=self.num_prototypes
        ).to(z.dtype)

        # [K, D]
        z_sum = torch.zeros_like(self.ema_sum)
        z_sum.index_add_(0, indices, z)

        # EMA
        self.ema_count.mul_(self.decay).add_(
            count,
            alpha=1 - self.decay
        )

        self.ema_sum.mul_(self.decay).add_(
            z_sum,
            alpha=1 - self.decay
        )

        # Update centers
        self.C.data.copy_(
            self.ema_sum /
            (self.ema_count.unsqueeze(1) + 1e-6)
        )

    @staticmethod
    def from_config(
        in_channels : int,
        num_relations : Tuple[int, int],
        total_iterations : int,
        config : ReconstructiveModelConfig                
    ) -> 'ReconstructiveModel':
        
        return ReconstructiveModel(
            in_channels = in_channels,
            num_relations = num_relations,
            total_iterations=total_iterations,
            **config.model_dump()
        )

    def update_temp(self) -> None:

        if not self.training:
            return

        t0, t_f = self.temps
        t0 = torch.scalar_tensor(t0)
        t_f = torch.scalar_tensor(t_f)

        self.temp.data = t0 * (t_f / t0) ** (self.iter / self.total_iterations)
        self.iter += 1

    def ste(self, h : Tensor, q : Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.ste_mode == 'one':
        
            h_quan, codebook_loss, commitement_loss = ops.STE(
                codebook=self.C.data,
                x=h,
                assignments=q,
                # temp=self.temp.data,
                # hard=False
            )

        elif self.ste_mode == 'multi':

            h_quan, codebook_loss, commitement_loss = ops.mSTE(
                codebook=self.C.data,
                x=h,
                assignments=q,
                threshold=0,
                weighted=False
            )

        elif self.ste_mode == 'topk':

            h_quan, codebook_loss, commitement_loss = ops.topkSTE(
                codebook=self.C.data,
                x=h,
                assignments=q,
                k=5,
                weighted=False
            )

        else:
            assert False, f"self.ste_mode = '{self.ste_mode}' is not allowed."

        self.update_temp()
        # self.update_centers(h, q)
        return h_quan, codebook_loss, commitement_loss

    def softSTE(self, h : Tensor, k : int) -> Tensor:

        h_quan = []

        for _ in range(k):

            h_norm = h / h.norm(dim=-1, keepdim=True, p=2)
            C_norm = self.C.data / self.C.data.norm(dim=-1, keepdim=True, p=2)
            entities_prototypes = torch.softmax((h_norm @ C_norm.t()) / self.temp, dim=-1)
            h_quan_current = entities_prototypes @ self.C
            h_quan.append(h_quan_current)

            h = h - h_quan_current

        h_quan = torch.stack(h_quan, dim=0).max(dim=0)

        self.update_temp()

    def forward(self, data : HeteroData):

        KG = {
            'x' : data['Entity'].x, 
            'edge_attr' : data['Entity', 'HASRELATIONWITH', 'Entity'].edge_attr, 
            'edge_index' : data['Entity', 'HASRELATIONWITH', 'Entity'].edge_index
        }

        ED = data['Entity', 'APPEARIN', 'Article'].edge_index
        N = data['Article'].x.size(0)

        ### Encoder
        h = self.encoder(KG)

        ### Assignments
        h_norm = h / h.norm(dim=-1, keepdim=True, p=2)
        C_norm = self.C.data / self.C.data.norm(dim=-1, keepdim=True, p=2)
        entities_prototypes = torch.softmax((h_norm @ C_norm.t()) / self.temp, dim=-1)
        self.update_temp()

        ### Quantization
        # h_quan, codebook_loss, commitement_loss = self.ste(h, entities_prototypes)
        codebook_loss = torch.scalar_tensor(0.0).to(h.device)
        commitement_loss = torch.scalar_tensor(0.0).to(h.device)
        h_quan = entities_prototypes @ self.C

        ### Decoder
        KG['x'] = h_quan # h_quan
        xe_hat = self.decoder(KG)

        ### Readout
        xd_hat, weights, _ = self.reduce.forward(
            x=xe_hat,
            edge_index=ED,
            num_nodes=N
        ).astuple()

        return ReconstructiveModelOutput(
            entities_embeddings=xe_hat,
            articles_embeddings=xd_hat,
            entities_prototypes=entities_prototypes,
            attn_weights=weights,
            h=h,
            h_quan=h_quan,
            codebook_loss=codebook_loss,
            commitement_loss=commitement_loss,
            temp=self.temp.data,
            codebook=self.C.data
        )