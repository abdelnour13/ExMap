import torch
import math
from torch import Tensor
from torch import nn
from typing import Tuple, List, Optional
from src.graph_dl import ops
from torch.nn import functional as F
from functools import reduce
from dataclasses import dataclass

@dataclass
class CodebookOutput:
    S : Tensor
    q : Tensor
    codebook_loss : Tensor
    commitement_loss : Tensor

def prod(iterable):
    return reduce(lambda a,b : a*b, iterable)

class MultiLevelCodebook(nn.Module):

    def __init__(
        self,
        channels: int,
        levels: Tuple[int, ...],
    ):
        super().__init__()

        if not levels or any(k <= 0 for k in levels):
            raise ValueError("levels must contain positive codebook sizes")

        self.channels = channels
        self.levels = levels

        self.codebooks = nn.ParameterList()

        for i in range(len(levels)):
            codebook = nn.Parameter(
                torch.empty(*levels[:i + 1], channels)
            )
            self.codebooks.append(codebook)

        self.reset_parameters()

    def reset_parameters(self):

        for codebook in self.codebooks:
            nn.init.kaiming_uniform_(codebook, a=math.sqrt(5))

    def assign(
        self,
        x: Tensor,
        codebook: Tensor,
        temp: float | Tensor = 1.0,
        indices: Optional[Tensor] = None,
    ) -> Tensor:
        
        flat_codebook = codebook.flatten(end_dim=-2)
        S = -ops.dist(x, flat_codebook.detach())

        if indices is not None:

            num_children = codebook.size(-2)

            parent_ids = (
                torch.arange(
                    flat_codebook.size(0),
                    device=x.device,
                )
                // num_children
            )

            valid = parent_ids.unsqueeze(0) == indices.unsqueeze(-1)
            S = S.masked_fill(~valid, -torch.inf)

        S = torch.softmax(S / temp, dim=-1)
        
        return S

    def STE(
        self,
        codebook: Tensor,
        x: Tensor,
        assignments: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        indices = assignments.argmax(dim=-1)

        # Selected codeword vectors; gradients update the codebook.
        q = codebook.flatten(end_dim=-2)[indices]

        # Straight-through estimator.
        x_quantized = x + (q - x).detach()

        codebook_loss = F.mse_loss(q, x.detach())
        commitment_loss = F.mse_loss(q.detach(), x)

        return x_quantized, codebook_loss, commitment_loss

    def forward(
        self,
        x: Tensor,
        temp: float | Tensor = 1.0,
    ) -> List["CodebookOutput"]:
        
        outputs = []
        residual = x
        indices = None

        for codebook in self.codebooks:
            
            assignments = self.assign(
                x=residual,
                codebook=codebook,
                temp=temp,
                indices=indices,
            )

            q, codebook_loss, commitment_loss = self.STE(
                codebook=codebook,
                x=residual,
                assignments=assignments,
            )

            indices = assignments.argmax(dim=-1)

            outputs.append(
                CodebookOutput(
                    S=assignments,
                    q=q,
                    codebook_loss=codebook_loss,
                    commitement_loss=commitment_loss,
                )
            )

            # Critical: subtract the quantized value without
            # propagating gradients through this subtraction.
            residual = residual - q.detach()

        return outputs

if __name__ == '__main__':

    N, D = 16, 64
    x = torch.randn((N, D))
    levels = [8, 4]

    codebook = MultiLevelCodebook(channels=D, levels=levels)
    outputs = codebook.forward(x)