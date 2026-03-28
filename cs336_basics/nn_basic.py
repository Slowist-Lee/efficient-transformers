import torch
import math
from einops import einsum

class Linear(torch.nn.Module):

    def __init__(self,in_features,out_features,device,dtype):

        super().__init__()

        self.in_features=in_features
        self.out_features=out_features

        self.weight=torch.nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))

        sigma=math.sqrt(2/(in_features+out_features))

        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=sigma, 
            a=-3*sigma, 
            b=3*sigma
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y
