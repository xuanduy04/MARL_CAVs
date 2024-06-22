import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    """
    A wrapper class for nn.Linear
    Initialize values using xavier_uniform_
    """
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        forward pass through linear layer.
        """
        return self.linear(x)
