from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.distributions.categorical import Categorical

from MARL.utils.debug_utils import checknan, checknan_Sequential, analyze


def layer_init(layer: nn.Linear, method: str = 'xavier', **kwargs) -> nn.Linear:
    # TODO: move this to utils
    if method == 'xavier':
        nn.init.xavier_uniform_(layer.weight)
    elif method == 'orthogonal':
        std = kwargs.get('std', np.sqrt(2))
        nn.init.orthogonal_(layer.weight, std)
    else:
        raise ValueError("Invalid initialization method")
    nn.init.zeros_(layer.bias)
    return layer
