import torch
from torch import Tensor

import inspect

def checknan(tensor: Tensor, print_when_false: bool = False) -> bool:
    """Check if tensor contains NaN, also outputs the checker results"""
    caller_locals = inspect.currentframe().f_back.f_locals
    name = [var_name for var_name, var in caller_locals.items() if var is tensor][0]
    
    if torch.isnan(tensor).any():
        print(f"{name} has NaN")
        return True
    if print_when_false:
        print(f"{name} is ok")
    return False
