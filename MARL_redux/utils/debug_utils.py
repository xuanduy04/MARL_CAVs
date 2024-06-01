import torch
from torch import Tensor


def checknan(**kwargs) -> bool:
    """Check if tensor contains NaN, also outputs the checker results
    
    Example:
        >>> checknan(InputTensor=tensor([0, 1]), print_when_false=True)
        InputTensor is ok
    """
    tensor = None
    print_when_false = False
    assert(1 <= len(kwargs) <= 2)

    if 'print_when_false' in kwargs:
        print_when_false = kwargs['print_when_false']
        
    name = list(kwargs.keys())[0]
    tensor = kwargs[name]
        
    if tensor is None:
        raise ValueError("Tensor argument not found.")
    
    if torch.isnan(tensor).any():
        print(f"{name} has NaN")
        return True
    if print_when_false:
        print(f"{name} is ok")
    return False

