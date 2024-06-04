import torch
import torch.nn as nn
from torch import Tensor

# scroll to bottom for templates

# set DEBUG to True to run debug funtions when called.
DEBUG = True


def printd(*args, **kwargs):
    """A wrapper for the print function"""
    if DEBUG:
        print(*args, **kwargs)


def analyze(tensor: Tensor):
    """Analyizes a Tensor.
    prints 4 numbers indicating (positive_count, negative_count, zero_count, nan_count)
    of the given tensor
    """
    if DEBUG is False:
        return
    positive_count, negative_count, zero_count, nan_count = torch.sum(tensor > 0).item(), torch.sum(tensor < 0).item(), torch.sum(tensor == 0).item(), torch.sum(torch.isnan(tensor)).item()
    print(positive_count, negative_count, zero_count, nan_count)


def checknan_Sequential(network: nn.Sequential, exit_when_NaN: bool = True):
    """check if any nn.Sequential parameters and gradients contains NaN"""
    if DEBUG is False:
        return
    contains_nan = False
    for name, param in network.named_parameters():
        if torch.isnan(param).any():
            contains_nan = True
            print(f"Parameter {name} contains NaNs")
        if param.grad is not None and torch.isnan(param.grad).any():
            contains_nan = True
            print(f"Gradient of {name} contains NaNs")
    if contains_nan is False:
        print(f"No parameter contains NaNs")
    else:
        if exit_when_NaN:
            exit(0)


def checknan(**kwargs) -> bool:
    """Check if tensor contains NaN, also outputs the checker results
    
    Example:
        >>> checknan(InputTensor=tensor([0, 1]), print_when_false=True)
        InputTensor is ok
    """
    if DEBUG is False:
        return False

    assert (1 <= len(kwargs) <= 2)
    print_when_false = kwargs['print_when_false'] if 'print_when_false' in kwargs else False

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

# Checknan checklist for the loss:

# if checknan(loss=loss, print_when_false=True):
#     if checknan(pg_loss=pg_loss):
#         if checknan(pg_loss1=pg_loss1) or checknan(pg_loss2=pg_loss2):
#             if checknan(mb_advantages=mb_advantages, print_when_false=True):
#                 printd(mb_advantages.max(), mb_advantages.min(), mb_advantages.mean(), mb_advantages.std())
#             checknan(preNorm_b_advantages=b_advantages[mb_inds], print_when_false=True)
#             checknan(ratio=ratio, print_when_false=True)
#             checknan(logratio=logratio, print_when_false=True)
#     checknan(v_loss=v_loss)
#     checknan(entropy_loss=entropy_loss)
#     exit(0)
