from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (section 3.2.1)

    Args:
        - dim (int): dimension of d_k or d_head
        - dropout_p (float): probability of dropout

    Input:
        - query (batch, num_heads, seq_len, d_head)
        - key   (batch, num_heads, seq_len, d_head)
        - value (batch, num_heads, seq_len, d_head)

    Output:
        - context (batch, num_head, seq_len, d_head): Context matrix.
        - attn (batch, num_head, seq_len, seq_len): Attention matrix for visualization.
    """
    def __init__(self, dim: int, dropout_p: float) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout = nn.Dropout(p = dropout_p)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        
        # (batch, num_heads, seq_len, d_head) @ (batch, num_heads, d_head, seq_len)
        # ==> score: (batch, num_heads, seq_len, seq_len)
        score = torch.matmul(query, key.transpose(-2, -1)) / self.sqrt_dim

        attn = F.softmax(score, -1) 
        # (batch, num_head, seq_len, seq_len)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, value)
        # (batch, num_head, seq_len, d_head)

        return context, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (section 3.2.2)

    Args:
        - d_model (int): dimension of model
        - num_heads (int): number of heads
        - dropout_p (float): probability of dropout

    Inputs:
        - query (batch, seq_len, d_model):
        - key   (batch, seq_len, d_model):
        - value (batch, seq_len, d_model):
        - mask ():

    Output: (Tensor, Tensor):
        - context ()
        - attn (): Attention matrix for visualization.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: int,
    ) -> None:
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be 0"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.W_query  = nn.Linear(d_model, d_model)
        self.W_key    = nn.Linear(d_model, d_model)
        self.W_value  = nn.Linear(d_model, d_model)
        self.W_output = nn.Linear(d_model, d_model)

        self.scaled_dot_attn = ScaledDotProductAttention(d_model, dropout_p)


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = query.shape[0]

        # original: (batch, seq_len, d_model) 
        # --forward--> (batch, seq_len, d_model) 
        # --view--> (batch, seq_len, num_heads, d_head) 
        # --transpose--> (batch, num_heads, seq_len, d_head)
        query = self.W_query(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)
        key   =     self.W_key(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)
        value = self.W_value(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1,2)
        
        context, attn = self.scaled_dot_attn(query, key, value, mask)

        # (batch, num_heads, seq_len, d_head)
        # --transpose--> (batch, seq_len, num_heads, d_head)
        # --view--> (batch, seq_len, d_model)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        context = self.W_output(context)

        return context, attn
    