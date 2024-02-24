from typing import Tuple

from torch import nn
from torch import Tensor
from single_agent.Attention import MultiHeadAttention


class Attention_Feed_Forward(nn.Module):
    """
    An (Attention + Feed-forward) network

    Input are passed through the Attention layer then the Feed-forward network.
    This mimics the standard encoder layer in the paper "Attention Is All You Need".

    Returns:
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): output
        * attn (torch.FloatTensor): attention 
    """
    def __init__(
        self,
        attention_layer: MultiHeadAttention,
        feed_forward_layer: nn.Module,
    ) -> None:
        super(Attention_Feed_Forward, self).__init__()
        assert attention_layer.d_model == feed_forward_layer.in_features, \
            "dimention mismatch between Attention and Feed-forward layer"

        self.self_attention = attention_layer
        self.feed_forward = feed_forward_layer
        
        self.d_model = self.self_attention.d_model

        self.attention_prenorm = nn.LayerNorm(self.d_model)
        self.feed_forward_prenorm = nn.LayerNorm(self.d_model)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward propagates

        Returns:
            outputs (torch.FloatTensor): output 
            attn (torch.FloatTensor): attention
        """
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs)
        outputs += residual

        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)

        return outputs, attn


