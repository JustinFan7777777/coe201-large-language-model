# Lab 6 Task 1: Transformer Encoder Block [30 points]

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network.

    FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2

    This is provided for you — use it as a sub-layer.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.ffn(x)


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LN Transformer Encoder Block with 2 sub-layers:
    1. Self-Attention
    2. Feed-Forward Network

    Each sub-layer uses Pre-LN residual connections:
        x = x + SubLayer(LayerNorm(x))

    Input/Output shape: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, nhead, d_ff):
        """
        ### TODO: Initialize all layers

        You need:
        1. self.self_attn: nn.MultiheadAttention(d_model, nhead, batch_first=True)
        2. self.ffn: FeedForwardNetwork(d_model, d_ff)
        3. self.norm1: nn.LayerNorm(d_model)  -- for sub-layer 1
        4. self.norm2: nn.LayerNorm(d_model)  -- for sub-layer 2
        """
        super().__init__()
        # --- Your code starts here ---

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, # d_model is the embedding dimension
            num_heads=nhead, # nhead is the number of attention heads
            batch_first=True # batch_first=True means input shape is (batch, seq_len, d_model)
        )

        # d_model is the input and output dimension of the FFN
        # d_ff is the hidden dimension of the FFN
        # The FFN consists of two linear layers with a ReLU activation in between
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # LayerNorm for the first sub-layer (self-attention)
        self.norm1 = nn.LayerNorm(d_model)

        # LayerNorm for the second sub-layer (FFN)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Your code ends here ---

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor, shape (batch, seq_len, d_model)

        ### TODO: Implement 2 sub-layers with Pre-LN residual connections
        Sub-layer 1: Self-Attention
          - Apply norm1 to x
          - Pass normalized x as query, key, value to self_attn
          - Add residual connection: x = x + attn_out

        Sub-layer 2: FFN
          - Apply norm2 to x
          - Pass normalized x through ffn
          - Add residual connection: x = x + ffn_out
        """
        # --- Your code starts here ---

        # first, apply layer normalization to the input x
        # for the self-attention sub-layer
        norm_x = self.norm1(x)

        # then, pass the normalized x as query, key, value to the self-attention layer
        # the self-attention layer will return the attention output
        # and the attention weights (which we can ignore)
        attn_out, _ = self.self_attn(
            query=norm_x, # the query is the normalized x
            key=norm_x, # the key is also the normalized x (self-attention)
            value=norm_x, # the value is also the normalized x (self-attention)
            attn_mask=mask # mask is optional and can be passed to the self-attention layer
        )

        # add the residual connection for the self-attention sub-layer
        # the output of the self-attention sub-layer is added to the original input x
        # this allows the model to learn to preserve the original input if needed
        x = x + attn_out

        # next, apply layer normalization to the output of the first sub-layer
        norm_x = self.norm2(x)

        # then, pass the normalized x through the feed-forward network
        # the feed-forward network will return the output of the FFN
        ffn_out = self.ffn(norm_x)

        # add the residual connection for the FFN sub-layer
        # the output of the FFN sub-layer is added to the input of the FFN sub-layer
        # (which is the output of the first sub-layer)
        x = x + ffn_out

        # finally, return the output x, which has the same shape as the input x
        # shape: (batch, seq_len, d_model)
        return x
        # --- Your code ends here ---


def main():
    print("Testing Task 1: Transformer Encoder Block")
    d_model, nhead, d_ff = 512, 8, 2048
    encoder_block = TransformerEncoderBlock(d_model, nhead, d_ff)

    x = torch.randn(2, 10, d_model)
    output = encoder_block(x)
    print(f"Encoder Block output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"
    print("Task 1 test passed!")


if __name__ == "__main__":
    main()
