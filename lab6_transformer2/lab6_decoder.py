# Lab 6 Task 2: Transformer Decoder Block [30 points]

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


class TransformerDecoderBlock(nn.Module):
    """
    Pre-LN Transformer Decoder Block with 3 sub-layers:
    1. Masked Self-Attention (causal)
    2. Cross-Attention (Q from decoder, K/V from encoder)
    3. Feed-Forward Network

    Each sub-layer uses Pre-LN residual connections:
        x = x + SubLayer(LayerNorm(x))

    This extends the Encoder Block by adding a Cross-Attention sub-layer.
    """

    def __init__(self, d_model, nhead, d_ff):
        """
        ### TODO: Initialize all layers

        You need (compare with the Encoder Block — what's new here?):
        1. self.self_attn: nn.MultiheadAttention(d_model, nhead, batch_first=True)
        2. self.cross_attn: nn.MultiheadAttention(d_model, nhead, batch_first=True)
        3. self.ffn: FeedForwardNetwork(d_model, d_ff)
        4. self.norm1, self.norm2, self.norm3: nn.LayerNorm(d_model)
        """
        super().__init__()
        # --- Your code starts here ---
        
        # The self-attention layer is the same as in the encoder block
        # Query, Key, Value all come from the target sequence (decoder input)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True # input shape: (batch, seq_len, d_model)
        )

        # The cross-attention layer is new in the decoder block
        # query comes from the target sequence (decoder input)
        # while key and value come from the encoder output
        # the purpose of this layer is to allow the decoder
        # to attend to the encoder's output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # The FFN is the same as in the encoder block
        # d_model is the input and output dimension of the FFN
        # d_ff is the hidden dimension of the FFN
        # contains two linear layers with a ReLU activation in between
        self.ffn = FeedForwardNetwork(d_model, d_ff)

        # LayerNorm for each sub-layer (3 sub-layers in total)
        # before each sub_layer, we apply LayerNorm to the input of that sub-layer
        self.norm1 = nn.LayerNorm(d_model) # for masked self-attention
        self.norm2 = nn.LayerNorm(d_model) # for cross-attention
        self.norm3 = nn.LayerNorm(d_model) # for FFN

        # --- Your code ends here ---

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x: Target sequence, shape (batch, tgt_len, d_model)
            enc_output: Encoder output, shape (batch, src_len, d_model)
            tgt_mask: Causal mask for self-attention, shape (tgt_len, tgt_len)
            memory_mask: Optional mask for cross-attention

        Returns:
            Output tensor, shape (batch, tgt_len, d_model)

        ### TODO: Implement 3 sub-layers with Pre-LN residual connections
        Sub-layer 1: Masked Self-Attention
          - Apply norm1, then self_attn(norm_x, norm_x, norm_x, attn_mask=tgt_mask)
          - Add residual connection

        Sub-layer 2: Cross-Attention (the NEW sub-layer vs Encoder!)
          - Apply norm2, then cross_attn(query=norm_x, key=enc_output, value=enc_output)
          - Add residual connection

        Sub-layer 3: FFN
          - Apply norm3, then ffn(norm_x)
          - Add residual connection
        """
        # --- Your code starts here ---

        # ------- First sub-layer: Masked Self-Attention -------
        # before the self-attention, we apply layer normalization to the input x
        norm_x = self.norm1(x)

        # the self-attention layer takes the normalized input as query, key, and value
        # the attn_mask is the causal mask that prevents attending to future tokens
        # returns the attention output and the attention weights (we can ignore the weights here)
        self_attn_out, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=tgt_mask
        )

        # Add the residual connection for the first sub-layer
        x = x + self_attn_out

        # ------- Second sub-layer: Cross-Attention -------
        # before the cross-attention, we apply layer normalization
        # to the output of the first sub-layer
        norm_x = self.norm2(x)

        # the cross-attention layer takes the normalized x as query
        # and the encoder output as key and value
        # enc_output is the output from the encoder
        # which contains information about the source sequence
        cross_attn_out, _ = self.cross_attn(
            query=norm_x,
            key=enc_output,
            value=enc_output,
            attn_mask=memory_mask
        )

        # again, we add the residual connection for the cross-attention sub-layer
        x = x + cross_attn_out

        # ------- Third sub-layer: FFN -------
        # before the FFN, we apply layer normalization to the output of the second sub-layer
        norm_x = self.norm3(x)

        # then, we pass the normalized x through the feed-forward network
        ffn_out = self.ffn(norm_x)

        # finally, we add the residual connection for the FFN sub-layer
        x = x + ffn_out

        # x shape: (batch, tgt_len, d_model) - same as the input target sequence
        return x
        # --- Your code ends here ---


def main():
    print("Testing Task 2: Transformer Decoder Block")
    d_model, nhead, d_ff = 512, 8, 2048
    decoder_block = TransformerDecoderBlock(d_model, nhead, d_ff)

    tgt = torch.randn(2, 5, d_model)  # Target sequence
    memory = torch.randn(2, 10, d_model)  # Encoder output
    output = decoder_block(tgt, memory)
    print(f"Decoder Block output shape: {output.shape}")
    assert output.shape == tgt.shape, "Output shape mismatch!"
    print("Task 2 test passed!")


if __name__ == "__main__":
    main()
