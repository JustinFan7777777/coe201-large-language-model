# lab5_mha.py
# Lab 5 Task 2: Multi-Head Attention

import torch
import torch.nn as nn
from lab5_sdpa import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        ### TODO: Define linear projection layers
        # --- Your code starts here ---

        # Each of W_Q, W_K, W_V projects from d_model to d_model
        # which will then be reshaped to (num_heads, d_k)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # --- Your code ends here ---

    def forward(self, Q_x, K_x, V_x, mask=None):
        batch_size = Q_x.size(0)

        # --- Your code starts here ---
        ### TODO: 1. Apply linear projections to Q_x, K_x, V_x to get Q, K, V
        # Hint: In self-attention, the incoming Q_x, K_x, and V_x are all exactly the same input sequence x.
        
        Q = self.W_Q(Q_x)  # (batch_size, seq_len, d_model)
        K = self.W_K(K_x)  # (batch_size, seq_len, d_model)
        V = self.W_V(V_x)  # (batch_size, seq_len, d_model)

        ### TODO: 2. Reshape for multi-head attention: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        # Hint: Use .view() and .transpose()

        q_len = Q.size(1) # sequence length
        k_len = K.size(1) # sequence length
        v_len = V.size(1) # sequence length

        # Reshape Q, K, V to (batch_size, seq_len, num_heads, d_k) and then transpose to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, q_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, k_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        V = V.view(batch_size, v_len, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        
        ### TODO: 3. Compute scaled dot-product attention
        # Implement the masking (Optional)
        if mask is not None:
            if mask.dim() == 2: 
                mask = mask.view(1, 1, mask.size(0), mask.size(1))
            elif mask.dim() == 3: 
                mask = mask.unsqueeze(1)

        ### TODO: Call scaled_dot_product_attention to get attn_output and attn_weights

        # attn_output shape: (batch_size, num_heads, seq_len, d_k)
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        ### TODO: 4. Concatenate the heads back together
        # Hint: Use .transpose(), .contiguous(), and .view()

        # transpose attn_output back to (batch_size, seq_len, num_heads, d_k)
        # contiguous() is needed before view() to ensure the tensor is stored in a contiguous chunk of memory
        # finally, reshape to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        
        ### TODO: 5.
        # Apply the final linear projection (W_O)

        # W_O projects from d_model back to d_model
        # so the output shape will be (batch_size, seq_len, d_model)
        output = self.W_O(attn_output)
        
        return output, attn_weights
        # --- Your code ends here ---

def main():
    print("Testing Task 2: Multi-Head Attention")
    batch_size, seq_len, d_model = 2, 4, 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, attn_weights = mha(x, x, x)
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {attn_weights.shape}")

if __name__ == "__main__":
    main()
