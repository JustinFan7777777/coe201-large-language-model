# lab5_gqa.py
# Lab 5 Task 3: Grouped Query Attention

import torch
import torch.nn as nn
from lab5_sdpa import scaled_dot_product_attention

class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA).
    G Query heads share 1 Key/Value head.
    """
    def __init__(self, d_model, num_queries, num_groups):
        super().__init__()
        assert num_queries % num_groups == 0
        self.d_model = d_model
        # d_model 
        self.num_queries = num_queries
        self.num_groups = num_groups # This is the number of Key/Value heads
        self.heads_per_group = num_queries // num_groups
        self.d_k = d_model // num_queries

        ### TODO: Define linear projection layers
        # Note: The output dimension of W_K and W_V should be num_groups * self.d_k

        # Q needs num_queries heads, K and V need num_groups heads
        self.W_Q = nn.Linear(d_model, num_queries * self.d_k)
        self.W_K = nn.Linear(d_model, num_groups * self.d_k)
        self.W_V = nn.Linear(d_model, num_groups * self.d_k)

        # Output projection layer
        # The input dimension is num_queries * self.d_k
        # because we will concatenate the outputs of all query heads
        # The output dimension is d_model to match the input dimension of the next layer
        self.W_O = nn.Linear(num_queries * self.d_k, d_model)

    def forward(self, Q_x, K_x, V_x, mask=None):
        """
        ### TODO: Implement the forward pass for GQA.
        Hints:
        1. Apply linear projections to Q_x, K_x, V_x to get Q, K, V (Note the output dimensions of K and V).
        2. Expand K and V to match the number of Q heads.
           Tip: Use torch.repeat_interleave or expand.
        3. Call scaled_dot_product_attention.
        
        Note: In self-attention, the incoming Q_x, K_x, and V_x are all exactly the same input sequence x.
        """
        batch_size = Q_x.size(0)

        # --- Your code starts here ---

        # Get the sequence lengths for Q, K, V
        # This is needed for the linear projections and for the attention computation
        # Q_x, K_x, V_x have shape (batch_size, seq_len, d_model)
        q_len = Q_x.size(1)
        k_len = K_x.size(1)
        v_len = V_x.size(1)

        # 1. Linear projections

        Q = self.W_Q(Q_x) # (batch_size, q_len, num_queries * d_k)
        K = self.W_K(K_x) # (batch_size, k_len, num_groups * d_k)
        V = self.W_V(V_x) # (batch_size, v_len, num_groups * d_k)

        # 2. Expand K and V to match the number of Q heads

        # view function is used to reshape the tensor
        # and transpose is used to swap dimensions
        Q = Q.view(batch_size, q_len, self.num_queries, self.d_k).transpose(1, 2)

        # view: (batch_size, k_len, num_groups * d_k) -> (batch_size, k_len, num_groups, d_k)
        K = K.view(batch_size, k_len, self.num_groups, self.d_k).transpose(1, 2)
        V = V.view(batch_size, v_len, self.num_groups, self.d_k).transpose(1, 2)
        # Now K and V have shape (batch_size, num_groups, seq_len, d_k)

        # Repeat K and V for each query head
        # For example, if num_groups=2 and heads_per_group=4
        # we need to repeat K and V 4 times along the head dimension
        # dim=1 because the head dimension is now the second dimension after transpose
        # heads_per_group is the number of query heads that share the same key/value head
        # heads_per_group * num_groups should equal num_queries
        K = K.repeat_interleave(self.heads_per_group, dim=1)
        V = V.repeat_interleave(self.heads_per_group, dim=1)
        # After repeat_interleave, K and V have shape (batch_size, num_queries, seq_len, d_k)

        # Implement the masking (Optional)
        if mask is not None:
            if mask.dim() == 2: mask = mask.view(1, 1, mask.size(0), mask.size(1))
            elif mask.dim() == 3: mask = mask.unsqueeze(1)

        # 3. Call scaled_dot_product_attention
        # Q has shape (batch_size, num_queries, q_len, d_k)
        # K and V have shape (batch_size, num_queries, seq_len, d_k)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output has shape (batch_size, num_queries, q_len, d_k)

        # 4. Concatenate the outputs of all query heads and apply the output projection
        # First, we need to reshape attn_output to (batch_size, q_len, num_queries * d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.num_queries * self.d_k)
        
        # Then we apply the output projection to get the final output
        # The output shape should be (batch_size, q_len, d_model)
        # W_O takes the concatenated output of all query heads and projects it back to d_model
        output = self.W_O(attn_output)

        return output, attn_weights
        # --- Your code ends here ---

def main():
    print("Testing Task 3: Grouped Query Attention (GQA)")
    batch_size, seq_len, d_model = 2, 4, 64
    num_queries = 8
    num_groups = 2 # 4 queries per group
    
    gqa = GroupedQueryAttention(d_model, num_queries, num_groups)
    x = torch.randn(batch_size, seq_len, d_model)
    output, _ = gqa(x, x, x)
    print(f"GQA Output shape: {output.shape}")

if __name__ == "__main__":
    main()
