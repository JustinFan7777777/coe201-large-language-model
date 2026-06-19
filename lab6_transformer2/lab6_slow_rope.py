# Lab 6 Task 3: Implementing and Comparing RoPE [Bonus]

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def get_sinusoidal_pe(max_len, d_model):
    """
    Compute sinusoidal positional encoding.
    
    ### TODO: Implement sinusoidal PE calculation 
    # Steps:
    # 1. Initialize a (max_len, d_model) zero tensor
    # 2. Compute the division term: exp(arange(0, d, 2) * -log(10000)/d)
    # 3. Apply sin to even indices and cos to odd indices
    # 4. Return as a numpy array
    """
    # --- Your code starts here ---
    
    # Initialize the positional encoding matrix
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)

    # unsqueeze to get shape (max_len, 1) for broadcasting
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)

    # Compute the division term for the frequencies
    div_term = torch.exp(
        # for even indices (0, 2, 4, ...), we compute the term for the sine function
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )

    # Apply sin to even indices (0, 2, 4, ...)
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply cos to odd indices (1, 3, 5, ...)
    # We need to ensure that the div_term for odd indices matches the even indices
    # Since div_term is computed for even indices, we can reuse it for odd indices
    # The number of odd dimensions will be the same as the number of even dimensions
    # if d_model is even
    # otherwise it will be one less
    num_odd_dims = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(position * div_term[:num_odd_dims])

    # Return the positional encoding as a numpy array
    return pe.numpy()

    # --- Your code ends here ---

def get_rope_matrix(max_len, d_model, base=10000):
    """
    Construct the (max_len, d_model, d_model) block-diagonal rotation matrix.
    
    ### TODO: Implement the RoPE rotation matrix
    """
    # --- Your code starts here ---

    # RoPE applies a rotation to the query and key vectors based on their position in the sequence
    # The rotation is defined by a block-diagonal matrix where each block corresponds to a pair of dimensions (even, odd)
    # For each pair of dimensions (2i, 2i+1), we have a 2x2 rotation matrix defined by the angle theta = position * (base ** (-2i/d_model))
    # The rotation matrix for each pair is:
    # R = [[cos(theta), -sin(theta)],
    #      [sin(theta),  cos(theta)]]
    # We need to construct a large block-diagonal matrix that applies these rotations to the entire
    rope = np.zeros((max_len, d_model, d_model), dtype=np.float32)

    # We will fill in the block-diagonal matrix for each position and each pair of dimensions
    for p in range(max_len):
        # For each position p, we compute the rotation for each pair of dimensions
        for i in range(0, d_model - 1, 2):
            # base: the base frequency for the rotation, typically 10000
            # i: the index of the dimension pair (2i, 2i+1)
            # d_model: the total number of dimensions
            theta = p / (base ** (i / d_model))
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            # Fill in the 2x2 block for dimensions (i, i+1) at position p
            rope[p, i, i] = cos_theta
            rope[p, i, i + 1] = -sin_theta
            rope[p, i + 1, i] = sin_theta
            rope[p, i + 1, i + 1] = cos_theta
        
        if d_model % 2 == 1:
            # If d_model is odd, we have one extra dimension that is not part of a pair
            # We can set it to identity (no rotation) for all positions
            rope[p, d_model - 1, d_model - 1] = 1.0

    return rope
    # --- Your code ends here ---

def apply_rope_matrix(q, rope_matrix):
    """
    Apply RoPE rotation using pure matrix multiplication.
    Formula: q_rope = q @ R
    """
    # ### TODO: Implement apply_rope_matrix
    # --- Your code starts here ---

    # q shape: (batch, seq_len, d_model)
    # rope_matrix shape: (max_len, d_model, d_model)
    # We need to apply the corresponding rotation matrix for each position in the sequence
    # We can use batch matrix multiplication to apply the rotation to each position
    # First, we need to select the appropriate rotation matrix for each position in the sequence
    # We can assume that seq_len <= max_len, so we can slice the rope_matrix to
    # get the rotation matrices for the positions in the sequence
    R = rope_matrix[: q.size(1)]

    # Now we can apply the rotation to the query tensor q
    # We can use torch.einsum to perform the batch matrix multiplication
    # The einsum equation "bpd,pde->bpe" means:
    # - b: batch dimension
    # - p: position dimension (seq_len)
    # - d: dimension of the model (d_model)
    # We are multiplying q (b, p, d) with R (p, d, d) to get q_rope (b, p, d)
    q_rope = torch.einsum("bpd,pde->bpe", q, R)

    return q_rope
    # --- Your code ends here ---

def main():
    print("Testing Bonus Task: Implementing and Comparing RoPE")
    # Reduced dimensions for easier analysis
    max_len = 50
    d_model = 64
    
    # 1. Get Sinusoidal PE
    sin_pe = get_sinusoidal_pe(max_len, d_model)
    
    # 2. Get RoPE rotation matrix
    rope_matrix = get_rope_matrix(max_len, d_model)
    
    if rope_matrix is None:
        print("Please implement get_rope_matrix() first.")
        return

    # 3. Test Rotation Logic
    print("Testing Rotation Logic...")
    q = torch.randn(1, max_len, d_model)
    q_rope = apply_rope_matrix(q, torch.from_numpy(rope_matrix).float())
    print(f"Applied RoPE to query, output shape: {q_rope.shape}")
    
    # 4. Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    im1 = axes[0].pcolormesh(sin_pe, cmap='RdBu')
    axes[0].set_title('Sinusoidal Positional Encoding')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Position')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot the diagonals of the rotation matrix (which corresponds to the cosine values)
    rope_diag = np.diagonal(rope_matrix, axis1=1, axis2=2)
    im2 = axes[1].pcolormesh(rope_diag, cmap='RdBu')
    axes[1].set_title('RoPE Matrix Diagonals (Cosine values)')
    axes[1].set_xlabel('Dimension Pair Index')
    axes[1].set_ylabel('Position')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    asset_dir = "assets"
    if not os.path.exists(asset_dir): os.makedirs(asset_dir)
    plt.savefig(os.path.join(asset_dir, 'pe_comparison.png'), dpi=300)
    print("Comparison plot saved to assets/pe_comparison.png")
    
    plt.show()

if __name__ == "__main__":
    main()
