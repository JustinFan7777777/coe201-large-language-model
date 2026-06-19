import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Problem 1a: Root Mean Square Normalization (RMSNorm) [15 points]
    
    RMSNorm is a simplified and more efficient alternative to LayerNorm, used in modern
    LLMs like LLaMA. Instead of centering the activations (subtracting the mean) as in
    LayerNorm, it only scales them by the root mean square.
    
    Mathematical Definition:
    RMS(x) = sqrt( 1/d * sum(x_i^2) + eps )
    output = (x / RMS(x)) * weight
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Normalized tensor of same shape as x.
            
        ### TODO: Implement the RMSNorm forward pass.
        # Step 1: Compute the mean of the squares of x along the last dimension.
        #         (Keep dimensions so it broadcasts correctly: keepdim=True)
        # Step 2: Add self.eps for numerical stability, then take the square root. (This is RMS(x))
        # Step 3: Divide x by RMS(x).
        # Step 4: Multiply by the learnable parameter self.weight.
        """
        # --- Your code starts here ---
        
        # step 1: keepdim=True ensures the output has the same number of dimensions as x
        # which allows for correct broadcasting in subsequent operations.
        # formula: mean_of_squares = 1/d * sum(x_i^2) where d is the size of the last dimension
        mean_of_squares = torch.mean(x ** 2, dim=-1, keepdim=True)

        # step 2: add eps for numerical stability to avoid division by zero, then take the square root
        rms = torch.sqrt(mean_of_squares + self.eps)

        # step 3: normalize x by dividing by the computed RMS
        # shape of rms is (..., 1), shape of x is (..., d_model)
        normalized_x = x / rms

        # step 4: scale the normalized output by the learnable weight parameter
        output = normalized_x * self.weight

        return output

        # --- Your code ends here ---


class SwiGLU(nn.Module):
    """
    Problem 1b: Swish Gated Linear Unit (SwiGLU) [15 points]
    
    SwiGLU is an activation function variant used in modern Transformers (e.g., LLaMA, PaLM).
    It replaces the standard two-layer FFN and ReLU/GELU.
    
    It operates by gating a linear projection using the Swish activation function on 
    another linear projection.
    
    Mathematical Definition:
    Swish(x) = x * sigmoid(beta * x)  (usually beta=1, which is also called SiLU)
    SwiGLU(x) = ( Swish(x * W_gate) * (x * W_up) ) * W_down
    
    Here, `* W` denotes a linear layer (without bias in common implementations).
    """
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()

        # hint: use nn.Linear with bias=False
        # --- Your code starts here ---

        # formula: gate_projection = x * W_gate
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False) 
        # This linear layer computes the gate projection
        # which will be passed through the Swish activation.
        # The input dimension is d_model and the output dimension is hidden_dim
        # which is typically larger than d_model (e.g., 4 * d_model).

        # formula: up_projection = x * W_up
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        # This linear layer computes the up projection which will be multiplied element-wise

        # formula: output = (activated_gate * up_projection) * W_down
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        # This linear layer takes the result of the element-wise multiplication
        # and projects it back to the original dimension d_model.

        # --- Your code ends here ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU FFN.
        
        Args:
            x: Input tensor of shape (..., d_model)
            
        Returns:
            Output tensor of shape (..., d_model)
            
        ### TODO: Implement the SwiGLU forward pass.
        # Step 1: Compute the gate projection: `self.w_gate(x)`.
        # Step 2: Apply the SiLU (Swish with beta=1) activation to the gate output.
        #         You can use `F.silu(gate)` or manually compute `gate * torch.sigmoid(gate)`.
        # Step 3: Compute the up projection: `self.w_up(x)`.
        # Step 4: Multiply (element-wise) the activated gate and the up projection.
        # Step 5: Pass the result through the down projection: `self.w_down(...)`.
        """
        # --- Your code starts here ---
        
        # x's shape is (..., d_model)
        # after linear layer w_gate, the shape becomes (..., hidden_dim)
        gate = self.w_gate(x) # shape: (..., hidden_dim)
        # now gate = x * W_gate

        swish_gate = F.silu(gate) # shape: (..., hidden_dim)
        # swish_gate = gate * sigmoid(gate)
        # Silu is the same as Swish with beta=1, so we can use F.silu for convenience.

        up_projection = self.w_up(x) # shape: (..., hidden_dim)
        # up_projection = x * W_up

        # element-wise multiplication of the activated gate and the up projection
        gated_up = swish_gate * up_projection # shape: (..., hidden_dim)
        # This is the core of the SwiGLU activation, where the gate modulates the up projection.
        # gated_up = Swish(x * W_gate) * (x * W_up)

        output = self.w_down(gated_up) # shape: (..., d_model)
        # Finally, we project back to the original dimension using the down projection.
        # output = (Swish(x * W_gate) * (x * W_up)) * W_down

        return output

        # --- Your code ends here ---
