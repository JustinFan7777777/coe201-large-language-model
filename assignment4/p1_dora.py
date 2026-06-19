import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoRALinear(nn.Module):
    """
    Problem 1: Weight-Decomposed Low-Rank Adaptation (DoRA) [35 points]
    
    DoRA decomposes the pre-trained weight into magnitude and direction.
    W' = m * (W_0 + B @ A) / ||W_0 + B @ A||_c
    """
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.scaling = alpha / r

        # 1. Freeze original weights
        self.weight = original_linear.weight
        self.weight.requires_grad_(False)
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        ### TODO:
        # Initialize the LoRA directional matrices A and B with appropriate dimensions and distributions.
        # Then, compute the initial magnitude `m` from the pre-trained weights and register it as a learnable parameter.
        
        # --- Your code starts here ---
        
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # use Kaiming uniform initialization for A to ensure good variance scaling
        # B is initialized to zero so that the initial update is zero and the model starts with the original pre-trained weights

        m_init = torch.linalg.vector_norm(self.weight, dim=1)
        self.m = nn.Parameter(m_init)
        # Initialize m with the row-wise L2 norm of the pre-trained weights
        # This captures the initial magnitude of the weights and allows it to be fine-tuned during training.
        # nn.Parameter is used to make m a learnable parameter that will be updated during training.

        # --- Your code ends here ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ### TODO:
        # Construct the updated weight matrix W_prime by composing the directional 
        # component (which includes the LoRA adapter) and the learned magnitude.
        # Finally, perform the linear projection using W_prime.
        
        # --- Your code starts here ---
        
        ba = (self.lora_B @ self.lora_A) * self.scaling
        # self.lora_B: (out_features, r)
        # self.lora_A: (r, in_features)
        # ba: (out_features, in_features)

        v_prime = self.weight + ba
        # self.weight is the frozen pre-trained weight
        # with shape = (out_features, in_features)
        # v_prime is the updated weight matrix before normalization

        row_norm = torch.linalg.vector_norm(v_prime, dim=1, keepdim=True)
        # calculate the row-wise L2 norm of v_prime for normalization
        # shape: (out_features, 1)

        eps = 1e-12
        v_normalized = v_prime / (row_norm + eps)
        # normalize v_prime to get the direction component
        # shape: (out_features, in_features)

        w_prime = self.m.unsqueeze(1) * v_normalized
        # scale the normalized direction by the learned magnitude m
        # self.m: (out_features,)
        # w_prime: (out_features, in_features)
        # unsqueeze m to match the dimensions for element-wise multiplication

        return F.linear(x, w_prime, self.bias)
        # linear projection using the updated weight w_prime and original bias
        # x: (batch_size, in_features) -> out: (batch_size, out_features)

        # --- Your code ends here ---

def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

if __name__ == "__main__":
    print("=" * 60)
    print("Task 1: DoRA Layer implementation")
    print("=" * 60)

    torch.manual_seed(42)
    base_layer = nn.Linear(64, 128)
    
    # Wrap it
    dora_layer = DoRALinear(base_layer, r=8, alpha=8)
    
    total, trainable = count_parameters(dora_layer)
    print(f"Total params: {total}, Trainable params: {trainable}")
    
    # Forward pass
    x = torch.randn(4, 64)
    try:
        out = dora_layer(x)
        print(f"Forward output shape: {out.shape}")
    except NotImplementedError:
        print("[Skip] Not implemented.")
