# Lab 9 Task 1: LoRA Layer from Scratch [40 points]

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    A linear layer with a LoRA (Low-Rank Adaptation) bypass.

    Instead of modifying the pre-trained weight W, LoRA adds a low-rank
    decomposition delta_W = B @ A, where:
      - A: (r, in_features) initialized with Kaiming uniform
      - B: (out_features, r) initialized with zeros

    The forward pass computes: y = Wx + BAx * (alpha / r)

    Args:
        original_linear: The nn.Linear layer to wrap
        r: LoRA rank
        alpha: Scaling factor (default: same as r)
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 8):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # ### TODO: 
        # 1. Freeze the original weight and bias (if it exists) by setting requires_grad_(False)
        # 2. Create LoRA matrices A and B as nn.Parameter
        # A: shape (r, in_features), initialize with Kaiming uniform
        # B: shape (out_features, r), initialize with zeros
        # Hint: Use nn.Parameter(torch.empty(...)) then call nn.init.kaiming_uniform_ on A

        # --- Your code starts here ---

        # 1. Freeze the original weight and bias
        # We set requires_grad_(False) to prevent these parameters from being updated during training.
        self.original_linear.weight.requires_grad_(False)
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad_(False)

        # 2. Create LoRA matrices A and B
        # A is initialized with Kaiming uniform, which is a common initialization for linear layers
        # B is initialized with zeros, meaning that initially, the LoRA contribution is zero (identity function)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.zeros_(self.lora_B)

        # --- Your code ends here ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ### TODO: Implement the forward pass
        # Note: original_linear(x) computes Wx + b

        # --- Your code starts here ---

        base_out = self.original_linear(x)  # Compute the original linear output Wx + b

        lora_out = self.lora_B @ (self.lora_A @ x.T)  # Compute the LoRA contribution BAx

        lora_out = lora_out.T * self.scaling  # Scale the LoRA contribution by alpha/r

        return base_out + lora_out  # Combine the original output with the LoRA contribution
 
        # --- Your code ends here ---


def inject_lora(model: nn.Module, r: int = 8, alpha: int = 8) -> nn.Module:
    """
    Replace all nn.Linear layers in the model with LoRALinear wrappers.

    Args:
        model: The model to inject LoRA into
        r: LoRA rank
        alpha: Scaling factor

    Returns:
        The model with LoRA injected
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            inject_lora(module, r=r, alpha=alpha)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params) for the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=" * 60)
    print("Task 1: LoRA Layer from Scratch")
    print("=" * 60)

    # 1. Create a simple model and test LoRA
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # Count before LoRA
    total_before, trainable_before = count_parameters(model)
    print(f"\nBefore LoRA: total={total_before:,}, trainable={trainable_before:,}")

    # Inject LoRA
    model = inject_lora(model, r=8, alpha=8)

    # Count after LoRA
    total_after, trainable_after = count_parameters(model)
    print(f"After LoRA:  total={total_after:,}, trainable={trainable_after:,}")
    print(f"Trainable ratio: {trainable_after / total_after:.2%}")

    # 2. Forward pass sanity check
    x = torch.randn(4, 64)
    out = model(x)
    print(f"\nForward pass: input={x.shape}, output={out.shape}")

    # 3. Verify that original weights are frozen and LoRA params are trainable
    lora_layer = model[0]  # First layer is LoRALinear
    print(f"\nOriginal weight frozen: {not lora_layer.original_linear.weight.requires_grad}")
    print(f"LoRA A trainable: {lora_layer.lora_A.requires_grad}")
    print(f"LoRA B trainable: {lora_layer.lora_B.requires_grad}")

    # 4. Verify LoRA starts as identity (B=0, so delta_W=0)
    with torch.no_grad():
        out_original = lora_layer.original_linear(x)
        out_lora = lora_layer(x)
        diff = (out_original - out_lora).abs().max().item()
    print(f"\nInitial LoRA is identity (max diff should be ~0): {diff:.6f}")


if __name__ == "__main__":
    main()
