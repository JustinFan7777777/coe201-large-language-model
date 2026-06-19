import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Task 1: Image Patchification & Linear Projection
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        ### TODO: Define a Linear layer to map flattened patches to embed_dim
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)

    def forward(self, x):
        """
        Forward pass for patchification and linear projection.
        x shape: (B, C, H, W)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        
        ### TODO: 1. Unfold image into sliding blocks
        # Shape should be: (B, C, H//P, W//P, P, P)
        patches = x.unfold(2, P, P).unfold(3, P, P)
        
        ### TODO: 2. Permute to group patches spatially
        # Shape should be: (B, H//P, W//P, C, P, P)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        
        ### TODO: 3. Flatten the spatial patches into a 1D sequence of pixels
        # Shape should be: (B, num_patches, C * P * P)
        patches = patches.reshape(B, -1, C * P * P)
        
        ### TODO: 4. Linearly project to hidden dimension D
        # Shape should be: (B, num_patches, embed_dim)
        patches = self.proj(patches)
        
        return patches


class Block(nn.Module):
    """
    Task 2: ViT Block (Self-Attention + MLP)
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        ### TODO: Initialize LayerNorms, MultiheadAttention (set batch_first=True), and the MLP
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim)
        )
        
    def forward(self, x):
        """
        Forward pass for a single Transformer block.
        x shape: (B, N, D)
        """
        ### TODO: Implement the Pre-norm architecture + Residual connection for Attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        ### TODO: Implement the Pre-norm architecture + Residual connection for MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionTransformer(nn.Module):
    """
    Task 2: Assembly of the Vision Transformer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        # Patch embedding module
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        ### TODO: Define a Learnable [CLS] token (to be prepended to the sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        ### TODO: Define a Learnable 1D Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        ### TODO: 1. Patchify and Project
        x = self.patch_embed(x)
        
        ### TODO: 2. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        ### TODO: 3. Add positional embeddings
        x = x + self.pos_embed
        
        ### TODO: 4. Pass through Transformer encoder blocks
        for block in self.blocks:
            x = block(x)
            
        ### TODO: 5. Extract [CLS] token representation for classification
        x = self.norm(x)
        out = self.head(x[:, 0])
        
        return out


def main():
    # Test the implementation
    model = VisionTransformer()
    dummy_image = torch.randn(2, 3, 224, 224) # Batch of 2 images
    
    print(f"Input shape: {dummy_image.shape}")
    try:
        logits = model(dummy_image)
        print(f"Output shape: {logits.shape}")
        
        assert logits.shape == (2, 1000), "Output shape should be (B, num_classes)"
        print("ViT implementation passed the shape test!")
    except Exception as e:
        print("Implementation incomplete:", e)

if __name__ == "__main__":
    main()
