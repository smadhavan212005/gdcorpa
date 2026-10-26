import torch
import torch.nn as nn
from model.attention import GDCoRPAAttention

class Mlp(nn.Module):
    """ Standard Multi-Layer Perceptron (MLP) block for ViTs. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GDCoRPABlock(nn.Module):
    """ Vision Transformer Block featuring GD-CoRPA Attention. """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, grid_size=20, ablation_mode=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GDCoRPAAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, grid_size=grid_size,
            ablation_mode=ablation_mode
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, center_coords):
        """
        Args:
            x: Input token tensor of shape (B, N, D)
            center_coords: Wafer center coordinates (B, 3)
        Returns:
            out: Refined token tensor of shape (B, N, D)
            gate: Attention gate values (B, H, N_patches, 1)
        """
        attn_out, gate = self.attn(self.norm1(x), center_coords)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x, gate
