import torch
import torch.nn as nn
import numpy as np

class DualRelativePositionBias(nn.Module):
    """
    DualRelativePositionBias:
    Generates:
      1. Cartesian relative position bias: (H, N_p, N_p)
      2. Polar relative position bias: (B, H, N_p, N_p)
    Where N_p is the number of patch tokens (grid_size * grid_size).
    """
    def __init__(self, grid_size=20, num_heads=6):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.num_heads = num_heads
        
        # Precompute patch centers in normalized coordinates [0, 1]
        patch_coords = []
        for r in range(grid_size):
            for c in range(grid_size):
                px = (c + 0.5) / grid_size
                py = (r + 0.5) / grid_size
                patch_coords.append([px, py])
                
        # Shape: (N_p, 2)
        self.register_buffer('patch_coords', torch.tensor(patch_coords, dtype=torch.float32))
        
        # Cartesian relative position bias generator (shared across batch)
        self.cart_bias_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, num_heads)
        )
        
        # Polar relative position bias generator (dependent on per-sample wafer center)
        self.polar_bias_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, num_heads)
        )
        
        # Precompute Cartesian differences: (N_p, N_p, 2)
        # diff_ij = pos_i - pos_j
        cart_diffs = self.patch_coords.unsqueeze(1) - self.patch_coords.unsqueeze(0)
        self.register_buffer('cart_diffs', cart_diffs)
        
        # Initialize MLP weights to small values to start training stably
        self._init_weights()

    def _init_weights(self):
        for m in [self.cart_bias_mlp, self.polar_bias_mlp]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, center_coords):
        """
        Args:
            center_coords: Tensor of shape (B, 3) representing (cx, cy, R) per sample.
        Returns:
            cart_bias: (H, N_p, N_p)
            polar_bias: (B, H, N_p, N_p)
        """
        B = center_coords.shape[0]
        device = center_coords.device
        
        # 1. Compute Cartesian relative position bias
        # Shape: (N_p, N_p, H) -> permute to (H, N_p, N_p)
        cart_bias = self.cart_bias_mlp(self.cart_diffs).permute(2, 0, 1)
        
        # 2. Compute Polar relative position bias
        cx = center_coords[:, 0].view(B, 1)  # (B, 1)
        cy = center_coords[:, 1].view(B, 1)  # (B, 1)
        R = center_coords[:, 2].view(B, 1)   # (B, 1)
        
        px = self.patch_coords[:, 0].view(1, self.num_patches)  # (1, N_p)
        py = self.patch_coords[:, 1].view(1, self.num_patches)  # (1, N_p)
        
        # Absolute Cartesian coords relative to estimated wafer center
        x_rel = px - cx  # (B, N_p)
        y_rel = py - cy  # (B, N_p)
        
        # Absolute polar coords
        r = torch.sqrt(x_rel**2 + y_rel**2) / (R + 1e-8)  # normalized radius, shape: (B, N_p)
        theta = torch.atan2(y_rel, x_rel)                 # angle in [-pi, pi], shape: (B, N_p)
        
        # Pairwise differences
        dr = r.unsqueeze(2) - r.unsqueeze(1)          # (B, N_p, N_p)
        dtheta = theta.unsqueeze(2) - theta.unsqueeze(1)  # (B, N_p, N_p)
        
        # Wrap dtheta to [-pi, pi]
        dtheta = torch.remainder(dtheta + np.pi, 2 * np.pi) - np.pi
        
        # Stack polar differences: (B, N_p, N_p, 2)
        polar_diffs = torch.stack([dr, dtheta], dim=-1)
        
        # Pass through MLP to get (B, N_p, N_p, H)
        polar_bias = self.polar_bias_mlp(polar_diffs)
        # Permute to (B, H, N_p, N_p)
        polar_bias = polar_bias.permute(0, 3, 1, 2)
        
        return cart_bias, polar_bias


class GDCoRPAAttention(nn.Module):
    """
    GDCoRPAAttention:
    Attention layer with gated Cartesian & Polar relative position biases.
    Supports ablation modes:
      - None or 'full': proposed content-conditioned gate.
      - 'cart_only': gate is forced to 1.0 (Cartesian bias only).
      - 'polar_only': gate is forced to 0.0 (Polar bias only).
      - 'static_gate': single learned scalar per head, not content-conditioned.
    """
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0., grid_size=20, ablation_mode=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.ablation_mode = ablation_mode
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Dual relative position bias generator
        self.relative_bias_gen = DualRelativePositionBias(grid_size=grid_size, num_heads=num_heads)
        
        # Content-conditioned gate or static/ablation gates
        if self.ablation_mode == "static_gate":
            # Per head learned scalar (init to 0.0, Sigmoid makes it 0.5)
            self.static_gate = nn.Parameter(torch.zeros(num_heads, 1, 1))
        elif self.ablation_mode not in ["cart_only", "polar_only"]:
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, num_heads),
                nn.Sigmoid()
            )

    def forward(self, x, center_coords):
        """
        Args:
            x: Tensor of shape (B, N, D) where N = num_patches + 1 (CLS token + patch tokens)
            center_coords: Tensor of shape (B, 3) representing (cx, cy, R) per sample.
        Returns:
            out: Attention-refined tokens (B, N, D)
            gate: Gate values (B, H, N_patches, 1) for visualization
        """
        B, N, D = x.shape
        N_patches = N - 1
        
        # 1. Project and split Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, D_head)
        
        # 2. Base attention score
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # 3. Compute query-content gate g for patch tokens
        if self.ablation_mode == "cart_only":
            gate = torch.ones(B, self.num_heads, N_patches, 1, device=x.device, dtype=x.dtype)
        elif self.ablation_mode == "polar_only":
            gate = torch.zeros(B, self.num_heads, N_patches, 1, device=x.device, dtype=x.dtype)
        elif self.ablation_mode == "static_gate":
            # Sigmoid maps parameter to [0.0, 1.0] range
            gate = torch.sigmoid(self.static_gate).unsqueeze(0).expand(B, -1, N_patches, -1)
        else:
            gate = self.gate_mlp(x[:, 1:])
            gate = gate.permute(0, 2, 1).unsqueeze(-1)
        
        # 4. Generate relative position biases
        cart_bias, polar_bias = self.relative_bias_gen(center_coords)  # (H, N_p, N_p), (B, H, N_p, N_p)
        
        # Combine biases: g * cart_bias + (1 - g) * polar_bias
        # Broadcast cart_bias to batch dimension: (1, H, N_p, N_p)
        # Combined shape: (B, H, N_p, N_p)
        combined_bias = gate * cart_bias.unsqueeze(0) + (1.0 - gate) * polar_bias
        
        # 5. Pad combined bias with zeros for CLS token (index 0)
        # Attn bias shape: (B, H, N, N) where N = N_patches + 1
        attn_bias = torch.zeros(B, self.num_heads, N, N, device=x.device, dtype=x.dtype)
        attn_bias[:, :, 1:, 1:] = combined_bias
        
        # Add biases to attention scores
        attn = attn + attn_bias
        
        # 6. Softmax and project
        attn_probs = torch.softmax(attn, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        
        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, gate
