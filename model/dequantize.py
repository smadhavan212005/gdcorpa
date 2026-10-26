import torch
import torch.nn as nn

class FixedViridisDequantizer(nn.Module):
    """
    FixedViridisDequantizer:
    1. Computes L2 distances to known anchors (background, normal, defective die).
    2. Uses a learnable gamma to get soft probability maps.
    3. Uses a small depthwise-separable residual conv to denoise the probabilities.
    4. Outputs a soft class probability map of shape (B, 3, H, W).
    """
    def __init__(self, init_gamma=15.0):
        super().__init__()
        # Buffer colors to run on GPU/CPU automatically
        self.register_buffer('c0', torch.tensor([68.0, 0.0, 83.0]) / 255.0)
        self.register_buffer('c1', torch.tensor([31.0, 144.0, 140.0]) / 255.0)
        self.register_buffer('c2', torch.tensor([253.0, 230.0, 36.0]) / 255.0)
        
        # Learnable gamma scaling factor
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
        
        # Depthwise-separable denoising residual conv
        self.res_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.Conv2d(3, 3, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False),
            nn.Conv2d(3, 3, kernel_size=1, bias=True)
        )

    def forward(self, x):
        # x shape: (B, 3, H, W)
        # Compute squared L2 distance to each anchor
        # Shape: (B, 1, H, W)
        d0 = torch.sum((x - self.c0.view(1, 3, 1, 1))**2, dim=1, keepdim=True)
        d1 = torch.sum((x - self.c1.view(1, 3, 1, 1))**2, dim=1, keepdim=True)
        d2 = torch.sum((x - self.c2.view(1, 3, 1, 1))**2, dim=1, keepdim=True)
        
        # Stack distances
        # Shape: (B, 3, H, W)
        dists = torch.cat([d0, d1, d2], dim=1)
        
        # Convert to soft probabilities using learnable scaling factor
        gamma_clamped = torch.clamp(self.gamma, min=0.1)
        soft_probs = torch.softmax(-gamma_clamped * dists, dim=1)
        
        # Apply depthwise-separable conv residual
        res = self.res_conv(soft_probs)
        
        # Re-apply softmax to ensure valid probability map
        output = torch.softmax(soft_probs + res, dim=1)
        
        return output
