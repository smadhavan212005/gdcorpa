import torch
import torch.nn as nn

class WaferCenterEstimator(nn.Module):
    """
    WaferCenterEstimator:
    Computes (cx, cy, R) representing wafer center coordinates and radius of gyration
    in normalized [0, 1] range.
    Uses a parameter-free closed-form calculation based on foreground mass (normal + defective die).
    An optional learned correction head can be enabled to adjust the predictions.
    """
    def __init__(self, learned_correction=False):
        super().__init__()
        self.learned_correction = learned_correction
        
        if learned_correction:
            # Learned correction head maps down features from soft probability maps
            self.correction_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(3 * 8 * 8, 32),
                nn.GELU(),
                nn.Linear(32, 3)  # Outputs (dcx, dcy, dR) corrections
            )

    def forward(self, soft_probs):
        # soft_probs shape: (B, 3, H, W)
        B, C, H, W = soft_probs.shape
        
        # Wafer foreground is composed of normal and defective dies (channels 1 and 2)
        # Shape: (B, H, W)
        mask = soft_probs[:, 1] + soft_probs[:, 2]
        
        # Calculate mass (number of die pixels)
        # Shape: (B, 1, 1)
        total_mass = torch.sum(mask, dim=[1, 2], keepdim=True)
        safe_mass = total_mass + 1e-8  # Prevent division by zero
        
        # Build coordinates grid in range [0.0, 1.0]
        y_grid = torch.arange(H, dtype=mask.dtype, device=mask.device).view(1, H, 1) / (H - 1)
        x_grid = torch.arange(W, dtype=mask.dtype, device=mask.device).view(1, 1, W) / (W - 1)
        
        # Compute centroid (cx, cy)
        # Shape: (B, 1, 1)
        cy = torch.sum(mask * y_grid, dim=[1, 2], keepdim=True) / safe_mass
        cx = torch.sum(mask * x_grid, dim=[1, 2], keepdim=True) / safe_mass
        
        # Compute radius of gyration (average distance of die pixels from center)
        # Shape: (B, 1, 1)
        r_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        radius_sq = torch.sum(mask * r_sq, dim=[1, 2], keepdim=True) / safe_mass
        R = torch.sqrt(radius_sq + 1e-8)
        
        # Flatten and combine
        cx = cx.squeeze(-1).squeeze(-1)
        cy = cy.squeeze(-1).squeeze(-1)
        R = R.squeeze(-1).squeeze(-1)
        
        coords = torch.stack([cx, cy, R], dim=-1)  # (B, 3)
        
        if self.learned_correction:
            # Predict corrections and add them to centroid/radius
            corrections = self.correction_head(soft_probs) * 0.1  # Initialized small
            coords = coords + corrections
            
        return coords
