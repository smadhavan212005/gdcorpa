import torch
import torch.nn as nn
from model.dequantize import FixedViridisDequantizer
from model.center_estimator import WaferCenterEstimator
from model.blocks import GDCoRPABlock

class PatchEmbed(nn.Module):
    """ Non-overlapping convolutional patch projection. """
    def __init__(self, img_size=640, patch_size=32, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input shape: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

class GDCoRPA(nn.Module):
    """
    GD-CoRPA: Gated Dual-Coordinate Relative Position Attention Vision Transformer.
    Single-backbone wafer defect classification network with auxiliary physics heads.
    """
    def __init__(self, img_size=640, patch_size=32, embed_dim=192, depth=4, num_heads=6,
                 num_classes=9, num_mechanism_classes=9, center_mode="closed_form",
                 drop_rate=0., attn_drop_rate=0., mlp_ratio=4., ablation_mode=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # 1. Dequantizer & Wafer Center Estimator
        self.dequantizer = FixedViridisDequantizer()
        self.center_estimator = WaferCenterEstimator(learned_correction=(center_mode == "learned"))
        
        # 2. Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim
        )
        self.grid_size = img_size // patch_size
        self.num_patches = self.patch_embed.num_patches
        
        # 3. Transformer tokens & positions
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 4. GD-CoRPA Blocks
        self.blocks = nn.ModuleList([
            GDCoRPABlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                grid_size=self.grid_size, ablation_mode=ablation_mode
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 5. Prediction Heads
        # Primary head
        self.pattern_head = nn.Linear(embed_dim, num_classes)
        # Auxiliary failure mechanism head
        self.mechanism_head = nn.Linear(embed_dim, num_mechanism_classes)
        # Auxiliary yield loss head (produces fraction in [0, 1])
        self.yield_head = nn.Linear(embed_dim, 1)
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_aux=True):
        """
        Args:
            x: Input wafer map image tensor of shape (B, 3, 640, 640)
            return_aux: Whether to return auxiliary predictions
        Returns:
            dict of outputs containing pattern_logits, mechanism_logits, yield_loss_pred, center_coords, gates, soft_probs
        """
        B = x.shape[0]
        
        # 1. Dequantize colors to class probability maps
        soft_probs = self.dequantizer(x)
        
        # 2. Estimate wafer center and radius
        center_coords = self.center_estimator(soft_probs)
        
        # 3. Patchify and add tokens
        tokens = self.patch_embed(soft_probs)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        
        # 4. Process tokens through GD-CoRPA blocks
        gates = []
        for block in self.blocks:
            tokens, gate = block(tokens, center_coords)
            gates.append(gate)
            
        tokens = self.norm(tokens)
        
        # 5. Extract CLS representation and classify
        cls_pooled = tokens[:, 0]
        
        pattern_logits = self.pattern_head(cls_pooled)
        
        outputs = {
            "pattern_logits": pattern_logits,
            "center_coords": center_coords,
            "gates": gates,
            "soft_probs": soft_probs
        }
        
        if return_aux:
            mechanism_logits = self.mechanism_head(cls_pooled)
            # Use sigmoid to clamp predicted yield loss strictly to [0.0, 1.0]
            yield_loss_pred = torch.sigmoid(self.yield_head(cls_pooled)).squeeze(-1)
            
            outputs["mechanism_logits"] = mechanism_logits
            outputs["yield_loss_pred"] = yield_loss_pred
            
        return outputs
