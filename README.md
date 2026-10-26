# GD-CoRPA: Gated Dual-Coordinate Relative Position Attention

A single-backbone Vision Transformer for multi-class semiconductor wafer defect classification, with physics-informed auxiliary prediction heads for estimated failure mechanism and expected yield loss.

---

## Table of Contents
1. [Overview](#overview)
2. [Why This Architecture](#why-this-architecture)
3. [Dataset](#dataset)
4. [Architecture](#architecture)
5. [Physics Knowledge Heads](#physics-knowledge-heads)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Ablations](#ablations)
12. [Baselines](#baselines)
13. [Limitations of Auxiliary Heads](#limitations-of-auxiliary-heads)
14. [Known Novelty Boundaries](#known-novelty-boundaries)
15. [Data Leakage / Deduplication Protocol](#data-leakage--deduplication-protocol)
16. [Citation](#citation)

---

## Overview
GD-CoRPA classifies 640×640×3 viridis-colormapped wafer bin maps into 9 classes — `Center`, `Donut`, `Edge-Loc`, `Edge-Ring`, `Local`, `Near-Full`, `Normal`, `Random`, `Scratch` — and additionally predicts, for each image:
- **Pattern** (the 9-class label — the primary task)
- **Estimated Failure Mechanism** (a physics-literature-derived proxy label)
- **Expected Yield Loss** (a defect-density-derived proxy regression target)

The goal of the two auxiliary heads is to make the model's output directly actionable for a process engineer, rather than stopping at a bare pattern label.

```
Pattern ──▶ Estimated Failure Mechanism
        └─▶ Expected Yield Loss
```

---

## Why This Architecture
Standard CNNs are translation-invariant, which makes them poorly suited to wafer defect patterns that are inherently *position-dependent relative to the wafer center*: `Center`, `Donut`, and `Edge-Ring` are naturally described in polar coordinates (radius `r`, angle `θ`), while `Scratch` is naturally described in Cartesian coordinates (a straight line has no privileged center).

The common approach in prior work is a **dual-backbone** design: one CNN/ViT encoder per coordinate system, joined by a late fusion block. This doubles backbone parameters and FLOPs, and the fusion decision is made once per sample, not per spatial location.

**GD-CoRPA uses one backbone.** Standard self-attention is augmented with *two* relative-position bias systems — Cartesian `(Δx, Δy)` and polar `(Δr, Δθ)`, the latter computed relative to a per-sample estimated wafer center — fused by a **content-conditioned gate computed per query token, per attention head, per layer**. This lets a single attention head, on a single image, weight polar evidence more heavily for tokens near the rim (useful for `Edge-Ring`/`Donut`) while weighting Cartesian evidence more heavily for tokens along a scratch line — without a second backbone.

---

## Dataset
| Split | Images per class | Total |
|---|---|---|
| Train | 623 | 5,607 |
| Validation | 78 | 702 |
| Test | 78 | 702 |

- **Format:** 640×640×3 RGB JPEG, rendered with the viridis colormap.
- **Known anchor colors:**
  - Background (outside wafer): `#440053` ≈ `[68, 0, 83]`
  - Normal die: `#1F908C` ≈ `[31, 144, 140]`
  - Defective die: `#FDE624` ≈ `[253, 230, 36]`
- **Dequantization:** JPEG compression blurs class boundaries into >50,000 unique colors per image rather than 3 discrete colors. The `FixedViridisDequantizer` module exists specifically to undo this.

---

## Architecture
```
Input (B, 3, 640, 640) RGB
        │
        ▼
┌───────────────────────────┐
│ FixedViridisDequantizer   │  fixed LUT anchors + learnable γ
│                           │  + small depthwise-separable residual conv
└───────────────────────────┘
        │  (B, 3, 640, 640) soft class-probability map
        ▼
┌───────────────────────────┐
│ WaferCenterEstimator      │  closed-form centroid + radius of gyration
│                           │  (optional learned correction head)
└───────────────────────────┘
        │  (cx, cy, R) per sample
        ▼
┌───────────────────────────┐
│ PatchEmbed                │  non-overlapping conv patchify
└───────────────────────────┘
        │  (B, N, D) tokens + [CLS]
        ▼
┌───────────────────────────┐
│ GDCoRPABlock × depth      │  self-attention with:
│                           │    - Cartesian relative-position bias (Δx, Δy)
│                           │    - Polar relative-position bias (Δr, Δθ)
│                           │    - content-conditioned gate g = σ(MLP(q)):
│                           │      attn += g·bias_cart + (1−g)·bias_polar
└───────────────────────────┘
        │  (B, N+1, D)
        ▼
┌───────────────────────────┐
│ LayerNorm → [CLS] pooled  │
└───────────────────────────┘
        │
        ├──▶ pattern_head    → (B, 9)                      [primary task]
        ├──▶ mechanism_head  → (B, 9)                      [auxiliary]
        └──▶ yield_head      → (B, 1)                      [auxiliary]
```
**Reference size:** depth=4, embed_dim=192, num_heads=6, patch_size=32 → **2.49M parameters** (2.38M trainable parameters in the backbone + heads, single forward pass, no warping).

---

## Physics Knowledge Heads

### Estimated Failure Mechanism
Maps each of the 9 pattern classes to its most likely root-cause failure mechanism:
| Pattern | Failure Mechanism |
|---|---|
| Center | Center-slow/fast process non-uniformity (e.g. CMP, spin-coat) |
| Donut | Edge-exclusion / implant-angle or deposition non-uniformity |
| Edge-Loc | Localized edge handling damage or edge-bead residue |
| Edge-Ring | Edge-focused process non-uniformity (etch, CMP, thermal gradient) |
| Local | Localized particle contamination or tool defect |
| Near-Full | Gross process failure (equipment fault, wrong recipe) |
| Normal | No defect mechanism |
| Random | Particle contamination, no systematic root cause |
| Scratch | Mechanical handling damage (transport, chuck) |

### Expected Yield Loss
Heuristically calculated as:
`yield_loss_proxy = (defect_die_count / total_die_count) × pattern_severity_weight`

---

## Repository Structure
```
gd_corpa/
├── data/
│   ├── dataset.py          # WaferMapDataset loader with yield cache
│   ├── dedup.py            # perceptual-hash dedup checker
│   ├── transforms.py       # spatial-only data transforms
│   └── yield_proxy.py      # defect-density-derived yield proxy calculation
├── model/
│   ├── dequantize.py       # FixedViridisDequantizer
│   ├── center_estimator.py # WaferCenterEstimator
│   ├── attention.py        # DualRelativePositionBias & GDCoRPAAttention
│   ├── blocks.py           # GDCoRPABlock & Mlp
│   ├── knowledge_base.py   # Pattern-to-mechanism dictionary
│   └── gd_corpa.py         # Full GDCoRPA model wrapper
├── configs/
│   └── default.yaml        # Training configurations
├── ablations/
│   ├── run_cart_only.py    # forced Cartesian bias (g ≡ 1)
│   ├── run_polar_only.py   # forced Polar bias (g ≡ 0)
│   ├── run_static_gate.py  # static learnable scalar per head
│   └── run_full.py         # proposed content-conditioned gate
├── baselines/
│   ├── efficientnetv2_s.py # EfficientNetV2 baseline
│   └── convnext_tiny.py    # ConvNeXt-Tiny baseline
├── train.py                # Multi-task training script
├── evaluate.py             # Evaluation and heatmap visualization
├── requirements.txt        # Package dependencies
└── README.md               # Documentation
```

---

## Installation
```bash
git clone <repo-url>
cd gd_corpa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage
```python
from model.gd_corpa import GDCoRPA
import torch

model = GDCoRPA(
    img_size=640, patch_size=32, embed_dim=192,
    depth=4, num_heads=6, num_classes=9,
    center_mode="closed_form"
)

x = torch.rand(2, 3, 640, 640)  # Normalized RGB in [0, 1]
outputs = model(x, return_aux=True)

print(outputs["pattern_logits"].shape)   # (2, 9)
print(outputs["mechanism_logits"].shape) # (2, 9)
print(outputs["yield_loss_pred"].shape)  # (2,)
```

---

## Training
To train the full model:
```bash
python train.py --config configs/default.yaml
```
Multi-task loss:
`total_loss = CE(pattern) + 0.3 * CE(mechanism) + 0.2 * Huber(yield) + 0.1 * center_regression + 1e-3 * gate_entropy`

---

## Evaluation
To evaluate a checkpoint and generate visualization plots (confusion matrix, scatter plot, and gate heatmaps overlaid on images):
```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --split test
```
Visualizations are saved directly to the configured outputs/artifacts directory.

---

## Ablations
Run specific ablation configs using the runner scripts:
- Cartesian bias only: `python ablations/run_cart_only.py`
- Polar bias only: `python ablations/run_polar_only.py`
- Static gate parameter: `python ablations/run_static_gate.py`
- Full proposed model: `python ablations/run_full.py`

---

## Baselines
Run baselines using their scripts:
- EfficientNetV2-S: `python baselines/efficientnetv2_s.py`
- ConvNeXt-Tiny: `python baselines/convnext_tiny.py`

---

## Limitations of Auxiliary Heads
1. **Failure-mechanism labels are a literature-derived proxy mapping, not diagnostic ground truth.**
2. **Yield-loss labels are a heuristic proxy** computed from defect density and a pattern-type severity multiplier — not measured fab yield data.

---

## Known Novelty Boundaries
- **Not novel:** Coordinate Attention (Hou et al., CVPR 2021); dual-branch Cartesian-polar fusion networks.
- **Specific contribution:** Dual relative-position bias fused by a *query-content-conditioned, per-token, per-head gate inside self-attention*, in a *single shared backbone*, rather than a post-hoc fusion block after two separate encoders.

---

## Data Leakage / Deduplication Protocol
Run the deduplication checker first:
```bash
python data/dedup.py --train-dir dataset/train --val-dir dataset/valid --test-dir dataset/test --hash-size 16 --hamming-threshold 5
```

---

## Citation
```
@misc{gdcorpa2026,
  title  = {GD-CoRPA: Gated Dual-Coordinate Relative Position Attention for Wafer Defect Classification},
  author = {Madhavan and Dharanika},
  year   = {2026},
  note   = {PSG Institute of Technology and Applied Research, Coimbatore}
}
```
