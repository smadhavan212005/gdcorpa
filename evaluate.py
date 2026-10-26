import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from data.transforms import get_transforms
from data.dataset import WaferMapDataset
from model.gd_corpa import GDCoRPA
from model.knowledge_base import PATTERN_CLASSES, MECHANISM_CLASSES

def main():
    parser = argparse.ArgumentParser(description="GD-CoRPA Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"], help="Dataset split to evaluate")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory to save artifacts and plots")
    args = parser.parse_args()
    
    # 1. Load checkpoint and config
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint['config']
    
    # Override split paths based on selection
    if args.split == "train":
        data_dir = config['dataset']['train_dir']
    elif args.split == "valid":
        data_dir = config['dataset']['val_dir']
    else:
        data_dir = config['dataset']['test_dir']
        
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # 2. Dataset and Loader
    _, val_transform = get_transforms()
    dataset = WaferMapDataset(data_dir=data_dir, transform=val_transform)
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)
    
    # 3. Model
    device = torch.device("cpu")  # Force CPU evaluation
    print(f"Evaluating on device: {device}")
    
    model = GDCoRPA(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        num_classes=config['model']['num_classes'],
        num_mechanism_classes=config['model']['num_mechanism_classes'],
        center_mode=config['model']['center_mode']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 4. Evaluation Loop
    all_pattern_preds = []
    all_pattern_targets = []
    all_mech_preds = []
    all_mech_targets = []
    all_yield_preds = []
    all_yield_targets = []
    
    # Gather samples for gate visualization (one from each class if possible)
    class_samples = {c: None for c in PATTERN_CLASSES}
    
    print("Running inference...")
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            pattern_labels = batch['pattern_label'].to(device)
            mechanism_labels = batch['mechanism_label'].to(device)
            yield_proxies = batch['yield_proxy'].to(device)
            paths = batch['path']
            
            outputs = model(images, return_aux=True)
            
            # Pattern
            p_preds = outputs["pattern_logits"].argmax(dim=-1).cpu().numpy()
            all_pattern_preds.extend(p_preds)
            all_pattern_targets.extend(pattern_labels.cpu().numpy())
            
            # Mechanism
            m_preds = outputs["mechanism_logits"].argmax(dim=-1).cpu().numpy()
            all_mech_preds.extend(m_preds)
            all_mech_targets.extend(mechanism_labels.cpu().numpy())
            
            # Yield
            y_preds = outputs["yield_loss_pred"].cpu().numpy()
            all_yield_preds.extend(y_preds)
            all_yield_targets.extend(yield_proxies.cpu().numpy())
            
            # Collect sample images for visualization
            for idx, label_idx in enumerate(pattern_labels.cpu().numpy()):
                class_name = PATTERN_CLASSES[label_idx]
                if class_samples[class_name] is None:
                    # Save image tensor, gate values, predictions, and path
                    # gates is a list of blocks, take the last block's gate values: outputs["gates"][-1]
                    # gate shape: (B, H, N_patches, 1)
                    class_samples[class_name] = {
                        'image': images[idx].cpu(),
                        'gate': outputs["gates"][-1][idx].cpu(), # Last layer's gate: (H, N_p, 1)
                        'pattern_pred': p_preds[idx],
                        'path': paths[idx]
                    }

    # Convert to arrays
    all_pattern_preds = np.array(all_pattern_preds)
    all_pattern_targets = np.array(all_pattern_targets)
    all_mech_preds = np.array(all_mech_preds)
    all_mech_targets = np.array(all_mech_targets)
    all_yield_preds = np.array(all_yield_preds)
    all_yield_targets = np.array(all_yield_targets)
    
    # 5. Calculate Metrics
    # Pattern
    pat_acc = accuracy_score(all_pattern_targets, all_pattern_preds)
    pat_f1 = f1_score(all_pattern_targets, all_pattern_preds, average='macro')
    
    # Mechanism
    mech_acc = accuracy_score(all_mech_targets, all_mech_preds)
    
    # Yield
    yield_mae = np.mean(np.abs(all_yield_preds - all_yield_targets))
    yield_rmse = np.sqrt(np.mean((all_yield_preds - all_yield_targets)**2))
    
    print("\n" + "="*50)
    print(f"GD-CoRPA EVALUATION ON {args.split.upper()} SPLIT")
    print("="*50)
    print(f"Pattern Classification Accuracy: {pat_acc*100:.2f}%")
    print(f"Pattern Classification Macro-F1: {pat_f1:.4f}")
    print(f"Mechanism Head Agreement (against proxy): {mech_acc*100:.2f}%")
    print(f"Yield Head MAE:  {yield_mae:.4f}")
    print(f"Yield Head RMSE: {yield_rmse:.4f}")
    print("="*50)
    
    print("\nDetailed Pattern Classification Report:")
    print(classification_report(all_pattern_targets, all_pattern_preds, target_names=PATTERN_CLASSES))
    
    # 6. Save Plots
    # Plot 1: Confusion Matrix
    cm = confusion_matrix(all_pattern_targets, all_pattern_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=PATTERN_CLASSES, yticklabels=PATTERN_CLASSES)
    plt.title('GD-CoRPA Wafer Defect Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(args.artifacts_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # Plot 2: Yield Scatter Plot
    plt.figure(figsize=(10, 8))
    for class_idx, class_name in enumerate(PATTERN_CLASSES):
        indices = np.where(all_pattern_targets == class_idx)[0]
        if len(indices) > 0:
            plt.scatter(
                all_yield_targets[indices], 
                all_yield_preds[indices], 
                label=class_name, 
                alpha=0.6,
                edgecolors='none'
            )
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.title('Yield Loss: Predicted vs. Heuristic Proxy')
    plt.xlabel('Heuristic Proxy Yield Loss')
    plt.ylabel('Predicted Yield Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(args.artifacts_dir, 'yield_scatter.png')
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved yield loss scatter plot to {scatter_path}")
    
    # Plot 3: Gate Visualization Overlays
    grid_size = config['model']['img_size'] // config['model']['patch_size']
    
    print("\nGenerating gate visualizations...")
    for class_name, sample in class_samples.items():
        if sample is None:
            continue
            
        # Get raw image (C, H, W) -> convert to (H, W, C)
        img_np = sample['image'].permute(1, 2, 0).numpy()
        
        # Mean gate value across attention heads
        # gate shape: (H_heads, N_patches, 1) -> squeeze to (H_heads, N_patches)
        gate_val = sample['gate'].squeeze(-1)
        mean_gate = gate_val.mean(dim=0)  # shape (N_patches,)
        
        # Reshape to 2D grid
        gate_grid = mean_gate.view(grid_size, grid_size).numpy()
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Original Wafer image
        axes[0].imshow(img_np)
        axes[0].set_title(f"Wafer Map: {class_name}")
        axes[0].axis('off')
        
        # 2. Wafer with Gating Heatmap overlay
        # g -> 1: Cartesian (Red), g -> 0: Polar (Blue)
        axes[1].imshow(img_np)
        # Interpolate heatmap to original image size (640x640)
        im = axes[1].imshow(
            gate_grid, 
            cmap='coolwarm', 
            alpha=0.6, 
            extent=[0, 640, 640, 0], 
            vmin=0.0, 
            vmax=1.0
        )
        axes[1].set_title(f"GD-CoRPA Gate overlay (Red=Cartesian, Blue=Polar)\nPred: {PATTERN_CLASSES[sample['pattern_pred']]}")
        axes[1].axis('off')
        
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        vis_path = os.path.join(args.artifacts_dir, f'gate_vis_{class_name}.png')
        plt.savefig(vis_path)
        plt.close()
        print(f"  Saved gate overlay for {class_name} to {vis_path}")
        
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
