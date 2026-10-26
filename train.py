import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from data.transforms import get_transforms
from data.dataset import WaferMapDataset
from model.gd_corpa import GDCoRPA

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="GD-CoRPA Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--ablation-mode", type=str, default=None, choices=["cart_only", "polar_only", "static_gate", "full"], help="Ablation mode override")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    args = parser.parse_args()
    
    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.ablation_mode:
        config['model']['ablation_mode'] = args.ablation_mode
    if args.epochs:
        config['training']['epochs'] = args.epochs
        
    set_seed(config['training']['seed'])
    
    # Create checkpoints directory
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2. Datasets & Dataloaders
    train_transform, val_transform = get_transforms()
    
    train_dataset = WaferMapDataset(
        data_dir=config['dataset']['train_dir'],
        transform=train_transform
    )
    val_dataset = WaferMapDataset(
        data_dir=config['dataset']['val_dir'],
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 3. Model
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    
    model = GDCoRPA(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        depth=config['model']['depth'],
        num_heads=config['model']['num_heads'],
        num_classes=config['model']['num_classes'],
        num_mechanism_classes=config['model']['num_mechanism_classes'],
        center_mode=config['model']['center_mode'],
        drop_rate=config['model']['drop_rate'],
        attn_drop_rate=config['model']['attn_drop_rate'],
        ablation_mode=config['model'].get('ablation_mode', None)
    )
    model.to(device)
    
    # 4. Optimizer and Losses
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion_pattern = nn.CrossEntropyLoss()
    criterion_mechanism = nn.CrossEntropyLoss()
    criterion_yield = nn.HuberLoss()
    
    # Extract loss weights
    w_pattern = config['loss_weights']['pattern']
    w_mechanism = config['loss_weights']['mechanism']
    w_yield = config['loss_weights']['yield']
    w_center = config['loss_weights']['center']
    w_gate_entropy = config['loss_weights']['gate_entropy']
    
    best_val_acc = 0.0
    
    print("\nStarting training loop...")
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        train_losses = {
            'total': 0.0, 'pattern': 0.0, 'mechanism': 0.0, 
            'yield': 0.0, 'center': 0.0, 'gate_entropy': 0.0
        }
        
        train_preds, train_targets = [], []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            pattern_labels = batch['pattern_label'].to(device)
            mechanism_labels = batch['mechanism_label'].to(device)
            yield_proxies = batch['yield_proxy'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, return_aux=True)
            
            # Calculate standard losses
            loss_pattern = criterion_pattern(outputs["pattern_logits"], pattern_labels)
            loss_mechanism = criterion_mechanism(outputs["mechanism_logits"], mechanism_labels)
            loss_yield = criterion_yield(outputs["yield_loss_pred"], yield_proxies)
            
            # Calculate center regression loss
            # Target center is the closed-form centroid without correction, detached
            center_pred = outputs["center_coords"]
            if model.center_estimator.learned_correction:
                # Retrieve correction values from the estimator
                # (since corrections = correction_head(soft_probs) * 0.1, we can compute closed-form target)
                with torch.no_grad():
                    # Closed-form coords can be computed by running the estimator on soft_probs but detaching correction
                    # Actually, we can run WaferCenterEstimator but temporarily override learned_correction or do it simply:
                    corrections = model.center_estimator.correction_head(outputs["soft_probs"]) * 0.1
                    target_center = (center_pred - corrections).detach()
                loss_center = F.mse_loss(center_pred, target_center)
            else:
                # If closed-form only, center regression is 0.0
                loss_center = torch.tensor(0.0, device=device)
                
            # Calculate gate entropy loss to prevent saturation/collapse
            entropy_losses = []
            for gate in outputs["gates"]:
                # gate shape: (B, H, N_patches, 1)
                # binary entropy: -g * log(g) - (1-g) * log(1-g)
                entropy = - (gate * torch.log(gate + 1e-6) + (1.0 - gate) * torch.log(1.0 - gate + 1e-6))
                entropy_losses.append(entropy.mean())
            loss_gate_entropy = - torch.stack(entropy_losses).mean() if entropy_losses else torch.tensor(0.0, device=device)
            
            # Total multi-task loss
            loss_total = (
                w_pattern * loss_pattern +
                w_mechanism * loss_mechanism +
                w_yield * loss_yield +
                w_center * loss_center +
                w_gate_entropy * loss_gate_entropy
            )
            
            loss_total.backward()
            optimizer.step()
            
            # Record losses
            train_losses['total'] += loss_total.item()
            train_losses['pattern'] += loss_pattern.item()
            train_losses['mechanism'] += loss_mechanism.item()
            train_losses['yield'] += loss_yield.item()
            train_losses['center'] += loss_center.item()
            train_losses['gate_entropy'] += loss_gate_entropy.item()
            
            # Predictions for training accuracy
            preds = outputs["pattern_logits"].argmax(dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(pattern_labels.cpu().numpy())
            
            if (batch_idx + 1) % config['training']['log_freq'] == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {loss_total.item():.4f} (Pat: {loss_pattern.item():.4f}, Mech: {loss_mechanism.item():.4f}, "
                      f"Yield: {loss_yield.item():.4f}, Center: {loss_center.item():.4f}, GateEnt: {loss_gate_entropy.item():.4f})")
                      
        # Normalize losses
        for k in train_losses:
            train_losses[k] /= len(train_loader)
            
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0.0, 'pattern': 0.0}
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                pattern_labels = batch['pattern_label'].to(device)
                mechanism_labels = batch['mechanism_label'].to(device)
                yield_proxies = batch['yield_proxy'].to(device)
                
                outputs = model(images, return_aux=True)
                
                loss_pattern = criterion_pattern(outputs["pattern_logits"], pattern_labels)
                loss_mechanism = criterion_mechanism(outputs["mechanism_logits"], mechanism_labels)
                loss_yield = criterion_yield(outputs["yield_loss_pred"], yield_proxies)
                
                loss_total = (
                    w_pattern * loss_pattern +
                    w_mechanism * loss_mechanism +
                    w_yield * loss_yield
                )
                
                val_losses['total'] += loss_total.item()
                val_losses['pattern'] += loss_pattern.item()
                
                preds = outputs["pattern_logits"].argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(pattern_labels.cpu().numpy())
                
        for k in val_losses:
            val_losses[k] /= len(val_loader)
            
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        # Log gate mean values for diagnostics
        mean_gates = []
        with torch.no_grad():
            # Get a sample batch to check gates
            sample_batch = next(iter(val_loader))
            sample_imgs = sample_batch['image'].to(device)
            sample_out = model(sample_imgs, return_aux=False)
            for i, gate in enumerate(sample_out["gates"]):
                mean_gates.append(f"L{i}: {gate.mean().item():.3f}")
                
        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {train_losses['total']:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_losses['total']:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1:   {val_f1:.4f}")
        print(f"Gating values (Val sample) -> " + ", ".join(mean_gates))
        print("-" * 30 + "\n")
        
        # Save checkpoints
        if epoch % config['training']['save_freq'] == 0 or val_acc > best_val_acc:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, best_path)
            print(f"Updated best model checkpoint: {best_path}")

if __name__ == "__main__":
    main()
