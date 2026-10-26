import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import f1_score, accuracy_score

from data.transforms import get_transforms
from data.dataset import WaferMapDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EfficientNetV2Wafer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Use tf_efficientnetv2_s.in21k_ft_in1k or standard efficientnetv2_s
        self.backbone = timm.create_model('efficientnetv2_s', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        
        self.pattern_head = nn.Linear(num_features, 9)
        self.mechanism_head = nn.Linear(num_features, 9)
        self.yield_head = nn.Linear(num_features, 1)

    def forward(self, x, return_aux=True):
        features = self.backbone(x)  # shape: (B, num_features)
        
        pattern_logits = self.pattern_head(features)
        
        outputs = {
            "pattern_logits": pattern_logits
        }
        
        if return_aux:
            mechanism_logits = self.mechanism_head(features)
            yield_loss_pred = torch.sigmoid(self.yield_head(features)).squeeze(-1)
            
            outputs["mechanism_logits"] = mechanism_logits
            outputs["yield_loss_pred"] = yield_loss_pred
            
        return outputs

def main():
    parser = argparse.ArgumentParser(description="EfficientNetV2-S Wafer Baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    set_seed(config['training']['seed'])
    
    # Datasets
    train_transform, val_transform = get_transforms()
    train_dataset = WaferMapDataset(data_dir=config['dataset']['train_dir'], transform=train_transform)
    val_dataset = WaferMapDataset(data_dir=config['dataset']['val_dir'], transform=val_transform)
    test_dataset = WaferMapDataset(data_dir=config['dataset']['test_dir'], transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)
    
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    
    # Model
    print("Loading EfficientNetV2-S model...")
    model = EfficientNetV2Wafer(pretrained=True)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    criterion_pattern = nn.CrossEntropyLoss()
    criterion_mechanism = nn.CrossEntropyLoss()
    criterion_yield = nn.HuberLoss()
    
    w_pattern = config['loss_weights']['pattern']
    w_mechanism = config['loss_weights']['mechanism']
    w_yield = config['loss_weights']['yield']
    
    best_val_acc = 0.0
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch in train_loader:
            images = batch['image'].to(device)
            pattern_labels = batch['pattern_label'].to(device)
            mechanism_labels = batch['mechanism_label'].to(device)
            yield_proxies = batch['yield_proxy'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, return_aux=True)
            
            loss_pattern = criterion_pattern(outputs["pattern_logits"], pattern_labels)
            loss_mechanism = criterion_mechanism(outputs["mechanism_logits"], mechanism_labels)
            loss_yield = criterion_yield(outputs["yield_loss_pred"], yield_proxies)
            
            loss = w_pattern * loss_pattern + w_mechanism * loss_mechanism + w_yield * loss_yield
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = outputs["pattern_logits"].argmax(dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(pattern_labels.cpu().numpy())
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                pattern_labels = batch['pattern_label'].to(device)
                
                outputs = model(images, return_aux=False)
                preds = outputs["pattern_logits"].argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(pattern_labels.cpu().numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        print(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(checkpoint_dir, "efficientnetv2_s_best.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_acc': val_acc
            }, best_path)
            print(f"  Updated best checkpoint: {best_path}")
            
    # Test Evaluation
    print("\nRunning Test Evaluation...")
    best_path = os.path.join(checkpoint_dir, "efficientnetv2_s_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    model.eval()
    
    test_preds, test_targets = [], []
    all_mech_preds, all_mech_targets = [], []
    all_yield_preds, all_yield_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            pattern_labels = batch['pattern_label'].to(device)
            mechanism_labels = batch['mechanism_label'].to(device)
            yield_proxies = batch['yield_proxy'].to(device)
            
            outputs = model(images, return_aux=True)
            
            preds = outputs["pattern_logits"].argmax(dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(pattern_labels.cpu().numpy())
            
            m_preds = outputs["mechanism_logits"].argmax(dim=-1)
            all_mech_preds.extend(m_preds.cpu().numpy())
            all_mech_targets.extend(mechanism_labels.cpu().numpy())
            
            y_preds = outputs["yield_loss_pred"].cpu().numpy()
            all_yield_preds.extend(y_preds)
            all_yield_targets.extend(yield_proxies.cpu().numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    mech_acc = accuracy_score(all_mech_targets, all_mech_preds)
    yield_mae = np.mean(np.abs(np.array(all_yield_preds) - np.array(all_yield_targets)))
    
    num_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*50)
    print("EFFICIENTNETV2-S BASELINE RESULTS")
    print("="*50)
    print(f"Total Parameters: {num_params:,}")
    print(f"Pattern Classification Accuracy: {test_acc*100:.2f}%")
    print(f"Pattern Classification Macro-F1: {test_f1:.4f}")
    print(f"Mechanism Agreement: {mech_acc*100:.2f}%")
    print(f"Yield MAE: {yield_mae:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
