import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from model.knowledge_base import PATTERN_CLASSES, PATTERN_IDX_TO_MECHANISM_IDX
from data.yield_proxy import compute_yield_loss_proxy

class WaferMapDataset(Dataset):
    """
    Dataset loader for Wafer Bin Maps.
    Returns:
        - Image tensor
        - Pattern class index (0-8)
        - Failure mechanism index (0-8)
        - Yield loss proxy (0.0-1.0)
    """
    def __init__(self, data_dir, transform=None, cache_dir=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Determine the split name from the folder
        split_name = os.path.basename(os.path.normpath(data_dir))
        if not cache_dir:
            cache_dir = data_dir
        self.cache_file = os.path.join(cache_dir, f"{split_name}_yield_cache.json")
        
        # 1. Collect all images and classes
        print(f"Loading dataset samples from {data_dir}...")
        for class_idx, class_name in enumerate(PATTERN_CLASSES):
            class_folder = os.path.join(data_dir, class_name)
            if not os.path.exists(class_folder):
                continue
                
            for filename in os.listdir(class_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_folder, filename)
                    mech_idx = PATTERN_IDX_TO_MECHANISM_IDX[class_idx]
                    self.samples.append({
                        'path': img_path,
                        'pattern_idx': class_idx,
                        'mechanism_idx': mech_idx,
                        'class_name': class_name
                    })
                    
        print(f"Found {len(self.samples)} samples in {data_dir}.")
        
        # 2. Load or compute yield proxy cache
        self.yield_proxies = {}
        if os.path.exists(self.cache_file):
            print(f"Loading precomputed yield proxy labels from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'r') as f:
                    self.yield_proxies = json.load(f)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")
                self.yield_proxies = {}
                
        # Fill missing values
        missing_samples = [s for s in self.samples if s['path'] not in self.yield_proxies]
        if missing_samples:
            print(f"Computing yield proxy labels for {len(missing_samples)} new/uncached samples...")
            for s in tqdm(missing_samples, desc="Yield Proxy Calculation"):
                path = s['path']
                try:
                    y_val = compute_yield_loss_proxy(path, s['class_name'])
                    self.yield_proxies[path] = y_val
                except Exception as e:
                    print(f"Error computing yield proxy for {path}: {e}")
                    self.yield_proxies[path] = 0.0
                    
            # Save updated cache
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.yield_proxies, f, indent=2)
                print(f"Saved yield proxy cache to {self.cache_file}")
            except Exception as e:
                print(f"Failed to save cache to {self.cache_file}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        pattern_idx = sample['pattern_idx']
        mechanism_idx = sample['mechanism_idx']
        yield_val = self.yield_proxies.get(img_path, 0.0)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.tensor(img_path)  # Fallback
            
        return {
            'image': img_tensor,
            'pattern_label': torch.tensor(pattern_idx, dtype=torch.long),
            'mechanism_label': torch.tensor(mechanism_idx, dtype=torch.long),
            'yield_proxy': torch.tensor(yield_val, dtype=torch.float32),
            'path': img_path
        }
