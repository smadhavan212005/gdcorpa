import numpy as np
from PIL import Image

# Known anchor colors for viridis wafer maps in RGB space (normalized to [0, 1])
ANCHOR_COLORS = {
    'background': np.array([68.0, 0.0, 83.0]) / 255.0,  # #440053
    'normal': np.array([31.0, 144.0, 140.0]) / 255.0,    # #1F908C
    'defective': np.array([253.0, 230.0, 36.0]) / 255.0  # #FDE624
}

# Heuristic pattern severity weights
PATTERN_SEVERITY_WEIGHTS = {
    'Normal': 1.0,
    'Random': 1.1,
    'Edge-Loc': 1.2,
    'Local': 1.3,
    'Scratch': 1.4,
    'Center': 1.5,
    'Donut': 1.6,
    'Edge-Ring': 1.8,
    'Near-Full': 2.0
}

def compute_yield_loss_proxy(image_path_or_array, pattern_class):
    """
    Computes a proxy yield loss target from the wafer map:
    yield_loss_proxy = (defect_die_count / total_die_count) * pattern_severity_weight
    
    Args:
        image_path_or_array: String path or PIL Image / NumPy array (H, W, 3).
        pattern_class: String name of the defect pattern class.
        
    Returns:
        float: yield loss proxy in [0.0, 1.0].
    """
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0
    elif isinstance(image_path_or_array, Image.Image):
        img_np = np.array(image_path_or_array.convert('RGB'), dtype=np.float32) / 255.0
    else:
        img_np = np.array(image_path_or_array, dtype=np.float32)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    # Reshape if necessary (make sure it's H x W x 3)
    if len(img_np.shape) == 4:
        img_np = img_np[0]
        
    # Get distances to anchors
    bg_dist = np.sum((img_np - ANCHOR_COLORS['background'])**2, axis=-1)
    norm_dist = np.sum((img_np - ANCHOR_COLORS['normal'])**2, axis=-1)
    def_dist = np.sum((img_np - ANCHOR_COLORS['defective'])**2, axis=-1)
    
    # Classify each pixel
    # 0 = background, 1 = normal, 2 = defective
    dist_stack = np.stack([bg_dist, norm_dist, def_dist], axis=-1)
    pixel_classes = np.argmin(dist_stack, axis=-1)
    
    # Count normal and defective dies
    defect_die_count = np.sum(pixel_classes == 2)
    normal_die_count = np.sum(pixel_classes == 1)
    
    total_die_count = defect_die_count + normal_die_count
    
    if total_die_count == 0:
        return 0.0
        
    defect_ratio = defect_die_count / total_die_count
    severity_weight = PATTERN_SEVERITY_WEIGHTS.get(pattern_class, 1.0)
    
    yield_loss = defect_ratio * severity_weight
    return float(np.clip(yield_loss, 0.0, 1.0))
