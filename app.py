import os
import shutil
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

from model.gd_corpa import GDCoRPA
from model.knowledge_base import PATTERN_CLASSES, PATTERN_TO_MECHANISM
from model.localization import find_defect_bboxes
from data.transforms import get_transforms

app = Flask(__name__)

# Constants
CHECKPOINT_PATH = "checkpoints/best_model.pt"
STATIC_DIR = "static"
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")
TEMP_DIR = os.path.join(STATIC_DIR, "temp_predictions")
TEST_DATA_DIR = "dataset/test"

# Create directories
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Clean up old temp predictions on startup
for filename in os.listdir(TEMP_DIR):
    file_path = os.path.join(TEMP_DIR, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting old temp file {file_path}: {e}")

# Pre-populate sample images from dataset/test if available
def populate_samples():
    if not os.path.exists(TEST_DATA_DIR):
        print(f"Test dataset directory {TEST_DATA_DIR} not found. Cannot populate samples.")
        return
    
    print("Populating sample wafer maps...")
    for class_name in PATTERN_CLASSES:
        class_folder = os.path.join(TEST_DATA_DIR, class_name)
        if os.path.exists(class_folder):
            files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                src_path = os.path.join(class_folder, sorted(files)[0])
                dest_path = os.path.join(SAMPLES_DIR, f"{class_name}.jpg")
                try:
                    shutil.copy(src_path, dest_path)
                    print(f"  Copied sample for {class_name}")
                except Exception as e:
                    print(f"  Failed to copy sample for {class_name}: {e}")

populate_samples()

# Load Model
device = torch.device("cpu")
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Please train a model first.")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
config = checkpoint['config']

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

_, val_transform = get_transforms()

def get_samples_list():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for filename in sorted(os.listdir(SAMPLES_DIR)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = os.path.splitext(filename)[0]
                samples.append({
                    'class': class_name,
                    'url': f'/static/samples/{filename}'
                })
    return samples

@app.route('/', methods=['GET', 'POST'])
def index():
    samples = get_samples_list()
    results = None
    error = None
    
    if request.method == 'POST':
        try:
            # Check if it is a sample prediction or an upload
            sample_class = request.form.get('sample_class')
            
            if sample_class:
                img_path = os.path.join(SAMPLES_DIR, f"{sample_class}.jpg")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Sample image for {sample_class} not found.")
                img = Image.open(img_path).convert('RGB')
            else:
                if 'image' not in request.files:
                    raise ValueError("No image file part in request.")
                file = request.files['image']
                if file.filename == '':
                    raise ValueError("No file selected. Please select a wafer map image.")
                img = Image.open(file.stream).convert('RGB')
            
            # Save original image to temp folder
            task_id = str(uuid.uuid4())
            orig_filename = f"orig_{task_id}.png"
            orig_path = os.path.join(TEMP_DIR, orig_filename)
            
            # Resize image to standard 640x640
            img_resized = img.resize((640, 640), Image.Resampling.BILINEAR)
            img_resized.save(orig_path)
            
            # Preprocess
            img_tensor = val_transform(img_resized).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(img_tensor, return_aux=True)
                
            # Pattern prediction
            pattern_logits = outputs["pattern_logits"]
            pattern_probs = F.softmax(pattern_logits, dim=-1).squeeze().numpy()
            pattern_pred_idx = int(pattern_probs.argmax())
            predicted_pattern = PATTERN_CLASSES[pattern_pred_idx]
            pattern_confidence = float(pattern_probs[pattern_pred_idx])
            
            # Failure Mechanism
            predicted_mechanism = PATTERN_TO_MECHANISM.get(predicted_pattern, "Unknown")
            
            # Yield Loss
            yield_loss = float(outputs["yield_loss_pred"].squeeze().numpy())
            
            # Defect localization: derive bounding boxes from the dequantizer's
            # per-pixel soft class map (no separate detection head required)
            soft_probs_np = outputs["soft_probs"].squeeze(0).numpy()  # (3, H, W)
            defect_boxes = find_defect_bboxes(soft_probs_np)

            localized_img = img_resized.copy()
            draw = ImageDraw.Draw(localized_img)
            for (bx0, by0, bx1, by1) in defect_boxes:
                draw.rectangle([bx0, by0, bx1, by1], outline=(255, 45, 45), width=5)

            localized_filename = f"localized_{task_id}.png"
            localized_path = os.path.join(TEMP_DIR, localized_filename)
            localized_img.save(localized_path)

            # Attention gates
            gate_tensor = outputs["gates"][-1].squeeze(0).squeeze(-1) # shape (H_heads, N_patches)
            mean_gate = gate_tensor.mean(dim=0)
            
            grid_size = config['model']['img_size'] // config['model']['patch_size']
            gate_grid = mean_gate.view(grid_size, grid_size).numpy()
            
            # Gating overlay
            cmap = plt.get_cmap('coolwarm')
            heatmap_rgba = cmap(gate_grid)
            heatmap_img = Image.fromarray((heatmap_rgba * 255).astype(np.uint8))
            heatmap_img = heatmap_img.resize((640, 640), Image.Resampling.BILINEAR)
            
            overlay_filename = f"gate_{task_id}.png"
            overlay_path = os.path.join(TEMP_DIR, overlay_filename)
            heatmap_img.save(overlay_path)
            
            # Probability breakdown
            pattern_confidences = []
            for i, p_class in enumerate(PATTERN_CLASSES):
                pattern_confidences.append({
                    'class': p_class,
                    'prob': float(pattern_probs[i]),
                    'prob_percent': float(pattern_probs[i]) * 100.0
                })
            # Sort descending
            pattern_confidences.sort(key=lambda x: x['prob'], reverse=True)
            
            results = {
                'predicted_class': predicted_pattern,
                'pattern_confidence': pattern_confidence,
                'pattern_confidence_percent': pattern_confidence * 100.0,
                'predicted_mechanism': predicted_mechanism,
                'yield_loss_percentage': yield_loss * 100.0,
                'original_image_url': f'/static/temp_predictions/{orig_filename}',
                'localized_image_url': f'/static/temp_predictions/{localized_filename}',
                'num_defect_regions': len(defect_boxes),
                'gate_overlay_url': f'/static/temp_predictions/{overlay_filename}',
                'pattern_confidences': pattern_confidences,
                'selected_sample': sample_class
            }
            
        except Exception as e:
            error = str(e)
            print(f"Error during POST processing: {error}")
            import traceback
            traceback.print_exc()
            
    return render_template('index.html', samples=samples, results=results, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
