import os
# Prevent the OMP duplicate library crash
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DeepGlobeLandCoverDataset, train_transform 
from models import get_landcover_model

# Official DeepGlobe Color Palette
COLOR_DICT = {
    0: [0, 255, 255],   # Urban (Cyan)
    1: [255, 255, 0],   # Agriculture (Yellow)
    2: [255, 0, 255],   # Rangeland (Magenta)
    3: [0, 255, 0],     # Forest (Green)
    4: [0, 0, 255],     # Water (Blue)
    5: [255, 255, 255], # Barren (White)
    6: [0, 0, 0]        # Unknown (Black)
}

def id_to_rgb(mask_id):
    """Converts a 2D integer ID map into a 3D RGB color image."""
    h, w = mask_id.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in COLOR_DICT.items():
        rgb_mask[mask_id == idx] = color
    return rgb_mask

def run_landcover_check(model_path="landcover_best.pth", image_idx=5):
    device = torch.device("cpu")
    
    # 1. Load Model
    model = get_landcover_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Data
    dataset = DeepGlobeLandCoverDataset(
        image_dir='datasets/train', 
        mask_dir='datasets/train', 
        transform=train_transform 
    )
    image, mask_id = dataset[image_idx]
    
    # 3. Inference
    input_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        # For multi-class, we take the 'argmax' to find the most likely class per pixel
        prediction_id = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 4. Prepare for Display
    # Convert integer IDs back to RGB colors
    gt_rgb = id_to_rgb(mask_id.numpy() if torch.is_tensor(mask_id) else mask_id)
    pred_rgb = id_to_rgb(prediction_id)

    # Un-normalize satellite image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    vis_img = image.permute(1, 2, 0).cpu().numpy()
    vis_img = std * vis_img + mean
    vis_img = np.clip(vis_img, 0, 1)

    # 5. Plotting
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(vis_img)
    ax[0].set_title("Input Satellite Image")
    
    ax[1].imshow(gt_rgb)
    ax[1].set_title("Ground Truth (Land Cover)")
    
    ax[2].imshow(pred_rgb)
    ax[2].set_title("Model Prediction")
    
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_landcover_check(model_path="landcover_best.pth", image_idx=10)