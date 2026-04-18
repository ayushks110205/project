import os
# FIX 1: Add this at the VERY TOP to prevent the OMP Error #15 crash
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DeepGlobeRoadDataset, train_transform 
from models import get_road_model

def run_visual_check(model_path, image_idx=5):
    device = torch.device("cpu")
    
    model = get_road_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = DeepGlobeRoadDataset(
        image_dir='datasets/train', 
        mask_dir='datasets/train', 
        transform=train_transform 
    )
    
    image, mask = dataset[image_idx]
    input_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_pred = (prediction_prob > 0.5).astype(np.float32)

    # --- FIX 2: UN-NORMALIZE THE IMAGE FOR DISPLAY ---
    # These are the standard ImageNet values we used in dataset.py
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert tensor back to visible numpy image
    vis_img = image.permute(1, 2, 0).cpu().numpy()
    vis_img = std * vis_img + mean  # Reversing the math
    vis_img = np.clip(vis_img, 0, 1) # Ensuring values stay in valid 0-1 range

    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    
    # Show the "Healed" Satellite Image
    ax[0].imshow(vis_img)
    ax[0].set_title("Input Satellite (Healed)")
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth (Roads)")
    
    im = ax[2].imshow(prediction_prob, cmap='hot') 
    ax[2].set_title("Confidence Heatmap")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    
    ax[3].imshow(binary_pred, cmap='gray')
    ax[3].set_title("Final Binary Prediction")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visual_check(model_path="best_model.pth", image_idx=5)