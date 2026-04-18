import os
# Prevents crashes on some local Windows/Mac environments
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DeepGlobeBuildingDataset, val_transform
from models import get_building_model

def run_building_check(model_path, image_idx=10, save_result=True):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🏠 Visualizing Buildings on: {device}")

    # 2. Load the Trained Model
    if not os.path.exists(model_path):
        print(f"❌ Error: Building weights not found at {model_path}")
        return

    model = get_building_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Load Dataset
    dataset = DeepGlobeBuildingDataset(
        image_dir='datasets/train', 
        mask_dir='datasets/train', 
        transform=val_transform 
    )
    
    image, mask = dataset[image_idx]
    input_tensor = image.unsqueeze(0).to(device)
    
    # 4. Perform Inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_pred = (prediction_prob > 0.5).astype(np.float32)

    # 5. --- UN-NORMALIZE FOR VISUAL POLISH ---
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    vis_img = image.permute(1, 2, 0).cpu().numpy()
    vis_img = std * vis_img + mean 
    vis_img = np.clip(vis_img, 0, 1) 

    # 6. Create the Comparison Plot
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    
    ax[0].imshow(vis_img)
    ax[0].set_title("Input Satellite", fontsize=14, fontweight='bold')
    ax[0].axis('off')
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Ground Truth (Buildings)", fontsize=14, fontweight='bold')
    ax[1].axis('off')
    
    # Using 'inferno' for buildings gives a high-contrast look
    im = ax[2].imshow(prediction_prob, cmap='inferno') 
    ax[2].set_title("Footprint Confidence", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[2].axis('off')
    
    ax[3].imshow(binary_pred, cmap='gray')
    ax[3].set_title("Final Detection", fontsize=14, fontweight='bold')
    ax[3].axis('off')
    
    plt.tight_layout()
    
    # 7. Save to the results folder
    if save_result:
        os.makedirs("results/buildings", exist_ok=True)
        out_path = f"results/buildings/check_idx_{image_idx}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"✅ Success! Building check saved to: {out_path}")
        
    plt.show()

if __name__ == "__main__":
    # Check for the Drive model first
    DRIVE_PATH = "/content/drive/MyDrive/datasets/building_model_latest.pth"
    LOCAL_PATH = "building_model_best.pth"
    
    current_model = DRIVE_PATH if os.path.exists(DRIVE_PATH) else LOCAL_PATH
    
    # Check indices that typically have dense buildings
    for idx in [10, 50, 100]:
        run_building_check(model_path=current_model, image_idx=idx)