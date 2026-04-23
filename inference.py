import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import cv2
import numpy as np
from models import get_road_model, get_landcover_model
from dataset import val_transform   # val_transform: clean resize+normalize, no augmentation

# Color mapping for Land Cover visualisation
LANDCOVER_COLORS = {
    0: [0, 255, 255],   # Urban land
    1: [255, 255, 0],   # Agriculture
    2: [255, 0, 255],   # Rangeland
    3: [0, 255, 0],     # Forest
    4: [0, 0, 255],     # Water
    5: [255, 255, 255], # Barren land
    6: [0, 0, 0],       # Unknown
}

# ── Kaggle model paths ────────────────────────────────────────────────────────
ROAD_WEIGHTS      = '/kaggle/working/road_model_best.pth'
LANDCOVER_WEIGHTS = '/kaggle/working/landcover_model_latest.pth'
RESULTS_DIR       = '/kaggle/working/results'


def run_inference(image_path: str, model_type: str = "road"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load the correct model and weights
    if model_type == "road":
        model        = get_road_model().to(device)
        weights_path = ROAD_WEIGHTS
    else:
        model        = get_landcover_model().to(device)
        weights_path = LANDCOVER_WEIGHTS

    if not os.path.exists(weights_path):
        print(f"❌ Weights not found at: {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"✅ Loaded {model_type} model from: {weights_path}")

    # 2. Preprocess the input image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"❌ Image not found: {image_path}")
        return
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Use val_transform (Resize 512, Normalize, ToTensor) — no augmentation
    input_tensor = val_transform(image=rgb_img)["image"].unsqueeze(0).to(device)

    # 3. Perform Inference
    with torch.no_grad():
        output = model(input_tensor)

        if model_type == "road":
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (prob > 0.5).astype(np.uint8) * 255
        else:
            pred_ids = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            h, w = pred_ids.shape
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in LANDCOVER_COLORS.items():
                mask[pred_ids == idx] = color
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    # 4. Save the Result
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename  = os.path.basename(image_path).replace(".jpg", "_predicted.png")
    save_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(save_path, mask)
    print(f"✅ Prediction saved to: {save_path}")


if __name__ == "__main__":
    # Update this path to any satellite image on Kaggle
    test_image = '/kaggle/input/datasets/ayushks07/deep-globe-extraction-dataset/train/206_sat.jpg'

    if os.path.exists(test_image):
        run_inference(test_image, model_type="road")
    else:
        print("⚠️  Update 'test_image' to a valid satellite image path.")