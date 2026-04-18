import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import cv2
import numpy as np
from models import get_road_model, get_landcover_model
from dataset import train_transform # We use the same Resize/Normalize logic

# Color mapping for Land Cover (copied from your visualize script)
LANDCOVER_COLORS = {
    0: [0, 255, 255], 1: [255, 255, 0], 2: [255, 0, 255],
    3: [0, 255, 0],   4: [0, 0, 255],   5: [255, 255, 255], 6: [0, 0, 0]
}

def run_inference(image_path, model_type="road"):
    device = torch.device("cpu")
    
    # 1. Load the correct model and weights
    if model_type == "road":
        model = get_road_model().to(device)
        weights_path = "best_model.pth"
    else:
        model = get_landcover_model().to(device)
        weights_path = "landcover_best.pth"
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Preprocess the input image
    original_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Use the pipeline's transform (Resize to 512, Normalize, ToTensor)
    input_tensor = train_transform(image=rgb_img)["image"].unsqueeze(0).to(device)

    # 3. Perform Inference
    with torch.no_grad():
        output = model(input_tensor)
        
        if model_type == "road":
            # Binary prediction: Sigmoid + Threshold
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (prob > 0.5).astype(np.uint8) * 255
        else:
            # Multi-class prediction: Argmax + Color Mapping
            pred_ids = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            h, w = pred_ids.shape
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for idx, color in LANDCOVER_COLORS.items():
                mask[pred_ids == idx] = color
            # Convert RGB to BGR for OpenCV saving
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    # 4. Save the Result
    if not os.path.exists("results"):
        os.makedirs("results")
    
    filename = os.path.basename(image_path).replace(".jpg", "_predicted.png")
    save_path = os.path.join("results", filename)
    cv2.imwrite(save_path, mask)
    print(f"✅ Prediction saved to: {save_path}")

if __name__ == "__main__":
    # Change these to test any image in your dataset!
    test_image = "datasets/test/206_sat.jpg" # Example path
    
    if os.path.exists(test_image):
        run_inference(test_image, model_type="road")
    else:
        print("Point 'test_image' to a real .jpg file to run inference!")