import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DeepGlobeRoadDataset, val_transform
from models import get_road_model
import os

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8)
    target = target.astype(np.uint8)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice / F1 Score
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    return iou, dice

def run_global_evaluation(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Starting Global Evaluation on: {device}")

    # 1. Load Model
    model = get_road_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Dataset
    dataset = DeepGlobeRoadDataset(
        image_dir='datasets/train', # We use the full set to get a global view
        mask_dir='datasets/train', 
        transform=val_transform
    )
    # Using a larger batch size for speed during evaluation
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    all_ious = []
    all_dices = []

    # 3. Evaluation Loop
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            targets = masks.cpu().numpy()

            for i in range(len(preds)):
                iou, dice = calculate_metrics(preds[i], targets[i])
                all_ious.append(iou)
                all_dices.append(dice)

    # 4. Final Report
    print("\n" + "="*30)
    print("      FINAL ROAD REPORT")
    print("="*30)
    print(f"✅ Total Images Evaluated: {len(all_ious)}")
    print(f"⭐ Mean IoU (mIoU):       {np.mean(all_ious):.4f}")
    print(f"⭐ Mean F1 (Dice):        {np.mean(all_dices):.4f}")
    print("="*30)

if __name__ == "__main__":
    DRIVE_PATH = "/content/drive/MyDrive/datasets/road_model_latest.pth"
    if os.path.exists(DRIVE_PATH):
        run_global_evaluation(DRIVE_PATH)
    else:
        print("❌ Model not found in Drive. Check path!")