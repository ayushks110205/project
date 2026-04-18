import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DeepGlobeBuildingDataset, val_transform
from models import get_building_model

def run_building_evaluation(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_building_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = DeepGlobeBuildingDataset(image_dir='datasets/train', mask_dir='datasets/train', transform=val_transform)
    loader = DataLoader(dataset, batch_size=8, num_workers=2)

    all_ious = []
    all_dices = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating Buildings"):
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).cpu().numpy()
            targets = masks.cpu().numpy()

            for p, t in zip(preds, targets):
                inter = np.logical_and(p, t).sum()
                union = np.logical_or(p, t).sum()
                
                iou = (inter + 1e-6) / (union + 1e-6)
                dice = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
                
                all_ious.append(iou)
                all_dices.append(dice)

    print("\n" + "🏢 BUILDING DETECTION REPORT")
    print(f"Mean IoU:  {np.mean(all_ious):.4f}")
    print(f"Mean Dice: {np.mean(all_dices):.4f} (F1-Score)")

if __name__ == "__main__":
    run_building_evaluation("/content/drive/MyDrive/datasets/building_model_latest.pth")