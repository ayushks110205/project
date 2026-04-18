import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DeepGlobeLandCoverDataset, val_transform
from models import get_landcover_model

def run_landcover_evaluation(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_landcover_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = DeepGlobeLandCoverDataset(image_dir='datasets/train', mask_dir='datasets/train', transform=val_transform)
    loader = DataLoader(dataset, batch_size=8, num_workers=2)

    class_names = ['Urban', 'Agriculture', 'Range', 'Forest', 'Water', 'Barren', 'Unknown']
    num_classes = len(class_names)
    
    # Confusion Matrix-style accumulators
    total_intersection = np.zeros(num_classes)
    total_union = np.zeros(num_classes)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating Land Cover"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            for cls in range(num_classes):
                inter = np.logical_and(preds == cls, targets == cls).sum()
                union = np.logical_or(preds == cls, targets == cls).sum()
                total_intersection[cls] += inter
                total_union[cls] += union

    print("\n" + "="*40)
    print("📊 PER-CLASS IoU REPORT")
    print("="*40)
    ious = []
    for i, name in enumerate(class_names):
        # Skip 'Unknown' class in the final mean calculation
        iou = total_intersection[i] / (total_union[i] + 1e-6)
        if name != 'Unknown': ious.append(iou)
        print(f"{name.ljust(12)}: {iou:.4f}")
    
    print("-" * 40)
    print(f"⭐ Mean IoU:   {np.mean(ious):.4f}")
    print("="*40)

if __name__ == "__main__":
    run_landcover_evaluation("/content/drive/MyDrive/datasets/landcover_latest.pth")