import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import RoadDataset, train_transform
from models import get_road_model

def evaluate():
    device = torch.device("cpu")
    model = get_road_model().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Load the VALIDATION folder
    val_ds = RoadDataset(image_dir='datasets/valid', mask_dir='datasets/valid', transform=train_transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    total_iou = 0
    with torch.no_grad():
        for images, masks in val_loader:
            outputs = model(images.to(device))
            tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks.to(device).long().unsqueeze(1), mode='binary', threshold=0.5)
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            total_iou += iou.item()

    print(f"Final Validation mIoU: {total_iou / len(val_loader):.4f}")

if __name__ == "__main__":
    evaluate()