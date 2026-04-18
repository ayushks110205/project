import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import DeepGlobeLandCoverDataset, train_transform
from models import get_landcover_model

device = torch.device("cpu") # Staying on CPU for stability

# 1. Initialize Data
train_ds = DeepGlobeLandCoverDataset(image_dir='datasets/train', mask_dir='datasets/train', transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# 2. Initialize Multi-Class Model
model = get_landcover_model().to(device)

# 3. Multi-Class Focal Loss
# We set ignore_index=6 to skip the 'Unknown/Cloud' class during training
criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=6)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_landcover():
    model.train()
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device) # Masks are now (B, H, W) integers
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            print(f"Batch {i} | Multi-class Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), "landcover_best.pth")

if __name__ == "__main__":
    train_landcover()