import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from dataset import DeepGlobeLandCoverDataset, train_transform
from models import get_landcover_model

# 1. Setup Device
device = torch.device("cpu")

# 2. Initialize Data
# Land Cover images are 2448x2448, much larger than the Road patches.
full_ds = DeepGlobeLandCoverDataset(image_dir='datasets/train', mask_dir='datasets/train', transform=train_transform)

# SMOKE TEST: Limit to 50 images for verification
subset_indices = list(range(50))
train_ds = Subset(full_ds, subset_indices)

# Batch size 2 is required for Batch Normalization layers
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# 3. Initialize Multi-Class Model (7 classes)
model = get_landcover_model().to(device)

# 4. Multi-Class Loss Logic
# We use 'ignore_index=6' to skip the 'Unknown' (Black/Cloud) class.
criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=6)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_landcover(epochs=1):
    print(f"Starting Land Cover SMOKE TEST on {device}...")
    
    for epoch in range(epochs):
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            # images: (B, 3, H, W), masks: (B, H, W)
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            if i % 5 == 0:
                print(f"Batch {i} | Multi-class Loss: {loss.item():.4f}")
                
            # Intermediate checkpoint
            if i % 10 == 0:
                torch.save(model.state_dict(), "landcover_best.pth")
                print(f"---> INTERMEDIATE SAVE: 'landcover_best.pth' updated.")

    print("Training finished! You can now run your visualization script.")

if __name__ == "__main__":
    train_landcover()