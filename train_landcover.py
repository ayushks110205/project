import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
# Fixed: Syncing names with your dataset.py
from dataset import DeepGlobeLandCoverDataset, train_transform 
from models import get_landcover_model

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Pro-Tip: Speeds up training for constant input sizes (512x512)
torch.backends.cudnn.benchmark = True

# 2. Initialize Data
train_ds = DeepGlobeLandCoverDataset(
    image_dir='datasets/train', 
    mask_dir='datasets/train', 
    transform=train_transform
)

# Batch size 8 is a sweet spot for T4 VRAM. 
# If you get an 'Out of Memory' error, change this to 4.
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)

# 3. Initialize Multi-Class Model (7 classes)
model = get_landcover_model().to(device)

# 4. Multi-Class Focal Loss
# ignore_index=6 skips 'Unknown/Clouds' to keep the gradients clean
criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=6)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_landcover(epochs=25):
    print(f"🚀 Starting Multiclass Training on: {device}")
    print(f"📊 Dataset Size: {len(train_ds)} images")
    
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        # Calculate average epoch loss for checkpointing
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

        # Permanent Save to Google Drive
        save_path = "/content/drive/MyDrive/datasets/landcover_latest.pth"
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Always keep a local copy and a cloud copy
            torch.save(model.state_dict(), "landcover_best.pth") 
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New Best Landcover Model Saved to Drive!")

    print("🏁 Training finished! You are ready for the Multiclass Visualization.")

if __name__ == "__main__":
    train_landcover(epochs=25)