import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import RoadDataset, train_transform 
from models import get_road_model

# 1. Setup Device - T4 GPU Optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize Full Data (Removing Smoke Test limits)
train_ds = RoadDataset(
    image_dir='datasets/train', 
    mask_dir='datasets/train', 
    transform=train_transform
)

# Batch size 16 or 32 is excellent for binary segmentation on a T4
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

# 3. Initialize Binary Model
model = get_road_model().to(device)

# 4. Binary Focal Loss 
# Optimized for high class imbalance (few road pixels vs lots of background)
criterion = smp.losses.FocalLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_road(epochs=20):
    print(f"🚀 Starting Road Extraction Training on: {device}")
    print(f"📊 Training on {len(train_ds)} high-res images")
    
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, masks) in enumerate(train_loader):
            # images: (B, 3, H, W), masks: (B, H, W) -> (B, 1, H, W)
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1) 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

        # Permanent Save to Google Drive
        save_path = "/content/drive/MyDrive/datasets/road_model_latest.pth"
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth") # Local checkpoint
            torch.save(model.state_dict(), save_path)        # Secure Drive storage
            print(f"🌟 New Best Road Model Saved to Drive!")

    print("🏁 Road training finished! Ready for inference.")

if __name__ == "__main__":
    train_road(epochs=20)