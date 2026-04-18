import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import DeepGlobeBuildingDataset, train_transform
from models import get_building_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_building(epochs=20):
    print(f"🏘️ Starting Building Detection Training on: {device}")
    
    dataset = DeepGlobeBuildingDataset(image_dir='datasets/train', mask_dir='datasets/train', transform=train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    model = get_building_model().to(device)
    
    # Using a combination of Dice and BCE loss often helps with 
    # geometric shapes like buildings
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for i, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}] Loss: {loss.item():.4f}")
        
        # Save to Drive path (ensure this exists in Colab later)
        torch.save(model.state_dict(), f"building_model_latest.pth")

if __name__ == "__main__":
    train_building()