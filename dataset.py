import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Road Extraction Dataset ---
class DeepGlobeRoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Only collect satellite images
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        
        # Load Image
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB)
        
        # Load Mask as Grayscale
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            # Fallback for inference if mask doesn't exist
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Binary threshold: Roads are white (255), background is black (0)
            mask = np.where(mask >= 128, 1.0, 0.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        return image, mask

# --- Land Cover Dataset ---
class DeepGlobeLandCoverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])
        
        # Mapping RGB colors to Class IDs
        self.color_map = {
            (0, 255, 255): 0,   # Urban
            (255, 255, 0): 1,   # Agriculture
            (255, 0, 255): 2,   # Range
            (0, 255, 0): 3,     # Forest
            (0, 0, 255): 4,     # Water
            (255, 255, 255): 5, # Barren
            (0, 0, 0): 6        # Unknown
        }

    def _rgb_to_mask(self, mask):
        id_mask = np.zeros(mask.shape[:2], dtype=np.int64)
        for rgb, idx in self.color_map.items():
            id_mask[np.all(mask == rgb, axis=-1)] = idx
        return id_mask

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(cv2.imread(os.path.join(self.mask_dir, mask_name)), cv2.COLOR_BGR2RGB)
        mask = self._rgb_to_mask(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        return image, mask

# Add this to your local dataset.py
class DeepGlobeBuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Filters for building-specific files if they have a unique prefix, 
        # otherwise uses the standard _sat.jpg pattern
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])

    def __len__(self): 
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Standardizing building masks to binary (0 or 1)
            mask = np.where(mask >= 128, 1.0, 0.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        return image, mask

# --- Transformation Pipelines ---

# 1. Training: Includes Augmentations (Flips)
train_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 2. Validation/Visualization: Clean (No Flips)
val_transform = A.Compose([
    A.Resize(height=512, width=512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])