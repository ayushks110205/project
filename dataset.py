import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Road Extraction Dataset ---
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask >= 128, 1.0, 0.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask

# --- Land Cover Dataset ---
class LandCoverDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')]
        self.color_map = {
            (0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
            (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5, (0, 0, 0): 6
        }

    def _rgb_to_mask(self, mask):
        id_mask = np.zeros(mask.shape[:2], dtype=np.int64)
        for rgb, idx in self.color_map.items():
            id_mask[np.all(mask == rgb, axis=-1)] = idx
        return id_mask

    def __len__(self): return len(self.images)

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

# Standard Transforms
train_transform = A.Compose([
    A.Resize(height=512, width=512), # For LandCover memory management
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])