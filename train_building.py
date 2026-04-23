import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np

from dataset import DeepGlobeBuildingDataset, train_transform
from models import get_building_model

# =============================================================================
# Config  — Kaggle paths
# =============================================================================
BASE_PATH  = '/kaggle/input/datasets/ayushks07/deep-globe-extraction-dataset'
IMAGE_DIR  = f'{BASE_PATH}/train'
MASK_DIR   = f'{BASE_PATH}/train'

LOCAL_BEST = '/kaggle/working/building_model_best.pth'
CKPT_DIR   = '/kaggle/working/building_ckpts'
os.makedirs(CKPT_DIR, exist_ok=True)

BATCH_SIZE          = 16    # T4 ×2 — safe with AMP at 512×512
NUM_EPOCHS          = 20
LR                  = 1e-4
EARLY_STOP_PATIENCE = 5
CHECKPOINT_EVERY    = 5

# =============================================================================
# GPU / Multi-GPU Setup
# =============================================================================
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus        = torch.cuda.device_count()
USE_MULTI_GPU = n_gpus > 1

if torch.cuda.is_available():
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"🖥️  GPU {i}: {props.name}  VRAM={props.total_memory/1e9:.1f} GB")
    print(f"   DataParallel: {'YES (×' + str(n_gpus) + ')' if USE_MULTI_GPU else 'NO'}")
else:
    print("⚠️  No GPU detected — running on CPU")

# =============================================================================
# Training Function
# =============================================================================
def train_building(epochs: int = NUM_EPOCHS):
    print(f"🏘️  Starting Building Detection Training on: {device}")

    dataset = DeepGlobeBuildingDataset(
        image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=train_transform
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    model = get_building_model().to(device)
    if USE_MULTI_GPU:
        model = nn.DataParallel(model)
        print(f"   Model wrapped in DataParallel across {n_gpus} GPUs")

    # Dice + BCE blend — works well for compact geometric shapes like buildings
    criterion = smp.losses.DiceLoss(mode='binary')
    base_model = model.module if USE_MULTI_GPU else model
    optimizer  = optim.Adam(base_model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler     = GradScaler()

    best_loss         = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        bar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                   unit='batch', leave=False)
        for images, masks in bar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(images)
                loss    = criterion(outputs, masks.unsqueeze(1).float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss   = epoch_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # ── Best checkpoint ───────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss         = avg_loss
            epochs_no_improve = 0
            state = (model.module if USE_MULTI_GPU else model).state_dict()
            torch.save(state, LOCAL_BEST)
            print(f"   🌟 New best loss={best_loss:.4f} → saved to {LOCAL_BEST}")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'building_ckpt_ep{epoch:02d}.pth')
            state = (model.module if USE_MULTI_GPU else model).state_dict()
            torch.save({'epoch': epoch, 'model_state': state,
                        'best_loss': best_loss}, ckpt_path)
            print(f"   💾 Checkpoint → {ckpt_path}")

        torch.cuda.empty_cache()

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break

    print(f"\n✅ Training complete. Best loss={best_loss:.4f}")
    print(f"   Model saved to: {LOCAL_BEST}")


if __name__ == "__main__":
    train_building()