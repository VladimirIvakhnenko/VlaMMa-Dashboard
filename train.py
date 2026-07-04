import os
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

IMAGES_DIR = Path("augmented/images")
MASKS_DIR  = Path("masks")
MODEL_SAVE_PATH = "best_unet_model.pth"

BATCH_SIZE = 8
EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class TalcDataset(Dataset):
    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        image = cv2_imread_unicode(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2_imread_unicode(str(mask_path), grayscale=True)
        
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)
        
        return torch.tensor(image), torch.tensor(mask)

def cv2_imread_unicode(path: str, grayscale=False) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(buf, mode)

SRC_DIR_1 = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Оталькованные руды")
SRC_DIR_2 = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч2/оталькованные")
SRC_DIR_3 = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч2/оталькованные")

ANNOTATED = [
    {"name": "2550378-1 5x", "dir": SRC_DIR_1},
    {"name": "2550381-1 10x", "dir": SRC_DIR_1},
    {"name": "2550381-2 10x", "dir": SRC_DIR_1},
    {"name": "2550382-1 10x", "dir": SRC_DIR_1},
    {"name": "150_", "dir": SRC_DIR_2},
    {"name": "1822101 1", "dir": SRC_DIR_2},
    {"name": "1822215 3 ", "dir": SRC_DIR_2},
    {"name": "1907296", "dir": SRC_DIR_2},
    {"name": "41", "dir": SRC_DIR_2},
    {"name": "48", "dir": SRC_DIR_2},
    {"name": "-42", "dir": SRC_DIR_3},
    {"name": "DSCN4273", "dir": SRC_DIR_3},
    {"name": "DSCN4290", "dir": SRC_DIR_3},
    {"name": "DSCN4719", "dir": SRC_DIR_3},
]

train_pairs = []

for item in ANNOTATED:
    name = item["name"]
    img_dir = item["dir"]
    img_path = None
    for ext in [".JPG", ".jpg", ".png", ".PNG"]:
        p = img_dir / (name + ext)
        if p.exists():
            img_path = p
            break
    mask_path = MASKS_DIR / (name + ".png")
    if img_path and mask_path.exists():
        train_pairs.append((img_path, mask_path))

aug_images = list(IMAGES_DIR.glob("*.jpg"))
for img_path in aug_images:
    name = img_path.stem
    mask_path = Path("augmented/masks") / f"{name}.png"
    if mask_path.exists():
        train_pairs.append((img_path, mask_path))

print(f"Total training pairs: {len(train_pairs)}")
print(f"  - Original: {len(ANNOTATED)}")
print(f"  - Augmented: {len(aug_images)}")

train_dataset = TalcDataset(train_pairs)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
model.to(DEVICE)

criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

train_losses = []
best_loss = float("inf")

print(f"\nTraining on {DEVICE} for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_train_loss = 0.0
    
    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion_bce(outputs, masks) + criterion_dice(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * images.size(0)
        
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_train_loss:.4f} | LR: {current_lr:.6f}")
    
    if epoch_train_loss < best_loss:
        best_loss = epoch_train_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> Saved best weights (Loss: {best_loss:.4f})")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE + Dice)")
plt.legend()
plt.title("U-Net Training Curve (Combined Dataset + Regularization)")
plt.grid(True)
plt.savefig("learning_curve.png")
print("\nLearning curve graph saved in learning_curve.png")
print(f"Model saved to {MODEL_SAVE_PATH}")
