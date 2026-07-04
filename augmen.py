import cv2
import numpy as np
import albumentations as A
from pathlib import Path

SRC_DIR_1 = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Оталькованные руды")
SRC_DIR_2 = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч2/оталькованные")
SRC_DIR_3 = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч2/оталькованные")
SRC_MASKS = Path("masks")

OUT_IMAGES = Path("augmented/images")
OUT_MASKS  = Path("augmented/masks")
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_MASKS.mkdir(parents=True, exist_ok=True)

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

N_AUG = 30

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.75),
    A.ShiftScaleRotate(
        shift_limit=0.08,
        scale_limit=0.15,
        rotate_limit=45,
        border_mode=cv2.BORDER_REFLECT,
        p=0.6
    ),
    A.ElasticTransform(
        alpha=30, sigma=5,
        p=0.2
    ),
    A.Resize(height=512, width=512),
    A.RandomBrightnessContrast(brightness_limit=(0.0, 0.25), contrast_limit=0.10, p=0.6),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=(0, 20), p=0.6),
])

def cv2_imread_unicode(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def main():
    for f in OUT_IMAGES.glob("*_aug*"): f.unlink()
    for f in OUT_MASKS.glob("*_aug*"): f.unlink()

    total = 0
    for item in ANNOTATED:
        name = item["name"]
        img_dir = item["dir"]

        img_path = None
        for ext in [".JPG", ".jpg", ".png", ".PNG"]:
            p = img_dir / (name + ext)
            if p.exists():
                img_path = p
                break

        mask_path = SRC_MASKS / (name + ".png")

        if img_path is None or not mask_path.exists():
            print(f"[!] SKIP: {name}")
            continue

        image = cv2_imread_unicode(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        print(f"Augmenting: {name} ({image.shape[1]}x{image.shape[0]})")

        for i in range(N_AUG):
            result = transform(image=image, mask=mask)
            aug_img  = result["image"]
            aug_mask = result["mask"]

            aug_mask = (aug_mask > 127).astype(np.uint8) * 255

            out_name = f"{name}_aug{i:03d}"
            cv2.imwrite(str(OUT_IMAGES / (out_name + ".jpg")), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(OUT_MASKS / (out_name + ".png")), aug_mask)
            total += 1

        print(f"  -> {N_AUG} par sohraneno")

    print(f"\nItogo: {total} par v {OUT_IMAGES.parent}/")

if __name__ == "__main__":
    main()
