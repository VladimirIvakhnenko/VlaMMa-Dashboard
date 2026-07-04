import cv2
import numpy as np
import torch
import json
from pathlib import Path
import segmentation_models_pytorch as smp
from scipy.spatial.distance import directed_hausdorff

MODEL_PATH = "best_unet_model.pth"
TILE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBS_THRESHOLD = 0.35
MASKS_DIR = Path("masks")

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

def cv2_imread_unicode(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def predict_patch(model, patch_rgb) -> np.ndarray:
    x = patch_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x_orig = torch.tensor(x).unsqueeze(0).to(DEVICE)
    x_hflip = torch.flip(x_orig, dims=[3])
    with torch.no_grad():
        pred_orig = torch.sigmoid(model(x_orig))
        pred_hflip = torch.flip(torch.sigmoid(model(x_hflip)), dims=[3])
        pred = (pred_orig + pred_hflip) / 2.0
    return pred.squeeze().cpu().numpy()

def predict_tiled(model, img_rgb, step=128) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    pad_h = (TILE_SIZE - h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - w % TILE_SIZE) % TILE_SIZE
    padded_img = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    ph, pw = padded_img.shape[:2]
    
    prob_map = np.zeros((ph, pw), dtype=np.float32)
    weight_map = np.zeros((ph, pw), dtype=np.float32)
    
    y_coords, x_coords = np.indices((TILE_SIZE, TILE_SIZE))
    dist_to_edge = np.minimum(
        np.minimum(y_coords, TILE_SIZE - 1 - y_coords),
        np.minimum(x_coords, TILE_SIZE - 1 - x_coords)
    )
    tile_weight = dist_to_edge.astype(np.float32) / (TILE_SIZE / 2)
    tile_weight = np.clip(tile_weight, 0.05, 1.0)
    
    for y in range(0, ph - TILE_SIZE + 1, step):
        for x in range(0, pw - TILE_SIZE + 1, step):
            patch = padded_img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            pred_patch = predict_patch(model, patch)
            prob_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += pred_patch * tile_weight
            weight_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += tile_weight
            
    prob_map /= np.maximum(weight_map, 1e-5)
    return prob_map[:h, :w]

def compute_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float, float]:
    gt = (gt_mask > 0).astype(bool)
    pred = (pred_mask > 0).astype(bool)
    
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    
    iou = intersection / union if union > 0 else 1.0
    f1 = (2 * intersection) / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) > 0 else 1.0
    
    pts_gt = np.argwhere(gt)
    pts_pred = np.argwhere(pred)
    
    if len(pts_gt) == 0 and len(pts_pred) == 0:
        hausdorff = 0.0
    elif len(pts_gt) == 0 or len(pts_pred) == 0:
        hausdorff = float(np.sqrt(gt.shape[0]**2 + gt.shape[1]**2))
    else:
        d1 = directed_hausdorff(pts_gt, pts_pred)[0]
        d2 = directed_hausdorff(pts_pred, pts_gt)[0]
        hausdorff = max(d1, d2)
        
    return iou, f1, hausdorff

def main():
    print(f"Загрузка модели {MODEL_PATH} на {DEVICE}...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    if not Path(MODEL_PATH).exists():
        print(f"Ошибка: веса {MODEL_PATH} не найдены! Сначала запусти train.py.")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    results = []
    print("\n" + "="*70)
    print(f"{'Файл':<25} | {'IoU':<8} | {'F1-Score':<8} | {'Hausdorff (px)':<15}")
    print("="*70)
    
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
        if not img_path or not mask_path.exists():
            continue
            
        img = cv2_imread_unicode(str(img_path))
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        if max(h, w) > 1024:
            probs = predict_tiled(model, img_rgb, step=128)
        else:
            img_resized = cv2.resize(img_rgb, (TILE_SIZE, TILE_SIZE))
            probs_resized = predict_patch(model, img_resized)
            probs = cv2.resize(probs_resized, (w, h), interpolation=cv2.INTER_LINEAR)
            
        pred_mask = (probs > PROBS_THRESHOLD).astype(np.uint8) * 255
        
        iou, f1, hd = compute_metrics(gt_mask, pred_mask)
        results.append({"name": name, "iou": iou, "f1": f1, "hausdorff": hd})
        
        print(f"{name:<25} | {iou:.4f}   | {f1:.4f}     | {hd:.2f}")
        
    mean_iou = np.mean([r["iou"] for r in results])
    mean_f1 = np.mean([r["f1"] for r in results])
    mean_hd = np.mean([r["hausdorff"] for r in results])
    
    print("="*70)
    print(f"{'СРЕДНЕЕ ЗНАЧЕНИЕ':<25} | {mean_iou:.4f}   | {mean_f1:.4f}     | {mean_hd:.2f}")
    print("="*70)
    
    with open("evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "mean_iou": mean_iou,
            "mean_f1": mean_f1,
            "mean_hausdorff_pixels": mean_hd,
            "detailed": results
        }, f, ensure_ascii=False, indent=2)
    print("\nМетрики сохранены в: evaluation_metrics.json")

if __name__ == "__main__":
    main()
