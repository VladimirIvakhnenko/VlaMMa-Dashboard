import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

DIR_ORDINARY = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Рядовые руды")
DIR_HARD = Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Труднообогатимые руды")

def imread_unicode(p):
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def resize_to_limit(img: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def main():
    print("Шаг 1. Кэширование масок сульфидов...")
    
    files_ord = [DIR_ORDINARY / f for f in os.listdir(DIR_ORDINARY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files_hard = [DIR_HARD / f for f in os.listdir(DIR_HARD) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    dataset = []
    
    for p in tqdm(files_ord, desc="Рядовые руды"):
        img = imread_unicode(p)
        if img is None: continue
        img = resize_to_limit(img, 1024)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2]
        
        sulfide_mask = (brightness > 150).astype(np.uint8) * 255
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sulfide_mask = cv2.morphologyEx(sulfide_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        dataset.append({
            "mask": sulfide_mask,
            "gt_class": 0
        })
        
    for p in tqdm(files_hard, desc="Труднообогатимые"):
        img = imread_unicode(p)
        if img is None: continue
        img = resize_to_limit(img, 1024)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = hsv[:, :, 2]
        
        sulfide_mask = (brightness > 150).astype(np.uint8) * 255
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sulfide_mask = cv2.morphologyEx(sulfide_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        dataset.append({
            "mask": sulfide_mask,
            "gt_class": 1
        })
        
    print("\nШаг 2. Grid Search с учетом порога макро-кристаллов...")
    
    coefficients = [0.020, 0.022, 0.025, 0.028]
    thresholds = [35.0, 40.0, 45.0, 50.0]
    macro_thresholds = [5000, 10000, 15000, 20000, 30000, 999999]
    
    best_f1 = 0.0
    best_params = {}
    
    for coeff in coefficients:
        for thresh in thresholds:
            for macro_t in macro_thresholds:
                tp, fp, tn, fn = 0, 0, 0, 0
                
                for item in dataset:
                    mask = item["mask"]
                    gt = item["gt_class"]
                    h, w = mask.shape
                    
                    k_size = int(max(h, w) * coeff)
                    if k_size % 2 == 0: k_size += 1
                    k_size = max(5, k_size)
                    
                    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    ord_pixels = 0
                    fine_pixels = 0
                    max_grain_area = 0.0
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 100: continue
                        
                        max_grain_area = max(max_grain_area, area)
                        
                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        roi_eroded = eroded[y:y+h_box, x:x+w_box]
                        temp = np.zeros((h_box, w_box), dtype=np.uint8)
                        cv2.drawContours(temp, [cnt], -1, 255, -1, offset=(-x, -y))
                        
                        if np.any(np.logical_and(temp, roi_eroded)):
                            ord_pixels += area
                        else:
                            fine_pixels += area
                            
                    total = ord_pixels + fine_pixels
                    fine_prev = (fine_pixels / total * 100) if total > 0 else 0.0
                    
                    if max_grain_area > macro_t:
                        pred = 0
                    else:
                        pred = 1 if fine_prev > thresh else 0
                        
                    if gt == 1 and pred == 1: tp += 1
                    elif gt == 1 and pred == 0: fn += 1
                    elif gt == 0 and pred == 0: tn += 1
                    elif gt == 0 and pred == 1: fp += 1
                    
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = (tp + tn) / len(dataset)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        "coeff": coeff, "thresh": thresh, "macro_t": macro_t,
                        "f1": f1, "accuracy": accuracy,
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn
                    }
                    
    print("\n" + "="*80)
    print("ЛУЧШАЯ НАЙДЕННАЯ КОМБИНАЦИЯ:")
    print("="*80)
    print(f"Коэффициент эрозии: {best_params['coeff']:.3f}")
    print(f"Порог преобладания тонких: {best_params['thresh']:.1f}%")
    print(f"Порог макро-кристалла (на 1024px): {best_params['macro_t']} px")
    print(f"-> F1-Score: {best_params['f1']*100:.2f}% (Accuracy: {best_params['accuracy']*100:.2f}%)")
    print(f"-> Матрица ошибок: TP={best_params['tp']} / FP={best_params['fp']} / TN={best_params['tn']} / FN={best_params['fn']}")
    print("="*80)

if __name__ == "__main__":
    main()
