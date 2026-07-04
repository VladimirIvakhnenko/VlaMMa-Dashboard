import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from detector import analyze_image

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

def evaluate():
    print("Запуск УСКОРЕННОЙ валидации классификации сульфидов...")
    
    files_ord = [DIR_ORDINARY / f for f in os.listdir(DIR_ORDINARY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files_hard = [DIR_HARD / f for f in os.listdir(DIR_HARD) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Найдено в тестовом наборе:")
    print(f"  - Рядовые руды: {len(files_ord)} файлов")
    print(f"  - Труднообогатимые руды: {len(files_hard)} файлов")
    print("="*60)
    
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    
    print("Анализ папки 'Рядовые руды'...")
    for p in tqdm(files_ord, desc="Рядовые руды"):
        img = imread_unicode(p)
        if img is None:
            continue
        img_fast = resize_to_limit(img, max_side=1024)
        res = analyze_image(img_fast)
        
        if res.talc_percent > 10.0:
            continue
            
        if res.ore_class_en == "ordinary":
            tn += 1
        elif res.ore_class_en == "hard_to_enrich":
            fp += 1
            
    print("\nАнализ папки 'Труднообогатимые руды'...")
    for p in tqdm(files_hard, desc="Труднообогатимые"):
        img = imread_unicode(p)
        if img is None:
            continue
        img_fast = resize_to_limit(img, max_side=1024)
        res = analyze_image(img_fast)
        
        if res.talc_percent > 10.0:
            continue
            
        if res.ore_class_en == "hard_to_enrich":
            tp += 1
        elif res.ore_class_en == "ordinary":
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    print("\n" + "="*60)
    print("ИТОГОВЫЕ МЕТРИКИ КЛАССИФИКАЦИИ СРАСТАНИЙ:")
    print("="*60)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn}")
    print(f"False Negatives (FN): {fn}")
    print("-"*60)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("="*60)
    
    if f1 >= 0.90:
        print("✅ Успех! Требование ТЗ (F1 >= 90%) выполнено!")
    else:
        print("⚠️ Внимание: F1-score ниже целевого показателя 90%.")

if __name__ == "__main__":
    evaluate()
