"""
predict.py — Профессиональный инференс модели U-Net с поддержкой тайлинга (скользящего окна).

Позволяет обрабатывать изображения высокого разрешения и панорамы без потери качества
путем нарезки на перекрывающиеся фрагменты 512x512.

Запуск:
    venv\\Scripts\\python predict.py "путь_к_изображению.jpg" [выходной_файл.jpg] [--threshold 0.35] [--tile]
"""

import sys
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

# Константы
MODEL_PATH = "best_unet_model.pth"
TILE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cv2_imread_unicode(path: str) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def predict_patch(model, patch_rgb):
    """
    Суперточный инференс для одного патча 512x512 с 8-кратным TTA (Flips + Rotations).
    Модель делает 8 предсказаний для разных ракурсов одного патча и усредняет их.
    """
    x = patch_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # (C, H, W)
    x_tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)
    
    preds = []
    with torch.no_grad():
        # Проходимся по 4 поворотам (0, 90, 180, 270)
        for rot in [0, 1, 2, 3]:
            # Поворот тензора
            x_rot = torch.rot90(x_tensor, rot, [2, 3])
            
            # Предсказание для повернутого
            pred_rot = torch.sigmoid(model(x_rot))
            # Возвращаем предсказание обратно
            pred_unrot = torch.rot90(pred_rot, -rot, [2, 3])
            preds.append(pred_unrot)
            
            # То же самое, но с горизонтальным флипом
            x_rot_flip = torch.flip(x_rot, dims=[3])
            pred_rot_flip = torch.sigmoid(model(x_rot_flip))
            pred_unrot_flip = torch.rot90(torch.flip(pred_rot_flip, dims=[3]), -rot, [2, 3])
            preds.append(pred_unrot_flip)
            
        # Усредняем все 8 предсказаний
        final_pred = torch.stack(preds).mean(dim=0)
        
    return final_pred.squeeze().cpu().numpy()

def predict_tiled(model, img_rgb, step=96):
    """Инференс скользящим окном с шагом step=96 (перекрытие 80%)."""
    h, w = img_rgb.shape[:2]
    
    pad_h = (TILE_SIZE - h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - w % TILE_SIZE) % TILE_SIZE
    
    padded_img = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    ph, pw = padded_img.shape[:2]
    
    # Матрицы для накопления вероятностей и весов перекрытий
    prob_map = np.zeros((ph, pw), dtype=np.float32)
    weight_map = np.zeros((ph, pw), dtype=np.float32)
    
    # Линейный спад весов к краям тайла, чтобы сгладить швы
    y_coords, x_coords = np.indices((TILE_SIZE, TILE_SIZE))
    dist_to_edge = np.minimum(
        np.minimum(y_coords, TILE_SIZE - 1 - y_coords),
        np.minimum(x_coords, TILE_SIZE - 1 - x_coords)
    )
    tile_weight = dist_to_edge.astype(np.float32) / (TILE_SIZE / 2)
    tile_weight = np.clip(tile_weight, 0.05, 1.0)
    
    # Скользим окном с шагом 128 (перекрытие 75%) для максимальной точности
    for y in range(0, ph - TILE_SIZE + 1, step):
        for x in range(0, pw - TILE_SIZE + 1, step):
            patch = padded_img[y:y+TILE_SIZE, x:x+TILE_SIZE]
            pred_patch = predict_patch(model, patch)
            
            prob_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += pred_patch * tile_weight
            weight_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += tile_weight
            
    # Усредняем по весам
    prob_map /= np.maximum(weight_map, 1e-5)
    
    # Обрезаем обратно под оригинальный размер
    return prob_map[:h, :w]

def main():
    # Парсинг аргументов командной строки
    args = sys.argv[1:]
    if len(args) < 1:
        print("Ошибка: не указан путь к файлу изображения!")
        print('Использование: venv\\Scripts\\python predict.py "путь.jpg" ["выход.jpg"] [--threshold 0.35] [--tile]')
        sys.exit(1)

    img_path = Path(args[0])
    if not img_path.exists():
        print(f"Ошибка: файл {img_path} не найден!")
        sys.exit(1)

    # Ищем порог уверенности
    threshold = 0.35  # По умолчанию снизили с 0.50 до 0.35
    if "--threshold" in args:
        idx = args.index("--threshold")
        if idx + 1 < len(args):
            threshold = float(args[idx+1])

    # Использовать ли скользящее окно (тайлинг)
    use_tiling = "--tile" in args

    # Имя выходного файла
    out_path = Path("prediction_result.jpg")
    if len(args) > 1 and not args[1].startswith("-"):
        out_path = Path(args[1])

    # ── Загрузка ──────────────────────────────────────────────────────────────
    orig_img = cv2_imread_unicode(str(img_path))
    h, w = orig_img.shape[:2]
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # ── Загрузка модели U-Net ──────────────────────────────────────────────────
    print(f"Загрузка модели на {DEVICE}...")
    model = smp.Unet(
        encoder_name="resnet34",  # Перешли на resnet34
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ── Инференс ──────────────────────────────────────────────────────────────
    if use_tiling:
        print("Запуск сегментации скользящим окном (Tiling, step=96)...")
        probs = predict_tiled(model, img_rgb, step=96)  # Шаг 96 для 80% перекрытия
    else:
        print("Запуск сегментации на сжатом изображении (Resize)...")
        img_resized = cv2.resize(orig_img, (TILE_SIZE, TILE_SIZE))
        img_rgb_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        probs_resized = predict_patch(model, img_rgb_resized)
        # Возвращаем маску вероятностей к оригинальному разрешению
        probs = cv2.resize(probs_resized, (w, h), interpolation=cv2.INTER_LINEAR)

    # Применяем порог уверенности
    pred_mask = (probs > threshold).astype(np.uint8) * 255

    # ── Создание оверлея ──────────────────────────────────────────────────────
    overlay = orig_img.copy()
    overlay[pred_mask > 0] = [0, 0, 220]  # красный цвет
    visualized = cv2.addWeighted(orig_img, 0.6, overlay, 0.4, 0)

    # Склеиваем 3 изображения для визуализации (масштабируем до 700px по высоте)
    scale = 700 / h
    new_h, new_w = int(h * scale), int(w * scale)
    
    img_show = cv2.resize(orig_img, (new_w, new_h))
    mask_show = cv2.cvtColor(cv2.resize(pred_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    vis_show = cv2.resize(visualized, (new_w, new_h))

    # Рисуем подписи
    cv2.putText(img_show, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(mask_show, f"Mask (thresh={threshold})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis_show, "Overlay Visualization", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Конкатенируем по горизонтали
    triptych = np.hstack([img_show, mask_show, vis_show])

    # Сохраняем результат
    cv2.imwrite(str(out_path), triptych)
    
    # Подсчитаем процент предсказанного талька
    talc_pixels = (pred_mask > 0).sum()
    total_pixels = w * h
    pct = (talc_pixels / total_pixels) * 100
    
    print(f"Результат сохранен в: {out_path.resolve()}")
    print(f"Доля талька на снимке: {pct:.2f}%")

if __name__ == "__main__":
    main()
