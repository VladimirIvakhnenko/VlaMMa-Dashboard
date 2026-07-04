"""
Шаг 1: Анализируем размеченные изображения.
Цель: подтвердить цвет аннотаций (синий) и собрать статистику.

По фото установлено:
  - Аннотации нарисованы СИНИМ цветом (Blue, H≈100-130 в OpenCV HSV)
  - Фон шлифа: оливково-зелёный (тальк+матрица) + жёлтые зоны (сульфиды)
  - Синие линии обводят зоны скопления талька
"""

import cv2
import numpy as np
from pathlib import Path
import json

ANNOTATED_DIR = r"C:\Users\arepe\OneDrive\Desktop\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования"
ORIGINAL_DIR  = r"C:\Users\arepe\OneDrive\Desktop\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды"

# ── Цвет аннотаций (синий) ────────────────────────────────────────────────────
# В OpenCV синий: H от 100 до 130 (это соответствует ~200-260° в реальном HSV)
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])

# ── Цвет талька (оливково-зелёный фон) ───────────────────────────────────────
# Тальк/матрица на шлифе: тёмно-оливковый, H≈25-55, умеренная насыщенность
TALC_LOWER = np.array([25, 30, 20])
TALC_UPPER = np.array([55, 180, 130])


def imread_unicode(path) -> np.ndarray | None:
    """Читает изображение с Unicode/кириллическим путём через numpy."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def analyze_single_image(ann_path):
    """
    Анализирует одно размеченное изображение:
    - Ищет синие аннотационные линии
    - Считает долю каждой фазы (сульфиды / тальк-матрица / аннотации)
    """
    img_bgr = imread_unicode(ann_path)
    if img_bgr is None:
        return None

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]

    # 1. Маска синих аннотаций
    blue_mask = cv2.inRange(img_hsv, BLUE_LOWER, BLUE_UPPER)
    n_blue = int((blue_mask > 0).sum())

    # 2. Маска оливково-зелёного (тальк + матрица) — исключая синие пиксели
    talc_mask_raw = cv2.inRange(img_hsv, TALC_LOWER, TALC_UPPER)
    talc_mask_raw = cv2.bitwise_and(talc_mask_raw, cv2.bitwise_not(blue_mask))
    n_talc = int((talc_mask_raw > 0).sum())

    # 3. Маска светлых зон (сульфиды) — яркие пиксели
    brightness = img_hsv[:, :, 2]
    sulfide_mask = (brightness > 150).astype(np.uint8) * 255
    sulfide_mask = cv2.bitwise_and(sulfide_mask, cv2.bitwise_not(blue_mask))
    n_sulfide = int((sulfide_mask > 0).sum())

    return {
        "file": ann_path.name,
        "shape": list(img_bgr.shape),
        "total_pixels": total_pixels,
        "blue_annotation_pixels": n_blue,
        "blue_fraction_pct": round(n_blue / total_pixels * 100, 3),
        "talc_matrix_pixels": n_talc,
        "talc_matrix_fraction_pct": round(n_talc / total_pixels * 100, 1),
        "sulfide_pixels": n_sulfide,
        "sulfide_fraction_pct": round(n_sulfide / total_pixels * 100, 1),
    }


def main():
    ann_dir = Path(ANNOTATED_DIR)
    ann_files = sorted(ann_dir.glob("*.JPG")) + sorted(ann_dir.glob("*.jpg"))

    print(f"Najdeno razmechennyh izobrazhenij: {len(ann_files)}\n")
    print(f"Iskhem annotacii: SINIJ (H={BLUE_LOWER[0]}-{BLUE_UPPER[0]} v OpenCV HSV)")
    print(f"Iskhem talk: OLIVKOVO-ZELENYJ (H={TALC_LOWER[0]}-{TALC_UPPER[0]})\n")
    print("=" * 60)

    all_results = []

    for f in ann_files[:10]:  # Первые 10 для скорости
        res = analyze_single_image(f)
        if res is None:
            continue
        all_results.append(res)

        print(f"[{res['file']}]")
        print(f"  Razmer: {res['shape'][1]}x{res['shape'][0]}")
        print(f"  Sinie annotacii: {res['blue_annotation_pixels']} px "
              f"({res['blue_fraction_pct']:.3f}%)")
        print(f"  Talk/matrica:    {res['talc_matrix_pixels']} px "
              f"({res['talc_matrix_fraction_pct']:.1f}%)")
        print(f"  Sul'fidy (svetl): {res['sulfide_pixels']} px "
              f"({res['sulfide_fraction_pct']:.1f}%)")
        print()

    # Итоговая статистика
    if all_results:
        avg_blue = sum(r["blue_fraction_pct"] for r in all_results) / len(all_results)
        avg_talc = sum(r["talc_matrix_fraction_pct"] for r in all_results) / len(all_results)
        avg_sulf = sum(r["sulfide_fraction_pct"] for r in all_results) / len(all_results)
        print("=" * 60)
        print("SREDNEE PO VSEM IZOBRAZHENIYAM:")
        print(f"  Sinie linii (annotacii): {avg_blue:.3f}%")
        print(f"  Talk/matrica:            {avg_talc:.1f}%")
        print(f"  Sul'fidy:                {avg_sulf:.1f}%")

    # Сохраняем результаты
    with open("annotation_analysis.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nRezul'taty sohraneny v annotation_analysis.json")


if __name__ == "__main__":
    main()
