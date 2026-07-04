import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
import json

MASKS_DIR  = Path("masks")
MASKS_DIR.mkdir(exist_ok=True)

TASKS = [
    {
        "xml": Path("annotations.xml"),
        "img_dir": Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч1/Оталькованные руды")
    },
    {
        "xml": Path("annotations1.xml"),
        "img_dir": Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч2/оталькованные")
    },
    {
        "xml": Path("annotations2.xml"),
        "img_dir": Path("Задача 3. Скажи мне, кто твой шлиф/Фото руд по сортам. ч2/оталькованные")
    }
]

def decode_rle(rle_str: str, left: int, top: int, width: int, height: int,
               img_w: int, img_h: int) -> np.ndarray:
    counts = [int(x.strip()) for x in rle_str.split(",") if x.strip()]
    flat = []
    val = 0
    for count in counts:
        flat.extend([val] * count)
        val = 1 - val

    patch_size = width * height
    flat = flat[:patch_size]
    if len(flat) < patch_size:
        flat.extend([0] * (patch_size - len(flat)))

    arr = np.array(flat, dtype=np.uint8)
    patch = arr.reshape((height, width))

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1 = max(0, left)
    y1 = max(0, top)
    x2 = min(img_w, left + width)
    y2 = min(img_h, top + height)
    px1 = x1 - left
    py1 = y1 - top
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)
    full_mask[y1:y2, x1:x2] = patch[py1:py2, px1:px2]
    return full_mask

def parse_polygon(points_str: str, img_w: int, img_h: int) -> np.ndarray:
    pts = []
    for pair in points_str.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        x = max(0, min(img_w - 1, float(x_str)))
        y = max(0, min(img_h - 1, float(y_str)))
        pts.append([x, y])

    if len(pts) < 3:
        return np.zeros((img_h, img_w), dtype=np.uint8)

    pts_array = np.array(pts, dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_array], 1)
    return mask

def main():
    all_stats = []
    total_annotated = 0

    for task in TASKS:
        xml_path = task["xml"]
        img_dir = task["img_dir"]

        if not xml_path.exists():
            print(f"[!] Файл разметки не найден: {xml_path}")
            continue

        print(f"\nРазбор файла: {xml_path}...")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for image_el in root.findall("image"):
            img_name = image_el.get("name")
            img_w = int(image_el.get("width"))
            img_h = int(image_el.get("height"))
            img_id = image_el.get("id")

            combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            n_polygons = 0
            n_masks = 0

            for poly_el in image_el.findall("polygon"):
                if poly_el.get("label") == "talk":
                    points_str = poly_el.get("points", "")
                    poly_mask = parse_polygon(points_str, img_w, img_h)
                    combined_mask = np.maximum(combined_mask, poly_mask)
                    n_polygons += 1

            for mask_el in image_el.findall("mask"):
                if mask_el.get("label") == "talk":
                    rle_str = mask_el.get("rle", "")
                    left  = int(mask_el.get("left",  0))
                    top   = int(mask_el.get("top",   0))
                    width = int(mask_el.get("width",  img_w))
                    height = int(mask_el.get("height", img_h))

                    rle_mask = decode_rle(rle_str, left, top, width, height, img_w, img_h)
                    combined_mask = np.maximum(combined_mask, rle_mask)
                    n_masks += 1

            has_ann = n_polygons + n_masks
            mask_filename = Path(img_name).stem + ".png"
            mask_path = MASKS_DIR / mask_filename
            
            if has_ann > 0:
                save_mask = (combined_mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_path), save_mask)
                total_annotated += 1
                talc_pixels = int(combined_mask.sum())
                talc_pct = talc_pixels / (img_w * img_h) * 100
                print(f"  [id={img_id}] {img_name} -> Talk area: {talc_pct:.1f}% (polys={n_polygons}, rles={n_masks})")
            else:
                talc_pct = 0.0

            all_stats.append({
                "xml_source": str(xml_path),
                "id": img_id,
                "name": img_name,
                "size": f"{img_w}x{img_h}",
                "polygons": n_polygons,
                "rle_masks": n_masks,
                "talc_pct": round(talc_pct, 2),
                "mask_saved": str(mask_path)
            })

    with open("masks_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"Всего изображений обработано: {len(all_stats)}")
    print(f"Всего размеченных масок:       {total_annotated}")
    print(f"Маски сохранены в папку:       {MASKS_DIR.resolve()}")

if __name__ == "__main__":
    main()
