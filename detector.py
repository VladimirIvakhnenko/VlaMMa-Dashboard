import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
import logging
import os
import time
import atexit

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))

logging.basicConfig(
    filename="analysis_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

MODEL_PATH = "best_unet_model.pth"
TILE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBS_THRESHOLD = 0.35

if DEVICE.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    device_msg = f"Device: {DEVICE} | GPU: {gpu_name} | VRAM: {gpu_mem:.1f} GB"
else:
    device_msg = "Device: CPU (GPU not available)"

logging.info(device_msg)
print(device_msg)

def _cleanup_gpu():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

atexit.register(_cleanup_gpu)

_MODEL_INSTANCE = None

@dataclass
class DetectionResult:
    talc_mask: np.ndarray
    sulfide_ordinary_mask: np.ndarray
    sulfide_fine_mask: np.ndarray
    annotation_mask: np.ndarray
    talc_fraction: float
    talc_percent: float
    sulfide_ordinary_percent: float
    sulfide_fine_percent: float
    fine_prevalence_percent: float
    ore_class: str
    ore_class_en: str
    conclusion: str
    overlay: np.ndarray
    talc_area_px: int
    ordinary_area_px: int
    fine_area_px: int

def get_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        weights = Path(MODEL_PATH)
        if weights.exists():
            model.load_state_dict(torch.load(str(weights), map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _MODEL_INSTANCE = model
    return _MODEL_INSTANCE

BATCH_SIZE = 8  # количество тайлов в батче для GPU

def predict_patch(model, patch_rgb) -> np.ndarray:
    x = patch_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))

    x_tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)
    del x

    with torch.no_grad():
        pred = torch.sigmoid(model(x_tensor))
        del x_tensor

    result = pred.squeeze().cpu().numpy()
    del pred
    return result

def predict_tiled(model, img_rgb, step=256) -> np.ndarray:
    h, w = img_rgb.shape[:2]

    pad_h = (TILE_SIZE - h % TILE_SIZE) % TILE_SIZE
    pad_w = (TILE_SIZE - w % TILE_SIZE) % TILE_SIZE
    ph, pw = h + pad_h, w + pad_w

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
            y_end = min(y + TILE_SIZE, h)
            x_end = min(x + TILE_SIZE, w)
            patch_h = y_end - y
            patch_w = x_end - x

            tile = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
            tile[:patch_h, :patch_w] = img_rgb[y:y_end, x:x_end]

            if patch_h < TILE_SIZE or patch_w < TILE_SIZE:
                tile = cv2.copyMakeBorder(
                    tile[:patch_h, :patch_w],
                    0, TILE_SIZE - patch_h,
                    0, TILE_SIZE - patch_w,
                    cv2.BORDER_REFLECT
                )

            pred_patch = predict_patch(model, tile)

            prob_map[y:y_end, x:x_end] += pred_patch[:patch_h, :patch_w] * tile_weight[:patch_h, :patch_w]
            weight_map[y:y_end, x:x_end] += tile_weight[:patch_h, :patch_w]

    prob_map /= np.maximum(weight_map, 1e-5)
    del weight_map
    return prob_map[:h, :w]

def detect_and_classify_sulfides(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(img_hsv)
    del img_hsv

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_norm = clahe.apply(v_chan)
    del v_chan, h_chan, s_chan

    sulfide_mask = (v_norm > 165).astype(np.uint8) * 255
    del v_norm
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sulfide_mask = cv2.morphologyEx(sulfide_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    h, w = sulfide_mask.shape
    ordinary_mask = np.zeros((h, w), dtype=np.uint8)
    fine_mask = np.zeros((h, w), dtype=np.uint8)

    k_size = int(max(h, w) * 0.025)
    if k_size % 2 == 0:
        k_size += 1
    k_size = max(5, k_size)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    eroded = cv2.erode(sulfide_mask, kernel_erode)

    contours, _ = cv2.findContours(sulfide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    del sulfide_mask

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        contour_mask[:] = 0
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)

        if np.any(np.logical_and(contour_mask, eroded)):
            ordinary_mask[contour_mask > 0] = 255
        else:
            fine_mask[contour_mask > 0] = 255

    return ordinary_mask, fine_mask

def extract_annotation_mask(img_bgr: np.ndarray) -> np.ndarray:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([100, 100, 50])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    del img_hsv
    return mask

def create_overlay(img_bgr: np.ndarray,
                   talc_mask: np.ndarray,
                   ordinary_mask: np.ndarray,
                   fine_mask: np.ndarray,
                   ann_mask: np.ndarray,
                   alpha: float = 0.45) -> np.ndarray:
    overlay = img_bgr.copy()

    overlay[talc_mask > 0] = [255, 0, 0]
    overlay[ordinary_mask > 0] = [0, 200, 0]
    overlay[fine_mask > 0] = [0, 0, 220]

    cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0, dst=overlay)

    contours_talc, _ = cv2.findContours(talc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_talc, -1, (255, 0, 0), 2)

    return overlay

def analyze_image(img_bgr: np.ndarray, filename: str = "unknown") -> DetectionResult:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    model = get_model()

    t_start = time.time()
    if max(h, w) > 1024:
        probs = predict_tiled(model, img_rgb, step=256)
    else:
        img_resized = cv2.resize(img_rgb, (TILE_SIZE, TILE_SIZE))
        probs_resized = predict_patch(model, img_resized)
        probs = cv2.resize(probs_resized, (w, h), interpolation=cv2.INTER_LINEAR)

    t_seg = time.time() - t_start
    logging.info(f"Segmentation time: {t_seg:.2f}s | Tiles: {((w + 255) // 256) * ((h + 255) // 256)} | Device: {DEVICE}")

    del img_rgb
    t_post = time.time()
    talc_mask = (probs > PROBS_THRESHOLD).astype(np.uint8) * 255
    del probs
    ordinary_mask, fine_mask = detect_and_classify_sulfides(img_bgr)
    ann_mask = extract_annotation_mask(img_bgr)
    logging.info(f"Post-processing time: {time.time() - t_post:.2f}s")

    total_pixels = h * w
    talc_pixels = int((talc_mask > 0).sum())
    ordinary_pixels = int((ordinary_mask > 0).sum())
    fine_pixels = int((fine_mask > 0).sum())

    talc_fraction = float(talc_pixels) / total_pixels
    talc_percent = talc_fraction * 100

    ordinary_percent = (float(ordinary_pixels) / total_pixels) * 100
    fine_percent = (float(fine_pixels) / total_pixels) * 100

    total_sulfide_pixels = ordinary_pixels + fine_pixels
    if total_sulfide_pixels > 0:
        fine_prevalence = (float(fine_pixels) / total_sulfide_pixels) * 100
    else:
        fine_prevalence = 0.0

    if talc_percent > 10.0:
        ore_class = "Оталькованная руда"
        ore_class_en = "talcified"
        conclusion = (
            f"Руда классифицирована как оталькованная: содержание талька — {talc_percent:.1f}%, "
            f"преобладание тонких срастаний — {fine_prevalence:.1f}%. "
            f"Порог оталькования (10%) превышен."
        )
    else:
        if fine_prevalence <= 40.0:
            ore_class = "Рядовая руда"
            ore_class_en = "ordinary"
            conclusion = (
                f"Руда классифицирована как рядовая: содержание талька — {talc_percent:.1f}%, "
                f"преобладание тонких срастаний — {fine_prevalence:.1f}%. "
                f"Содержание талька в норме, преобладают крупные сплошные сульфиды."
            )
        else:
            ore_class = "Труднообогатимая руда"
            ore_class_en = "hard_to_enrich"
            conclusion = (
                f"Руда классифицирована как труднообогатимая: содержание талька — {talc_percent:.1f}%, "
                f"преобладание тонких срастаний — {fine_prevalence:.1f}%. "
                f"Содержание талька в норме, но преобладают разрушенные тонкие срастания сульфидов."
            )

    overlay = create_overlay(img_bgr, talc_mask, ordinary_mask, fine_mask, ann_mask)

    logging.info(
        f"File: {filename} | Resolution: {w}x{h} | Talc: {talc_percent:.2f}% | "
        f"Ordinary Sulfides: {ordinary_percent:.2f}% | Fine Sulfides: {fine_percent:.2f}% | "
        f"Fine Prevalence: {fine_prevalence:.2f}% | Verdict: {ore_class}"
    )

    return DetectionResult(
        talc_mask=talc_mask,
        sulfide_ordinary_mask=ordinary_mask,
        sulfide_fine_mask=fine_mask,
        annotation_mask=ann_mask,
        talc_fraction=talc_fraction,
        talc_percent=talc_percent,
        sulfide_ordinary_percent=ordinary_percent,
        sulfide_fine_percent=fine_percent,
        fine_prevalence_percent=fine_prevalence,
        ore_class=ore_class,
        ore_class_en=ore_class_en,
        conclusion=conclusion,
        overlay=overlay,
        talc_area_px=talc_pixels,
        ordinary_area_px=ordinary_pixels,
        fine_area_px=fine_pixels
    )
