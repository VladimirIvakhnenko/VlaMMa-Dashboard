"""
Анализатор геолого-технологических сортов руды (FINAL v7)
Нормализация к единому формату + выбор метода детекции талька (CV/UNet)
"""

import cv2
import numpy as np
import streamlit as st
import pandas as pd
import json
import io
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. КОНФИГУРАЦИЯ
# ============================================================================
DEFAULT_CONFIG = {
    "processing": {
        "max_resolution": 2048,
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": 16,
        "normalize_method": "clahe"  # clahe, histogram, adaptive
    },
    "sulfide": {
        "percentile": 88.0,
        "min_area": 800,
        "morph_kernel": 7,
        "morph_close_iter": 2
    },
    "replacement": {
        "intensity_thresh": 140,
        "ratio_threshold": 0.35
    },
    "talc": {
        "detection_method": "cv",  # cv или unet
        "bh_kernel": 15,
        "morph_kernel": 11,
        "min_area": 400,
        "max_area": 20000,
        "halo_radius": 30
    },
    "classification": {
        "talc_threshold": 10.0
    }
}

def load_config() -> dict:
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    if Path("config.json").exists():
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            for sec in DEFAULT_CONFIG:
                if sec in user_cfg:
                    cfg[sec].update(user_cfg[sec])
        except Exception:
            pass
    return cfg

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

def load_image_robust(uploaded_file):
    """Безопасная загрузка OM-изображений с сохранением 16-битного диапазона"""
    img = Image.open(uploaded_file)
    
    if img.mode in ('I;16', 'I', 'I;16B'):
        arr = np.array(img, dtype=np.float32)
        arr = (arr / 65535.0) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
        
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = img.convert('RGB')
        
    return np.array(img)

# ============================================================================
# 2. НОРМАЛИЗАЦИЯ К ЕДИНОМУ ФОРМАТУ
# ============================================================================
def normalize_to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """
    Конвертация в grayscale с учетом восприятия яркости человеком
    Использует weighted average (ITU-R BT.601)
    """
    # Weighted grayscale: Y = 0.299*R + 0.587*G + 0.114*B
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return gray

def normalize_histogram(gray: np.ndarray) -> np.ndarray:
    """Глобальное выравнивание гистограммы"""
    return cv2.equalizeHist(gray)

def normalize_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 16) -> np.ndarray:
    """CLAHE - адаптивное выравнивание контраста"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)

def normalize_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Адаптивная нормализация для OM-снимков:
    1. Gamma correction для тёмных снимков
    2. CLAHE для локального контраста
    3. Линейное растяжение для стабилизации диапазона
    """
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Gamma correction для тёмных снимков
    if mean_val < 80:
        gamma = 0.7 if mean_val < 60 else 0.85
        gray = 255.0 * np.power(gray.astype(np.float32) / 255.0, gamma)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # Линейное растяжение контраста
    if std_val < 40:
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    # CLAHE для локального контраста
    clip_limit = 2.0 if mean_val >= 80 else 1.5
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
    return clahe.apply(gray)

def normalize_for_unet(img_rgb: np.ndarray) -> np.ndarray:
    """
    Нормализация для UNet:
    - Приведение к [0, 1]
    - Нормализация по ImageNet stats (или dataset-specific)
    """
    # Нормализация к [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # ImageNet normalization (mean, std per channel)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_norm - mean) / std
    
    return img_norm

def preprocess_image(img_rgb: np.ndarray, cfg: dict) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Полная предобработка:
    1. Масштабирование для производительности
    2. Нормализация к единому формату (CV)
    3. Подготовка для UNet (если нужно)
    
    Returns:
        gray_eq: нормализованное grayscale для CV
        scale: коэффициент масштабирования
        img_for_unet: нормализованное RGB для UNet (или None)
    """
    h, w = img_rgb.shape[:2]
    max_dim = cfg['processing']['max_resolution']
    scale = min(1.0, max_dim / max(h, w))
    
    # Масштабирование
    if scale < 1.0:
        img_proc = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    else:
        img_proc = img_rgb.copy()
    
    # Конвертация в grayscale
    gray = normalize_to_grayscale(img_proc)
    
    # Нормализация по выбранному методу
    method = cfg['processing'].get('normalize_method', 'adaptive')
    if method == 'clahe':
        gray_eq = normalize_clahe(gray, 
                                   cfg['processing']['clahe_clip_limit'],
                                   cfg['processing']['clahe_tile_grid'])
    elif method == 'histogram':
        gray_eq = normalize_histogram(gray)
    elif method == 'adaptive':
        gray_eq = normalize_adaptive(gray)
    else:
        gray_eq = gray
    
    # Подготовка для UNet (если используется)
    img_for_unet = None
    if cfg['talc'].get('detection_method') == 'unet':
        img_for_unet = normalize_for_unet(img_proc)
    
    return gray_eq, scale, img_for_unet

# ============================================================================
# 3. ДЕТЕКЦИЯ И КЛАССИФИКАЦИЯ
# ============================================================================
def detect_sulfides(gray_eq: np.ndarray, cfg: dict) -> np.ndarray:
    """Выделение сульфидов по верхнему процентилю яркости"""
    thresh = np.percentile(gray_eq, cfg['sulfide']['percentile'])
    mask = (gray_eq >= thresh).astype(np.uint8) * 255
    k = np.ones((cfg['sulfide']['morph_kernel'],)*2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=cfg['sulfide']['morph_close_iter'])
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

def classify_aggregates(gray_eq: np.ndarray, sulfide_mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Разделение сульфидов на обычные и замещённые"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sulfide_mask, connectivity=8)
    ordinary, refractory = np.zeros_like(sulfide_mask), np.zeros_like(sulfide_mask)
    
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg['sulfide']['min_area']: 
            continue
        agg_mask = (labels == i)
        if np.mean(gray_eq[agg_mask]) < cfg['replacement']['intensity_thresh']:
            refractory[agg_mask] = 255
        else:
            ordinary[agg_mask] = 255
            
    total = sulfide_mask.sum()
    ord_pct = 100.0 * ordinary.sum() / total if total else 0
    ref_pct = 100.0 * refractory.sum() / total if total else 0
    return ordinary, refractory, ord_pct, ref_pct

def detect_talc_cv(gray_eq: np.ndarray, sulfide_mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """Детекция талька через CV: Black-Hat морфология"""
    talc_cfg = cfg['talc']
    
    # Нерудная зона
    halo_k = np.ones((talc_cfg['halo_radius']*2+1,)*2, np.uint8)
    non_ore_zone = cv2.bitwise_not(cv2.dilate(sulfide_mask, halo_k, iterations=1))
    
    # Black-Hat
    bh_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(talc_cfg['bh_kernel']),)*2)
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, bh_k)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Только в нерудной зоне
    talc_cand = cv2.bitwise_and(thresh, thresh, mask=non_ore_zone)
    
    # Морфология
    close_k = np.ones((ensure_odd(talc_cfg['morph_kernel']),)*2, np.uint8)
    talc_mask = cv2.morphologyEx(talc_cand, cv2.MORPH_CLOSE, close_k, iterations=2)
    talc_mask = cv2.morphologyEx(talc_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    
    # Фильтрация по площади
    n, labels, stats, _ = cv2.connectedComponentsWithStats(talc_mask, connectivity=8)
    final_talc = np.zeros_like(talc_mask)
    valid = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if talc_cfg['min_area'] <= area <= talc_cfg['max_area']:
            final_talc[labels == i] = 255
            valid += 1
            
    talc_pct = 100.0 * np.count_nonzero(final_talc) / gray_eq.size
    return final_talc, talc_pct, valid, non_ore_zone

def predict_talc_unet(img_for_unet: np.ndarray, cfg: dict) -> Tuple[np.ndarray, float]:
    """
    Заглушка для UNet сегментации талька
    TODO: Заменить на реальный инференс модели
    
    Args:
        img_for_unet: нормализованное изображение [batch, channels, height, width]
        cfg: конфигурация
    
    Returns:
        talc_mask: бинарная маска талька
        talc_pct: процент талька
    """
    if img_for_unet is None:
        return np.zeros(img_for_unet.shape[:2], dtype=np.uint8), 0.0
    
    # TODO: Реальный инференс
    # 1. Загрузить модель: model = torch.load('talc_unet.pth')
    # 2. Подготовить вход: input_tensor = torch.from_numpy(img_for_unet).permute(2, 0, 1).unsqueeze(0)
    # 3. Предсказать: with torch.no_grad(): output = model(input_tensor)
    # 4. Постобработка: talc_mask = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255
    
    # Заглушка: возвращаем пустую маску
    h, w = img_for_unet.shape[:2]
    talc_mask = np.zeros((h, w), dtype=np.uint8)
    talc_pct = 0.0
    
    return talc_mask, talc_pct

def detect_talc(gray_eq: np.ndarray, sulfide_mask: np.ndarray, 
                img_for_unet: Optional[np.ndarray], cfg: dict) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """Универсальная функция детекции талька (CV или UNet)"""
    method = cfg['talc'].get('detection_method', 'cv')
    
    if method == 'unet' and img_for_unet is not None:
        talc_mask, talc_pct = predict_talc_unet(img_for_unet, cfg)
        # Для совместимости возвращаем dummy значения
        halo_k = np.ones((cfg['talc']['halo_radius']*2+1,)*2, np.uint8)
        non_ore_zone = cv2.bitwise_not(cv2.dilate(sulfide_mask, halo_k, iterations=1))
        return talc_mask, talc_pct, 0, non_ore_zone
    else:
        return detect_talc_cv(gray_eq, sulfide_mask, cfg)

def generate_report(talc_pct, ord_pct, ref_pct, talc_thresh, rep_thresh):
    """Генерация отчёта строго по ТЗ"""
    if talc_pct > talc_thresh:
        t, txt = "Оталькованная", f"Руда классифицирована как оталькованная: содержание талька — {talc_pct:.1f}%, преобладание тонких срастаний — {ref_pct:.0f}%."
    elif ref_pct > rep_thresh*100:
        t, txt = "Труднообогатимая", f"Руда классифицирована как труднообогатимая: содержание талька — {talc_pct:.1f}%, преобладание тонких срастаний — {ref_pct:.0f}%."
    else:
        t, txt = "Рядовая", f"Руда классифицирована как рядовая: содержание талька — {talc_pct:.1f}%, преобладание обычных срастаний — {ord_pct:.0f}%."
    return {"ore_type": t, "text": txt, "metrics": {
        "Классификация": t, 
        "Доля талька (%)": round(talc_pct,1), 
        "Доля обычных срастаний (%)": round(ord_pct,1), 
        "Доля тонких срастаний (%)": round(ref_pct,1)
    }}

# ============================================================================
# 4. UI
# ============================================================================
def main():
    st.set_page_config(page_title="Классификатор руд v7", layout="wide")
    st.title("Геолого-технологическая классификация руд")
    st.caption("Зеленый: обычные | Красный: тонкие/замещённые | Синий: тальк")
    
    cfg = load_config()
    st.sidebar.header("Параметры")
    uploaded = st.sidebar.file_uploader("Загрузить панораму", type=['tif','tiff','png','jpg','jpeg'])
    
    # Классификация
    cfg['classification']['talc_threshold'] = st.sidebar.slider("Порог талька (%)", 5.0, 25.0, cfg['classification']['talc_threshold'], 0.5)
    cfg['replacement']['ratio_threshold'] = st.sidebar.slider("Порог тонких срастаний (%)", 10.0, 70.0, cfg['replacement']['ratio_threshold']*100, 1.0)/100.0
    
    # Детекция
    with st.sidebar.expander("Настройки детекции"):
        # Нормализация
        st.subheader("Нормализация")
        normalize_method = st.selectbox(
            "Метод нормализации",
            ["adaptive", "clahe", "histogram"],
            index=["adaptive", "clahe", "histogram"].index(cfg['processing'].get('normalize_method', 'adaptive'))
        )
        cfg['processing']['normalize_method'] = normalize_method
        
        # Сульфиды
        st.subheader("Сульфиды")
        cfg['sulfide']['percentile'] = st.slider("Яркость сульфидов (перцентиль)", 70.0, 99.0, cfg['sulfide']['percentile'], 1.0)
        cfg['replacement']['intensity_thresh'] = st.slider("Порог яркости замещения", 100, 220, cfg['replacement']['intensity_thresh'], 5)
        
        # Тальк
        st.subheader("Тальк")
        detection_method = st.selectbox(
            "Метод детекции талька",
            ["cv", "unet"],
            index=["cv", "unet"].index(cfg['talc'].get('detection_method', 'cv'))
        )
        cfg['talc']['detection_method'] = detection_method
        
        if detection_method == 'cv':
            bh = st.slider("Размер зерна талька (BH Kernel)", 5, 51, cfg['talc']['bh_kernel'], 2)
            cfg['talc']['bh_kernel'] = ensure_odd(bh)
            
            morph = st.slider("Морфология талька", 5, 31, cfg['talc']['morph_kernel'], 2)
            cfg['talc']['morph_kernel'] = ensure_odd(morph)
            
            cfg['talc']['halo_radius'] = st.slider("Буфер вокруг сульфидов", 5, 150, cfg['talc']['halo_radius'], 5)
            cfg['talc']['min_area'] = st.slider("Мин. площадь талька", 100, 5000, cfg['talc']['min_area'], 100)
            cfg['talc']['max_area'] = st.slider("Макс. площадь талька", 1000, 80000, cfg['talc']['max_area'], 1000)
        else:
            st.info("UNet модель: заглушка (будет заменена на обученные веса)")
            st.warning("Для работы UNet необходимо обучить модель на размеченных данных")
        
    if st.sidebar.button("Сохранить config.json"):
        with open("config.json", "w", encoding="utf-8") as f: 
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        st.sidebar.success("Сохранено")

    if not uploaded:
        st.info("Загрузите изображение для анализа.")
        st.markdown("""
        ### Метод нормализации:
        - **adaptive**: автоматическая подстройка под тёмные/светлые снимки (рекомендуется)
        - **clahe**: адаптивное выравнивание контраста
        - **histogram**: глобальное выравнивание гистограммы
        
        ### Метод детекции талька:
        - **cv**: Black-Hat морфология (быстро, интерпретируемо)
        - **unet**: нейросетевая сегментация (точнее, требует обучения)
        """)
        return

    # Загрузка изображения
    img_rgb = load_image_robust(uploaded)
    if len(img_rgb.shape)==2: 
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2]==4: 
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    
    with st.spinner("Обработка..."):
        # Предобработка с нормализацией
        gray_eq, scale, img_for_unet = preprocess_image(img_rgb, cfg)
        
        # Детекция
        sulf_mask = detect_sulfides(gray_eq, cfg)
        ord_m, ref_m, ord_p, ref_p = classify_aggregates(gray_eq, sulf_mask, cfg)
        talc_m, talc_p, _, non_ore = detect_talc(gray_eq, sulf_mask, img_for_unet, cfg)
        rep = generate_report(talc_p, ord_p, ref_p, cfg['classification']['talc_threshold'], cfg['replacement']['ratio_threshold'])
        
        # Визуализация
        h_s, w_s = gray_eq.shape
        vis = cv2.resize(img_rgb, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        
        mask_vis = np.zeros_like(vis)
        mask_vis[ord_m>0] = [0, 200, 0]
        mask_vis[ref_m>0] = [200, 0, 0]
        mask_vis[talc_m>0] = [0, 0, 200]
        
        alpha = 0.15
        blend = vis.copy()
        mask_bool = mask_vis.any(axis=2)
        blend[mask_bool] = cv2.addWeighted(vis, 1-alpha, mask_vis, alpha, 0)[mask_bool]

    # Вывод
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: 
        st.image(blend, caption="Зеленый: обычные | Красный: тонкие | Синий: тальк", use_column_width=True)
    with c2: 
        st.metric("Тип руды", rep['ore_type'])
        st.metric("Тальк", f"{talc_p:.1f}%")
    with c3: 
        st.metric("Обычные", f"{ord_p:.1f}%")
        st.metric("Тонкие", f"{ref_p:.1f}%")
        
    st.success(rep['text'])
    
    # DataFrame с метриками
    metrics_data = {
        "Параметр": list(rep['metrics'].keys()),
        "Значение": [str(v) for v in rep['metrics'].values()]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Экспорт
    if st.button("Скачать отчёт и маску"):
        csv = io.StringIO()
        metrics_df.to_csv(csv, index=False)
        st.download_button("metrics.csv", csv.getvalue(), "metrics.csv", "text/csv")
        
        buf = io.BytesIO()
        success, encoded = cv2.imencode('.png', cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))
        if success:
            buf.write(encoded.tobytes())
            st.download_button("mask.png", buf.getvalue(), "mask.png", "image/png")
        else:
            st.error("Ошибка кодирования PNG")

if __name__ == "__main__":
    main()