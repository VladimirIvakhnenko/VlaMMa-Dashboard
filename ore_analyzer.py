"""
🔬 Анализатор геолого-технологических сортов руды (FINAL v6 - исправлено)
Полное соответствие ТЗ + обработка 16-bit TIFF + защита тёмных снимков
"""

import cv2
import numpy as np
import streamlit as st
import pandas as pd
import json
import io
from pathlib import Path
from typing import Dict, Tuple
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
        "clahe_tile_grid": 16
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
    """Гарантирует нечётное число для ядер морфологии OpenCV"""
    return n if n % 2 == 1 else n + 1

def load_image_robust(uploaded_file):
    """Безопасная загрузка OM-изображений с сохранением 16-битного диапазона"""
    img = Image.open(uploaded_file)
    
    # Нормализация 16-bit TIFF -> 8-bit без потери контраста
    if img.mode in ('I;16', 'I', 'I;16B'):
        arr = np.array(img, dtype=np.float32)
        arr = (arr / 65535.0) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
        
    # Приведение к RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode == 'L':
        img = img.convert('RGB')
        
    return np.array(img)

def auto_gamma(gray: np.ndarray) -> float:
    """Подбирает gamma на основе распределения яркости"""
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
    cumsum = np.cumsum(hist) / hist.sum()
    
    # Если 50% пикселей темнее 60 — нужно сильное усиление
    if cumsum[60] > 0.5:
        return 0.6
    elif cumsum[80] > 0.5:
        return 0.8
    else:
        return 1.0  # Без коррекции

# ============================================================================
# 2. ПРЕДОБРАБОТКА (ИСПРАВЛЕНА)
# ============================================================================
def preprocess_image(img_rgb: np.ndarray, cfg: dict) -> Tuple[np.ndarray, float]:
    """
    Адаптивная предобработка для OM-панорам:
    - Gamma-коррекция для тёмных снимков
    - CLAHE для локального контраста
    - Масштабирование для производительности
    """
    h, w = img_rgb.shape[:2]
    max_dim = cfg['processing']['max_resolution']
    scale = min(1.0, max_dim / max(h, w))
    
    # Масштабируем для анализа
    if scale < 1.0:
        img_proc = cv2.resize(img_rgb, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    else:
        img_proc = img_rgb.copy()
        
    gray = cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # === АДАПТИВНОЕ УСИЛЕНИЕ ДЛЯ ТЁМНЫХ СНИМКОВ ===
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    # Если изображение очень тёмное И имеет низкий контраст
    if mean_val < 80 and std_val < 40:
        # 1. Gamma-коррекция: поднимаем тени (gamma < 1)
        gamma = auto_gamma(gray)
        gray = 255.0 * np.power(gray / 255.0, gamma)
        
        # 2. Лёгкое линейное растяжение контраста
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # 3. CLAHE с мягкими параметрами (чтобы не усилить шум)
        clip_limit = min(cfg['processing']['clahe_clip_limit'], 1.5)
    else:
        # Для нормальных/светлых снимков — стандартный CLAHE
        clip_limit = cfg['processing']['clahe_clip_limit']
        gray = gray.astype(np.uint8)
    
    # Применяем CLAHE (если ещё не преобразовали в uint8)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                            tileGridSize=(cfg['processing']['clahe_tile_grid'],)*2)
    gray_eq = clahe.apply(gray)
    
    return gray_eq, scale

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
    """
    Разделение сульфидов на обычные (светлые) и замещённые (тёмные внутри)
    Returns: ordinary_mask, refractory_mask, ordinary_pct, refractory_pct
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(sulfide_mask, connectivity=8)
    ordinary, refractory = np.zeros_like(sulfide_mask), np.zeros_like(sulfide_mask)
    
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < cfg['sulfide']['min_area']: 
            continue
        agg_mask = (labels == i)
        # Если средняя яркость внутри агрегата низкая → замещение
        if np.mean(gray_eq[agg_mask]) < cfg['replacement']['intensity_thresh']:
            refractory[agg_mask] = 255
        else:
            ordinary[agg_mask] = 255
            
    total = sulfide_mask.sum()
    ord_pct = 100.0 * ordinary.sum() / total if total else 0
    ref_pct = 100.0 * refractory.sum() / total if total else 0
    return ordinary, refractory, ord_pct, ref_pct

def detect_talc_robust(gray_eq: np.ndarray, sulfide_mask: np.ndarray, cfg: dict) -> Tuple[np.ndarray, float, int, np.ndarray]:
    """
    Детекция талька: тёмные рассеянные области ВНЕ сульфидов
    Метод: Black-Hat морфология + адаптивные пороги
    """
    talc_cfg = cfg['talc']
    
    # 1. Нерудная зона (вне сульфидов + буфер)
    halo_k = np.ones((talc_cfg['halo_radius']*2+1,)*2, np.uint8)
    non_ore_zone = cv2.bitwise_not(cv2.dilate(sulfide_mask, halo_k, iterations=1))
    
    # 2. Black-Hat: выделяет тёмные пятна на светлом фоне
    bh_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ensure_odd(talc_cfg['bh_kernel']),)*2)
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, bh_k)
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Оставляем только в нерудной зоне
    talc_cand = cv2.bitwise_and(thresh, thresh, mask=non_ore_zone)
    
    # 4. Морфология: замыкаем рассеянные зёрна в зоны
    close_k = np.ones((ensure_odd(talc_cfg['morph_kernel']),)*2, np.uint8)
    talc_mask = cv2.morphologyEx(talc_cand, cv2.MORPH_CLOSE, close_k, iterations=2)
    talc_mask = cv2.morphologyEx(talc_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    
    # 5. Фильтрация по площади
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
# 4. UI (ИСПРАВЛЕН)
# ============================================================================
def main():
    st.set_page_config(page_title="Классификатор руд", layout="wide")
    st.title("🔬 Геолого-технологическая классификация руд")
    st.caption("🟢 Обычные | 🔴 Тонкие/Замещённые | 🔵 Тальк")
    
    cfg = load_config()
    st.sidebar.header("⚙️ Параметры")
    uploaded = st.sidebar.file_uploader("Загрузить панораму", type=['tif','tiff','png','jpg','jpeg'])
    
    # Классификация
    cfg['classification']['talc_threshold'] = st.sidebar.slider("Порог талька (%)", 5.0, 25.0, cfg['classification']['talc_threshold'], 0.5)
    cfg['replacement']['ratio_threshold'] = st.sidebar.slider("Порог тонких срастаний (%)", 10.0, 70.0, cfg['replacement']['ratio_threshold']*100, 1.0)/100.0
    
    # Детекция
    with st.sidebar.expander("🔧 Настройки детекции"):
        cfg['sulfide']['percentile'] = st.slider("Яркость сульфидов (перцентиль)", 70.0, 99.0, cfg['sulfide']['percentile'], 1.0)
        cfg['replacement']['intensity_thresh'] = st.slider("Порог яркости замещения", 100, 220, cfg['replacement']['intensity_thresh'], 5)
        
        st.markdown("---")
        st.caption("🌫 Тальк")
        
        bh = st.slider("Размер зерна талька (BH Kernel)", 5, 51, cfg['talc']['bh_kernel'], 2)
        cfg['talc']['bh_kernel'] = ensure_odd(bh)
        
        morph = st.slider("Морфология талька", 5, 31, cfg['talc']['morph_kernel'], 2)
        cfg['talc']['morph_kernel'] = ensure_odd(morph)
        
        cfg['talc']['halo_radius'] = st.slider("Буфер вокруг сульфидов", 5, 150, cfg['talc']['halo_radius'], 5)
        cfg['talc']['min_area'] = st.slider("Мин. площадь талька", 100, 5000, cfg['talc']['min_area'], 100)
        cfg['talc']['max_area'] = st.slider("Макс. площадь талька", 1000, 80000, cfg['talc']['max_area'], 1000)
        
    if st.sidebar.button("💾 Сохранить config.json"):
        with open("config.json", "w", encoding="utf-8") as f: 
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        st.sidebar.success("✅ Сохранено")

    if not uploaded:
        st.info("👆 Загрузите изображение для анализа.")
        return

    # Загрузка изображения (16-bit safe)
    img_rgb = load_image_robust(uploaded)
    if len(img_rgb.shape)==2: 
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.shape[2]==4: 
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    
    with st.spinner("Обработка..."):
        # ✅ ИСПРАВЛЕНО: распаковываем 2 значения, как возвращает функция
        gray_eq, scale = preprocess_image(img_rgb, cfg)
        
        sulf_mask = detect_sulfides(gray_eq, cfg)
        ord_m, ref_m, ord_p, ref_p = classify_aggregates(gray_eq, sulf_mask, cfg)
        talc_m, talc_p, _, non_ore = detect_talc_robust(gray_eq, sulf_mask, cfg)
        rep = generate_report(talc_p, ord_p, ref_p, cfg['classification']['talc_threshold'], cfg['replacement']['ratio_threshold'])
        
        # Приводим оригинал к размеру анализа для наложения маски
        h_s, w_s = gray_eq.shape
        vis = cv2.resize(img_rgb, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        
        # Цветная маска: 🟢 обычные, 🔴 замещённые, 🔵 тальк
        mask_vis = np.zeros_like(vis)
        mask_vis[ord_m>0] = [0, 200, 0]
        mask_vis[ref_m>0] = [200, 0, 0]
        mask_vis[talc_m>0] = [0, 0, 200]
        
        # Мягкое наложение (15% маска, 85% оригинал)
        alpha = 0.15
        blend = vis.copy()
        mask_bool = mask_vis.any(axis=2)
        blend[mask_bool] = cv2.addWeighted(vis, 1-alpha, mask_vis, alpha, 0)[mask_bool]

    # === ВЫВОД ===
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: 
        st.image(blend, caption="🟢 Обычные | 🔴 Тонкие | 🔵 Тальк", use_column_width=True)
    with c2: 
        st.metric("Тип руды", rep['ore_type'])
        st.metric("Тальк", f"{talc_p:.1f}%")
    with c3: 
        st.metric("Обычные", f"{ord_p:.1f}%")
        st.metric("Тонкие", f"{ref_p:.1f}%")
        
    st.success(f"📝 {rep['text']}")
    st.dataframe(pd.DataFrame([rep['metrics']]).T, use_container_width=True)
    
    # Экспорт
    if st.button("📥 Скачать отчёт и маску"):
        csv = io.StringIO()
        pd.DataFrame([rep['metrics']]).T.to_csv(csv)
        st.download_button("metrics.csv", csv.getvalue(), "metrics.csv", "text/csv")
        
        buf = io.BytesIO()
        cv2.imencode('.png', cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))[1].tofile(buf)
        st.download_button("mask.png", buf.getvalue(), "mask.png", "image/png")

if __name__ == "__main__":
    main()