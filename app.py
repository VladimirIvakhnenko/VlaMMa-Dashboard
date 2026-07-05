import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
import time
import csv

from detector import analyze_image

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("Arial", font_path))
        pdfmetrics.registerFont(TTFont("Arial-Bold", "C:\\Windows\\Fonts\\arialbd.ttf"))
        USE_CYRILLIC_FONT = True
    else:
        USE_CYRILLIC_FONT = False
except Exception:
    USE_CYRILLIC_FONT = False

st.set_page_config(
    page_title="Скажи мне, кто твой шлиф",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_DIMENSION = 15000
MAX_FILE_SIZE_MB = 100

def load_image(uploaded_file) -> np.ndarray:
    data = uploaded_file.read()
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Файл слишком большой ({len(data) / 1024 / 1024:.0f} МБ). Максимум: {MAX_FILE_SIZE_MB} МБ.")
    file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    del file_bytes
    if img_bgr is None:
        raise ValueError("Не удалось декодировать изображение.")
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_DIMENSION:
        raise ValueError(f"Изображение слишком большое ({w}x{h}). Максимум: {MAX_DIMENSION}x{MAX_DIMENSION} пикселей.")
    return img_bgr

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def calculate_physical_area(pixels: int, um_per_pixel: float) -> float:
    return pixels * ((um_per_pixel / 1000.0) ** 2)

def generate_pdf_report(result, filename: str, um_per_pixel: float) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

    styles = getSampleStyleSheet()
    title_font = "Arial-Bold" if USE_CYRILLIC_FONT else "Helvetica-Bold"
    body_font = "Arial" if USE_CYRILLIC_FONT else "Helvetica"

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName=title_font,
        fontSize=20,
        leading=24,
        textColor="#1e293b",
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=10,
        leading=14,
        textColor="#64748b",
        spaceAfter=20
    )

    header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName=title_font,
        fontSize=14,
        leading=18,
        textColor="#0f172a",
        spaceBefore=15,
        spaceAfter=10
    )

    text_style = ParagraphStyle(
        "ReportText",
        parent=styles["Normal"],
        fontName=body_font,
        fontSize=11,
        leading=16,
        textColor="#334155",
        spaceAfter=8
    )

    bold_text_style = ParagraphStyle(
        "ReportTextBold",
        parent=text_style,
        fontName=title_font
    )

    story = []

    story.append(Paragraph("🔬 АВТОМАТИЧЕСКИЙ ОТЧЕТ АНАЛИЗА ШЛИФА", title_style))
    story.append(Paragraph(f"Файл: {filename} | Дата: {time.strftime('%d.%m.%Y %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Результаты классификации", header_style))
    story.append(Paragraph(f"<b>Класс руды:</b> {result.ore_class}", text_style))
    story.append(Paragraph(f"<b>Заключение:</b> {result.conclusion}", text_style))
    story.append(Spacer(1, 10))

    talc_mm2 = calculate_physical_area(result.talc_area_px, um_per_pixel)
    ord_mm2 = calculate_physical_area(result.ordinary_area_px, um_per_pixel)
    fine_mm2 = calculate_physical_area(result.fine_area_px, um_per_pixel)

    story.append(Paragraph("Количественные метрики", header_style))
    data = [
        [Paragraph("<b>Показатель</b>", bold_text_style),
         Paragraph("<b>Значение (%)</b>", bold_text_style),
         Paragraph("<b>Площадь (мм²)</b>", bold_text_style)],
        [Paragraph("Доля талька", text_style),
         Paragraph(f"{result.talc_percent:.2f}%", text_style),
         Paragraph(f"{talc_mm2:.4f} мм²", text_style)],
        [Paragraph("Обычные сульфидные срастания", text_style),
         Paragraph(f"{result.sulfide_ordinary_percent:.2f}%", text_style),
         Paragraph(f"{ord_mm2:.4f} мм²", text_style)],
        [Paragraph("Тонкие сульфидные срастания", text_style),
         Paragraph(f"{result.sulfide_fine_percent:.2f}%", text_style),
         Paragraph(f"{fine_mm2:.4f} мм²", text_style)],
        [Paragraph("Преобладание тонких срастаний", text_style),
         Paragraph(f"{result.fine_prevalence_percent:.2f}%", text_style),
         Paragraph("-", text_style)],
    ]

    t = Table(data, colWidths=[200, 150, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (2,0), '#f1f5f9'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, '#cbd5e1'),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))

    story.append(Paragraph("Визуализация распределения фаз", header_style))
    story.append(Paragraph("<font color='#2563eb'>■</font> Синий = Тальк | <font color='#16a34a'>■</font> Зеленый = Обычные сульфиды | <font color='#dc2626'>■</font> Красный = Тонкие сульфиды", text_style))
    story.append(Spacer(1, 5))

    oh, ow = result.overlay.shape[:2]
    max_w, max_h = 500, 300
    scale = min(max_w / ow, max_h / oh)
    pdf_w, pdf_h = int(ow * scale), int(oh * scale)

    overlay_small = cv2.resize(result.overlay, (pdf_w, pdf_h), interpolation=cv2.INTER_AREA)
    overlay_rgb_small = cv2.cvtColor(overlay_small, cv2.COLOR_BGR2RGB)
    del overlay_small

    img_pil = Image.fromarray(overlay_rgb_small)
    del overlay_rgb_small
    img_buf = io.BytesIO()
    img_pil.save(img_buf, format="JPEG", quality=85)
    del img_pil
    img_buf.seek(0)

    story.append(RLImage(img_buf, width=pdf_w, height=pdf_h))

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.title("🔬 Настройки")
    st.markdown("---")

    st.subheader("Масштаб и калибровка")
    opt_lens = st.selectbox(
        "Увеличение объектива",
        ["5x (1.50 мкм/пиксель)", "10x (0.75 мкм/пиксель)", "20x (0.37 мкм/пиксель)", "Ввести вручную"]
    )

    if opt_lens == "5x (1.50 мкм/пиксель)":
        um_per_pixel = 1.50
    elif opt_lens == "10x (0.75 мкм/пиксель)":
        um_per_pixel = 0.75
    elif opt_lens == "20x (0.37 мкм/пиксель)":
        um_per_pixel = 0.37
    else:
        um_per_pixel = st.number_input("Разрешение (мкм/пиксель)", min_value=0.01, max_value=100.0, value=1.00, step=0.05)

    st.markdown("---")
    st.subheader("Визуализация")
    overlay_alpha = st.slider(
        "Прозрачность маски", 0.1, 0.9, 0.45, 0.05,
        help="0.1 = только оригинал, 0.9 = только маска"
    )

    st.markdown("---")
    st.caption("Задача 3 · Хакатон 2026")

st.title("🔬 Скажи мне, кто твой шлиф")
st.markdown(
    "**Автоматическая классификация руд по зонам оталькования** на панорамных OM-изображениях полированных шлифов"
)

tab_single, tab_batch, tab_about = st.tabs(["📷 Один шлиф", "📂 Пакетная обработка", "ℹ️ О системе"])

with tab_single:
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Загрузите изображение шлифа",
            type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
            help="Поддерживаются JPEG, PNG, TIFF, BMP"
        )

    with col_info:
        st.info(
            "**Как работает система:**\n"
            "1. Загрузка U-Net ResNet-34\n"
            "2. Сегментация талька скользящим окном\n"
            "3. Разделение сульфидов морфологическим методом\n"
            "4. Формирование PDF-отчёта"
        )

    if uploaded is not None:
        try:
            img_bgr = load_image(uploaded)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        h, w = img_bgr.shape[:2]

        st.markdown(f"**Файл:** `{uploaded.name}` | **Разрешение:** {w}×{h} px | **Размер:** {uploaded.size/1024:.0f} КБ")
        st.markdown("---")

        if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
            with st.spinner("Анализируем шлиф с помощью нейросети U-Net..."):
                t0 = time.time()
                result = analyze_image(img_bgr, filename=uploaded.name)
                elapsed = time.time() - t0

            st.success(f"✅ Анализ завершён за {elapsed:.1f} сек.")

            talc_mm2 = calculate_physical_area(result.talc_area_px, um_per_pixel)
            ord_mm2 = calculate_physical_area(result.ordinary_area_px, um_per_pixel)
            fine_mm2 = calculate_physical_area(result.fine_area_px, um_per_pixel)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    "🔵 Доля талька",
                    f"{result.talc_percent:.2f}%",
                    delta=f"{talc_mm2:.4f} мм²"
                )
            with m2:
                st.metric("🟢 Обычные сульфиды", f"{result.sulfide_ordinary_percent:.2f}%", delta=f"{ord_mm2:.4f} мм²")
            with m3:
                st.metric("🔴 Тонкие сульфиды", f"{result.sulfide_fine_percent:.2f}%", delta=f"{fine_mm2:.4f} мм²")
            with m4:
                st.metric("📈 Тонкие срастания", f"{result.fine_prevalence_percent:.1f}%")

            st.markdown("---")
            st.subheader("Итоговый геологический вердикт")
            if result.ore_class_en == "talcified":
                st.error(f"🔴 **{result.ore_class}**\n\n{result.conclusion}")
            elif result.ore_class_en == "hard_to_enrich":
                st.warning(f"🟡 **{result.ore_class}**\n\n{result.conclusion}")
            else:
                st.success(f"🟢 **{result.ore_class}**\n\n{result.conclusion}")

            st.markdown("---")

            st.subheader("Визуализация распределения фаз")
            vcol1, vcol2 = st.columns(2)

            with vcol1:
                st.caption("📷 Исходное изображение")
                st.image(bgr_to_rgb(img_bgr if img_bgr.shape[0] <= 4096 else
                         cv2.resize(img_bgr, (min(w, 1200), min(h, 900)))),
                         use_container_width=True)

            with vcol2:
                st.caption("🗺️ Оверлей маски (Синий = тальк · Зеленый = обычные сульфиды · Красный = тонкие сульфиды)")

                overlay = result.overlay
                if overlay_alpha != 0.45:
                    from detector import create_overlay
                    overlay = create_overlay(
                        img_bgr,
                        result.talc_mask,
                        result.sulfide_ordinary_mask,
                        result.sulfide_fine_mask,
                        result.annotation_mask,
                        alpha=overlay_alpha
                    )

                display_overlay = overlay if overlay.shape[0] <= 4096 else \
                    cv2.resize(overlay, (min(w, 1200), min(h, 900)))
                st.image(bgr_to_rgb(display_overlay), use_container_width=True)

            st.markdown("---")

            st.subheader("🔬 Экспорт результатов")
            ecol1, ecol2, ecol3 = st.columns(3)

            with ecol1:
                pdf_data = generate_pdf_report(result, uploaded.name, um_per_pixel)
                st.download_button(
                    "⬇️ Скачать PDF Отчет",
                    data=pdf_data,
                    file_name=f"{Path(uploaded.name).stem}_report.pdf",
                    mime="application/pdf"
                )

            with ecol2:
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                del overlay
                overlay_pil = Image.fromarray(overlay_rgb)
                del overlay_rgb
                ov_buf = io.BytesIO()
                overlay_pil.save(ov_buf, format="JPEG", quality=90)
                del overlay_pil
                st.download_button(
                    "⬇️ Скачать оверлей (JPEG)",
                    data=ov_buf.getvalue(),
                    file_name=f"{Path(uploaded.name).stem}_overlay.jpg",
                    mime="image/jpeg"
                )

            with ecol3:
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow([
                    "Файл", "Класс руды", "Доля талька (%)", "Площадь талька (мм²)",
                    "Обычные сульфиды (%)", "Площадь обычных сульфидов (мм²)",
                    "Тонкие сульфиды (%)", "Площадь тонких сульфидов (мм²)",
                    "Преобладание тонких срастаний (%)"
                ])
                writer.writerow([
                    uploaded.name,
                    result.ore_class,
                    f"{result.talc_percent:.2f}",
                    f"{talc_mm2:.6f}",
                    f"{result.sulfide_ordinary_percent:.2f}",
                    f"{ord_mm2:.6f}",
                    f"{result.sulfide_fine_percent:.2f}",
                    f"{fine_mm2:.6f}",
                    f"{result.fine_prevalence_percent:.2f}"
                ])
                st.download_button(
                    "⬇️ Скачать метрики (CSV)",
                    data=csv_buf.getvalue().encode("utf-8-sig"),
                    file_name=f"{Path(uploaded.name).stem}_metrics.csv",
                    mime="text/csv"
                )

    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 60px; border: 2px dashed #555; border-radius: 12px; color: #888;">
                <h3>📂 Загрузите изображение шлифа</h3>
                <p>Поддерживаемые форматы: JPG, PNG, TIFF, BMP</p>
                <p>Разрешение до 15 000 x 15 000 пикселей</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_batch:
    st.subheader("📂 Пакетная обработка")
    st.info("Загрузите несколько изображений для автоматической обработки.")

    uploaded_files = st.file_uploader(
        "Загрузите изображения шлифов",
        type=["jpg", "jpeg", "png", "tiff", "tif", "bmp"],
        accept_multiple_files=True,
        help="Можно выбрать несколько файлов одновременно (Ctrl+click)"
    )

    if uploaded_files:
        st.success(f"Загружено файлов: {len(uploaded_files)}")

        if st.button("🚀 Обработать все", type="primary"):
            results_list = []
            progress_bar = st.progress(0, text="Обработка...")
            status_text = st.empty()

            for i, up_file in enumerate(uploaded_files):
                status_text.text(f"Обрабатываем: {up_file.name} ({i+1}/{len(uploaded_files)})")
                try:
                    file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    del file_bytes
                    if img_bgr is None:
                        results_list.append({
                            "Файл": up_file.name,
                            "Класс руды": "ОШИБКА",
                            "Доля талька (%)": "-",
                            "Вывод": "Не удалось декодировать",
                        })
                        continue
                    h, w = img_bgr.shape[:2]
                    if max(h, w) > MAX_DIMENSION:
                        results_list.append({
                            "Файл": up_file.name,
                            "Класс руды": "ПРОПУСК",
                            "Доля талька (%)": "-",
                            "Вывод": f"Слишком большое ({w}x{h})",
                        })
                        del img_bgr
                        continue
                    result = analyze_image(img_bgr, filename=up_file.name)
                    del img_bgr

                    talc_mm2 = calculate_physical_area(result.talc_area_px, um_per_pixel)
                    ord_mm2 = calculate_physical_area(result.ordinary_area_px, um_per_pixel)
                    fine_mm2 = calculate_physical_area(result.fine_area_px, um_per_pixel)

                    results_list.append({
                        "Файл": up_file.name,
                        "Класс руды": result.ore_class,
                        "Доля талька (%)": f"{result.talc_percent:.2f}",
                        "Площадь талька (мм²)": f"{talc_mm2:.4f}",
                        "Обычные сульфиды (%)": f"{result.sulfide_ordinary_percent:.2f}",
                        "Площадь обычных сульфидов (мм²)": f"{ord_mm2:.4f}",
                        "Тонкие сульфиды (%)": f"{result.sulfide_fine_percent:.2f}",
                        "Площадь тонких сульфидов (мм²)": f"{fine_mm2:.4f}",
                        "Преобладание тонких (%)": f"{result.fine_prevalence_percent:.2f}",
                        "Вывод": result.conclusion,
                    })
                    del result
                except Exception as e:
                    results_list.append({
                        "Файл": up_file.name,
                        "Класс руды": "ОШИБКА",
                        "Доля талька (%)": "-",
                        "Вывод": str(e),
                    })
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("✅ Готово!")
            st.dataframe(results_list, use_container_width=True)

            buf = io.StringIO()
            if results_list:
                writer = csv.DictWriter(buf, fieldnames=results_list[0].keys())
                writer.writeheader()
                writer.writerows(results_list)
                st.download_button(
                    "⬇️ Скачать результаты пакета (CSV)",
                    data=buf.getvalue().encode("utf-8-sig"),
                    file_name="batch_results.csv",
                    mime="text/csv"
                )

with tab_about:
    st.subheader("О системе автоматического минералогического анализа")
    st.markdown("""
    ### Назначение
    End-to-end система классификации руд по панорамным OM-изображениям полированных шлифов.

    ### Классификационная логика ТЗ
    1. **Оталькованная руда**: доля талька > 10%.
    2. **Рядовая руда**: доля талька <= 10%, при этом площадь обычных срастаний больше площади тонких.
    3. **Труднообогатимая руда**: доля талька <= 10%, при этом преобладают тонкие срастания сульфидов.

    ### Цветовая легенда оверлея
    * Синий - Зоны оталькования (тальк).
    * Зеленый - Обычные срастания сульфидов.
    * Красный - Тонкие срастания сульфидов.

    ### Воспроизводимость и логирование
    Все сессии анализа логгируются в файл `analysis_log.txt` в корневой папке проекта. Записывается время запуска, имя файла, его параметры и итоговое геологическое заключение.

    ### Технологический стек
    - **Сегментация**: U-Net (ResNet-34) на фреймворке PyTorch.
    - **Инференс панорам**: Скользящее окно (tiling, патч 512x512, шаг 128) с накоплением и линейным сглаживанием стыков.
    - **Морфология сульфидов**: Анализ контуров на основе эрозии с CLAHE предобработкой освещения.
    - **Экспорт отчетов**: ReportLab PDF Engine.
    """, unsafe_allow_html=True)
