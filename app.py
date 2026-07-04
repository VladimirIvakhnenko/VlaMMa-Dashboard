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

def load_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_pdf_report(result, filename: str) -> bytes:
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
    
    story.append(Paragraph("Количественные метрики", header_style))
    data = [
        [Paragraph("<b>Показатель</b>", bold_text_style), Paragraph("<b>Значение</b>", bold_text_style)],
        [Paragraph("Доля талька (%)", text_style), Paragraph(f"{result.talc_percent:.2f}%", text_style)],
        [Paragraph("Обычные сульфидные срастания (%)", text_style), Paragraph(f"{result.sulfide_ordinary_percent:.2f}%", text_style)],
        [Paragraph("Тонкие сульфидные срастания (%)", text_style), Paragraph(f"{result.sulfide_fine_percent:.2f}%", text_style)],
        [Paragraph("Преобладание тонких срастаний среди сульфидов (%)", text_style), Paragraph(f"{result.fine_prevalence_percent:.2f}%", text_style)],
    ]
    
    t = Table(data, colWidths=[320, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), '#f1f5f9'),
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
    
    overlay_rgb = bgr_to_rgb(result.overlay)
    oh, ow = overlay_rgb.shape[:2]
    max_w, max_h = 500, 300
    scale = min(max_w / ow, max_h / oh)
    pdf_w, pdf_h = int(ow * scale), int(oh * scale)
    
    img_pil = Image.fromarray(overlay_rgb)
    img_buf = io.BytesIO()
    img_pil.save(img_buf, format="JPEG", quality=85)
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
        img_bgr = load_image(uploaded)
        h, w = img_bgr.shape[:2]

        st.markdown(f"**Файл:** `{uploaded.name}` | **Разрешение:** {w}×{h} px | **Размер:** {uploaded.size/1024:.0f} КБ")
        st.markdown("---")

        if st.button("🚀 Запустить анализ", type="primary", use_container_width=True):
            with st.spinner("Анализируем шлиф с помощью нейросети U-Net..."):
                t0 = time.time()
                result = analyze_image(img_bgr)
                elapsed = time.time() - t0

            st.success(f"✅ Анализ завершён за {elapsed:.1f} сек.")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric(
                    "🔵 Доля талька",
                    f"{result.talc_percent:.2f}%",
                    delta="Порог: 10.0%"
                )
            with m2:
                st.metric("🟢 Обычные сульфиды", f"{result.sulfide_ordinary_percent:.2f}%")
            with m3:
                st.metric("🔴 Тонкие сульфиды", f"{result.sulfide_fine_percent:.2f}%")
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
                
                overlay = result.overlay.copy()
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
                pdf_data = generate_pdf_report(result, uploaded.name)
                st.download_button(
                    "⬇️ Скачать PDF Отчет",
                    data=pdf_data,
                    file_name=f"{Path(uploaded.name).stem}_report.pdf",
                    mime="application/pdf"
                )

            with ecol2:
                overlay_pil = Image.fromarray(bgr_to_rgb(overlay))
                ov_buf = io.BytesIO()
                overlay_pil.save(ov_buf, format="JPEG", quality=90)
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
                    "Файл", "Класс руды", "Доля талька (%)", 
                    "Обычные сульфиды (%)", "Тонкие сульфиды (%)", "Преобладание тонких срастаний (%)"
                ])
                writer.writerow([
                    uploaded.name,
                    result.ore_class,
                    f"{result.talc_percent:.2f}",
                    f"{result.sulfide_ordinary_percent:.2f}",
                    f"{result.sulfide_fine_percent:.2f}",
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
    st.subheader("📂 Пакетная обработка папок")
    st.info("Укажите путь к папке с изображениями для автоматической обработки всех файлов.")

    folder_path = st.text_input(
        "Путь к папке на компьютере",
        placeholder="C:\\Users\\...\\Фото руд",
        help="Будут обработаны все JPG/PNG/TIFF файлы в папке"
    )

    if folder_path and os.path.isdir(folder_path):
        folder = Path(folder_path)
        img_files = (
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.JPG")) +
            list(folder.glob("*.png")) +
            list(folder.glob("*.PNG")) +
            list(folder.glob("*.tiff")) +
            list(folder.glob("*.TIFF"))
        )
        st.success(f"Найдено {len(img_files)} изображений")

        if img_files and st.button("🚀 Обработать все", type="primary"):
            results_list = []
            progress_bar = st.progress(0, text="Обработка...")
            status_text = st.empty()

            for i, img_path in enumerate(img_files):
                status_text.text(f"Обрабатываем: {img_path.name} ({i+1}/{len(img_files)})")
                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue
                    result = analyze_image(img_bgr)
                    results_list.append({
                        "Файл": img_path.name,
                        "Класс руды": result.ore_class,
                        "Доля талька (%)": f"{result.talc_percent:.2f}",
                        "Обычные сульфиды (%)": f"{result.sulfide_ordinary_percent:.2f}",
                        "Тонкие сульфиды (%)": f"{result.sulfide_fine_percent:.2f}",
                        "Преобладание тонких (%)": f"{result.fine_prevalence_percent:.2f}",
                        "Вывод": result.conclusion,
                    })
                except Exception as e:
                    results_list.append({
                        "Файл": img_path.name,
                        "Класс руды": "ОШИБКА",
                        "Доля талька (%)": "-",
                        "Вывод": str(e),
                    })
                progress_bar.progress((i + 1) / len(img_files))

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
    elif folder_path:
        st.error(f"Папка не найдена: {folder_path}")

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

    ### Технологический стек
    - **Сегментация**: U-Net (ResNet-34) на фреймворке PyTorch.
    - **Инференс панорам**: Скользящее окно (tiling, патч 512x512, шаг 128) с накоплением и линейным сглаживанием стыков.
    - **Морфология сульфидов**: Анализ контуров на основе эрозии.
    - **Экспорт отчетов**: ReportLab PDF Engine.
    """, unsafe_allow_html=True)
