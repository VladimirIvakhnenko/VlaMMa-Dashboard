# ============================================================================
# Этап 1: Сборка зависимостей (build stage)
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements для кэширования слоёв
COPY requirements.txt .

# Устанавливаем зависимости в виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Этап 2: Финальный образ (runtime stage)
# ============================================================================
FROM python:3.11-slim-bookworm

# Метаданные
LABEL maintainer="VlaMMa Team"
LABEL description="Геолого-технологическая классификация руд по OM-панорамам"
LABEL version="1.0"

# ВАЖНО: opencv-python-headless НЕ требует libGL/libSM/libXrender
# Нужны только базовые библиотеки для scikit-image и Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаём невысокого пользователя
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Рабочая директория
WORKDIR /app

# Копируем виртуальное окружение из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Копируем код приложения
COPY ore_analyzer.py .

# Создаём директории для данных и результатов
RUN mkdir -p /app/data /app/results /app/config && \
    chown -R appuser:appuser /app

# Переключаемся на невысокого пользователя
USER appuser

# Streamlit конфигурация
RUN mkdir -p /home/appuser/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\naddress = "0.0.0.0"\nenableCORS = false\nmaxUploadSize = 500\n' \
    > /home/appuser/.streamlit/config.toml

# Открываем порт Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Запуск приложения
CMD ["streamlit", "run", "ore_analyzer.py", "--server.port=8501", "--server.address=0.0.0.0"]