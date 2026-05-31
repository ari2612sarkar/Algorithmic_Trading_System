# Algorithmic Trading System — Streamlit dashboard image
FROM python:3.11-slim

# System deps: libgomp is required by LightGBM; build-essential for any wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 build-essential curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install CPU-only PyTorch first (smaller, no CUDA), then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Persisted at runtime via the compose volume; created here so first run works.
RUN mkdir -p data models reports

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
