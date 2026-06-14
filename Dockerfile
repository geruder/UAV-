# UAV Intent Estimation — FastAPI service + static dashboard.
FROM python:3.12-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Install deps. The CPU torch index keeps the image small (no CUDA).
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# App code + trained model + frontend (heavy data artifacts excluded via .dockerignore).
COPY core ./core
COPY api ./api
COPY data ./data
COPY model ./model
COPY web ./web

EXPOSE 8000
# Honor the platform-provided $PORT (Render/Railway/Fly); default 8000 locally.
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
