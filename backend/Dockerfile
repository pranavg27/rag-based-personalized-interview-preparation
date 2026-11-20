FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
# CPU-only torch keeps the image small; remove this line if you have a GPU base image
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt /srv/requirements.txt
RUN pip install -r /srv/requirements.txt
COPY backend/app /srv/app
RUN useradd -m appuser && mkdir -p /srv/data && chown -R appuser /srv
USER appuser
ENV PYTHONPATH=/srv HF_HOME=/srv/data/hf_cache
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
