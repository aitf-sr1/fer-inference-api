FROM python:3.13-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN uv sync --frozen --no-dev \
    && find /app/.venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /app/.venv -type f -name '*.pyc' -delete 2>/dev/null || true \
    && find /app/.venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true


FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=2
ENV OMP_DYNAMIC=FALSE

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "8", \
     "--bind", "0.0.0.0:8000", "--capture-output", \
     "--enable-stdio-inheritance", "fer_inference_api.main:app"]
