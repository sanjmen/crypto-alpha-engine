FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies required by LightGBM (libgomp1) and SQLite WAL tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    sqlite3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached Docker layer unless requirements.txt changes)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip wheel && \
    pip install -r /app/requirements.txt

# Copy application source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY README.md /app/README.md

# Prepare persistent directories
RUN mkdir -p /app/data /app/logs

CMD ["python", "scripts/run_trading.py", "--daemon", "--shadow"]
