FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# System deps (OpenCV runtime)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv $VIRTUAL_ENV

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default command shows CLI help; can be overridden in compose
CMD ["python", "sampling_tool.py", "--help"]

