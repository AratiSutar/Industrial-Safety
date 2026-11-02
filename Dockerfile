# Industrial Safety Monitor: CPU-only image.
# Build:  docker compose build      Run:  docker compose up      View:  http://localhost:8000
FROM python:3.11-slim

# OpenCV needs these system libraries even without a screen
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# CPU-only PyTorch first: the default wheel brings ~2 GB of GPU (CUDA) libraries we can't use
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY safety/ safety/
COPY tools/ tools/
COPY main.py .
COPY config/settings.yaml config/settings.yaml

# Don't run as root. models/ and logs/ are mounted from the host (see docker-compose.yml).
RUN useradd --create-home app && mkdir -p models logs && chown app models logs
USER app

EXPOSE 8000
ENTRYPOINT ["python", "main.py"]
CMD ["--web", "8000"]
