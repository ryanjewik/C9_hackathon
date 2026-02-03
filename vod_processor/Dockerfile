# Multi-stage Dockerfile for VALORANT VOD Processor
# Supports both CPU and GPU modes

# ============================================
# Stage 1: Base with system dependencies
# Use NVIDIA CUDA + cuDNN base for GPU support (PaddleOCR + Surya)
# ============================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python and minimal system packages for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    # Create CUDA/cuDNN symlinks required by PaddlePaddle (needs unversioned .so files)
    && ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so \
    && ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcublas.so \
    && ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcublasLt.so \
    && echo "/usr/local/cuda-12.1/targets/x86_64-linux/lib" >> /etc/ld.so.conf.d/cuda.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Python dependencies  
# ============================================
FROM base AS dependencies

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install dependencies with pre-built wheels (much faster)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# ============================================
# Stage 3: Production image
# ============================================
FROM dependencies AS production

WORKDIR /app

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs /app/logs

# Copy application code
COPY ./app /app/app
COPY ./config /app/config

# Create non-root user for security and set up Surya model cache directory
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/datalab/models \
    && chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Stage 4: Development image (optional)
# ============================================
FROM dependencies AS development

WORKDIR /app

# Install dev dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black isort mypy

# Copy all code
COPY . /app

# Run with reload for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
