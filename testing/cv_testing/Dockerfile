FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages required for OpenCV, Tesseract and image libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        wget \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

# Install CPU-only PyTorch wheels first (preferred) and then other deps
RUN pip --no-cache-dir install --index-url https://download.pytorch.org/whl/cpu torch torchvision
RUN pip --no-cache-dir install -r /app/requirements.txt

# Copy project
COPY . /app

# Default entry: run the test pipeline. You can override with docker run <image> <cmd>
CMD ["python", "test.py"]
