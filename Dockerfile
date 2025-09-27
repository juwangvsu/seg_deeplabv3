# CUDA 12.1 + Ubuntu 22.04 runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirement files
COPY requirements.txt /workspace/requirements.txt

# Install pip deps
RUN python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy source
COPY . /workspace

# Default command
CMD ["/bin/bash"]
