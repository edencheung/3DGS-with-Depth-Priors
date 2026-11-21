FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
RUN python -m pip install --no-cache-dir \
  numpy==1.26.4 \
  plyfile==1.1.3 \
  pillow==12.3.0 \
  tqdm==4.70.1 \
  tensorboard==2.21.0 \
  opencv-python==4.11.0.86 \
  timm==0.6.13 \
  scikit-image==0.25.2 \
  imageio==2.38.0 \
  matplotlib==3.10.9 \
  ninja==1.13.2 \
  setuptools==80.10.2
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*
# compile only for 50XX cards. Use 8.6/8.9 for 30XX and 40XX respectively
ARG CUDA_ARCH_LIST="12.0"
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
  CUDA_HOME=/usr/local/cuda \
  MAX_JOBS=2
RUN python -m pip install --no-cache-dir --no-build-isolation \
    ./gridencoder \
    ./shencoder \
    ./submodules/diff-gaussian-rasterization \
    ./submodules/simple-knn