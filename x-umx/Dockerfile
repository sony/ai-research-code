# NOTE: This Dockerfile is not actively maintained.
ARG CUDA_VER=11.0
ARG CUDNN_VER=8

FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-runtime-ubuntu18.04

ARG PYTHON_VER=3.7
ARG CUDA_VER=11.0
ENV PATH /opt/miniconda3/bin:$PATH
ENV OMP_NUM_THREADS 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    ssh \
    && rm -rf /var/lib/apt/lists/*

RUN umask 0 \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && conda install -y python=${PYTHON_VER} \
    && pip install -U setuptools \
    && conda install -y opencv jupyter

RUN umask 0 \
    && pip install nnabla-ext-cuda`echo $CUDA_VER | sed 's/\.//g'`-nccl2-mpi2-1-1

RUN umask 0 \
    && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda$(echo $CUDA_VER | sed "s/\.//g")


RUN umask 0 \
    && pip install \
       musdb \
       norbert \
       resampy \
       sklearn \
       pydub \
       soundfile \
    && conda install -c conda-forge -y \
      ffmpeg \
      libsndfile
