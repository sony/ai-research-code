FROM nnabla/nnabla-ext-cuda-multi-gpu:py38-cuda110-mpi3.1.6-v1.19.0
USER root

ENV HTTP_PROXY ${http_proxy}
ENV HTTPS_PROXY ${https_proxy}


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt-get install -y libsndfile1 git sox
RUN pip install --upgrade pip
RUN pip install tqdm seaborn sklearn librosa numba==0.48.0 matplotlib sox
