FROM nnabla/nnabla-ext-cuda-multi-gpu:py38-cuda110-mpi3.1.6-v1.19.0
USER root

ENV HTTP_PROXY ${http_proxy}
ENV HTTPS_PROXY ${https_proxy}

RUN apt-get update
RUN apt-get install -y libsndfile1 git sox
RUN pip install --upgrade pip
RUN pip install tqdm seaborn sklearn librosa numba==0.48.0 matplotlib sox