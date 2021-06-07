# FROM ubuntu:latest
# FROM nvidia/cuda:10.2-base
FROM nvidia/cuda:10.2-base-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install base packages.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
RUN apt-get update --fix-missing && apt-get install -y tzdata && apt-get install -y bzip2 ca-certificates curl gcc git libc-dev libglib2.0-0 libsm6 libxext6 libxrender1 wget libevent-dev build-essential &&  rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN /opt/conda/bin/conda update -n base -c defaults conda && \
    /opt/conda/bin/conda install pytorch torchvision cudatoolkit=10.2 -c pytorch && \
    /opt/conda/bin/conda install tqdm lxml nltk && \
    python -m nltk.downloader punkt && \
    /opt/conda/bin/pip install transformers==3.0.2 yattag
RUN /opt/conda/bin/conda clean -tipsy

# RUN /bin/bash -c "source ~/.bashrc"

# oneie env
ADD ./models /models
ADD ./oneie /oneie

# # echo "conda activate oneie" > ~/.bashrc && \
# RUN conda activate oneie
# RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# RUN conda install tqdm lxml nltk
# RUN python -m nltk.downloader punkt
# RUN /opt/conda/envs/oneie/bin/pip install transformers
# ENV PATH /opt/conda/envs/oenie/bin:$PATH
# RUN conda activate oneie
# RUN echo "conda activate oneie" > ~/.bashrc
# && \
# /opt/conda/envs/oneie/bin/pip install pytorch torchvision cudatoolkit=10.2 -c pytorch && \
# /opt/conda/envs/oneie/bin/pip install tqdm transformers lxml nltk && \
# /opt/conda/envs/oneie/bin/python -m nltk.downloader punkt



LABEL maintainer="hengji@illinois.edu"

CMD ["/bin/bash"]

