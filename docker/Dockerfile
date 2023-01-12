# syntax = docker/dockerfile:experimental
ARG CUDA_VERSION=11.1.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON_VER=python3.8
RUN echo $PYTHON_VER
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update --fix-missing && \
    apt-get -y install --fix-missing $PYTHON_VER && \
    apt-get -y install --fix-missing ${PYTHON_VER}-dev && \
    apt-get -y install --fix-missing python3-pip && \
    $PYTHON_VER -m pip install pip --upgrade

# Put default python to the installed version
RUN rm -f /usr/bin/python                                   && \
    ln -s /usr/bin/$PYTHON_VER /usr/bin/python
RUN rm -f /usr/bin/python3                                  && \
    ln -s /usr/bin/$PYTHON_VER /usr/bin/python3

# System packages
RUN apt-get update && apt-get install -y curl git

# git
RUN apt-get update && \
    apt-get install -y git

# Fix UTF-8
ENV LANG=C.UTF-8

# Set workdir
WORKDIR /app
COPY . /app

## gcloud SDK
#RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y
#RUN gcloud auth activate-service-account --key-file=key.json
#RUN rm key.json

## Install Mecab
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends tzdata g++ git curl
# RUN apt-get install python3-setuptools
# RUN apt-get install -y default-jdk default-jre

# Install kamino, transformers-addons
RUN --mount=type=secret,id=github_secret pip install git+https://$(cat /run/secrets/github_secret)@github.com/42maru-ai/Kamino.git@v0.2.2
RUN rm github_secret.txt

# python requirements
RUN pip install -r requirements.txt

## Install Mecab
# RUN curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -s
