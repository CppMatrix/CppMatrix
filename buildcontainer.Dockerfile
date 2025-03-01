FROM ubuntu:24.04

RUN apt update && apt install -y \
    wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8 && \
    rm cuda-keyring_1.1-1_all.deb

RUN apt install -y \
    cmake \
    clang \
    clang-tools \
    libgtest-dev \
    ninja-build
