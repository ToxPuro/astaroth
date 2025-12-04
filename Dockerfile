# Start from the same CUDA base image
FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

# Set timezone to Europe/Helsinki (optional)
RUN ln -fs /usr/share/zoneinfo/Europe/Helsinki /etc/localtime

# Install dependencies as in your workflow
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
      apt-transport-https \
      ca-certificates \
      gnupg \
      software-properties-common \
      wget && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ noble main' && \
    apt-get update && \
    apt-get install -y \
      cmake \
      flex \
      bison \
      openmpi-bin \
      libopenmpi-dev \
      vim \
      gdb \
      gfortran && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set workdir (optional, adjust as needed)
WORKDIR /workspace

# Copy your code (if building locally)
# COPY . .

# If you want to build something inside container on build time, add RUN commands here
# RUN cmake . && make

# Default command, adjust as needed
CMD ["/bin/bash"]

