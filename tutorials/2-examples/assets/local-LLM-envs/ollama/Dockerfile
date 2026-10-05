FROM ollama/ollama
  
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update \
   && apt-get install -y \
    build-essential \
    gcc g++ gfortran \
    cmake \
    make \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    vim \
    flex \
    hwloc \
    gawk \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN python3 -m venv .deeptrack && \
    . .deeptrack/bin/activate && \
    pip3 install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir deeptrack opencv-python && \
    echo 'source /opt/.deeptrack/bin/activate' >> /root/.bashrc

CMD ["/bin/bash"]