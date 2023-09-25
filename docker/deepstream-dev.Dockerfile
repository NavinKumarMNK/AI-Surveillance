FROM nvcr.io/nvidia/deepstream:6.3-gc-triton-devel AS deepstream-development
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    net-tools \
    htop \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV CUDA_VER=12.1

WORKDIR /opt/nvidia/deepstream/deepstream
RUN ./install.sh
RUN ./user_additional_install.sh

EXPOSE 22
EXPOSE 8554
EXPOSE 8000
EXPOSE 9001

RUN /opt/nvidia/graph-composer/extension-dev/install_dependencies.sh --allow-root
RUN apt-get install build-essential -y

WORKDIR /workspace
COPY deepstream /workspace/SurveillanceAI/deepstream

WORKDIR /workspace/SurveillanceAI/deepstream
RUN make -C src



CMD ["tail", "-f", "/dev/null"]
