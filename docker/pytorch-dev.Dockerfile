# Path: MLOps/PyTorch.DockerFile

FROM nvcr.io/nvidia/pytorch:23.04-py3
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


WORKDIR /workspace

COPY ../src/arcface /workspace/arcface
WORKDIR /workspace/arcface

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 22
EXPOSE 8888
EXPOSE 8080
EXPOSE 5000

RUN chmod -R a+w /workspace
RUN apt-get update && apt-get upgrade -y

CMD ["tail", "-f", "/dev/null"]
