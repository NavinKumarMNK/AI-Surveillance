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

COPY ../requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /opt/nvidia/deepstream/deepstream
RUN ./install.sh
RUN ./user_additional_install.sh
RUN ./user_deepstream_python_apps_install.sh -b

EXPOSE 22
EXPOSE 8554
EXPOSE 8000

WORKDIR /workspace
RUN /opt/nvidia/graph-composer/extension-dev/install_dependencies.sh --allow-root

RUN apt-get update && apt-get upgrade -y
CMD ["tail", "-f", "/dev/null"]
