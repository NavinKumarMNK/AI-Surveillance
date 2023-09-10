FROM qdrant/qdrant:latest AS qdrant
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

EXPOSE 6333
EXPOSE 6334

RUN apt-get update && apt-get upgrade -y
CMD ["tail", "-f", "/dev/null"]
