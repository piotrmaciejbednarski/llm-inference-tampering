FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    LLAMA_CPP_DIR=/opt/llama.cpp \
    MODEL_DIR=/models \
    VIRTUAL_ENV=/opt/venv \
    WORKSPACE_DIR=/workspace \
    PATH=/opt/venv/bin:/opt/llama.cpp/build/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        gdb \
        strace \
        procps \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements.txt /tmp/requirements.txt

RUN python3 -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git ${LLAMA_CPP_DIR} \
    && cmake -S ${LLAMA_CPP_DIR} -B ${LLAMA_CPP_DIR}/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
    && cmake --build ${LLAMA_CPP_DIR}/build --config Release -j"$(nproc)"

RUN mkdir -p ${MODEL_DIR} ${WORKSPACE_DIR}

WORKDIR /workspace
