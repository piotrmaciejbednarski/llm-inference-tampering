#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf?download=1}"
MODEL_FILE="${MODEL_FILE:-tinyllama-1.1b-chat-q4_k_m.gguf}"
TARGET_PATH="${MODEL_DIR}/${MODEL_FILE}"

mkdir -p "${MODEL_DIR}"

if [[ -f "${TARGET_PATH}" ]]; then
    echo "Model already present at ${TARGET_PATH}"
    exit 0
fi

echo "Downloading model to ${TARGET_PATH}"
curl -L --fail --output "${TARGET_PATH}" "${MODEL_URL}"
echo "Model download complete"
