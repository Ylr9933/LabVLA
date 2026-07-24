#!/usr/bin/env bash
# Deploy the JAKA 8-D LabVLA policy through the LabVLA-compatible WebSocket API.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABVLA_ROOT="${LABVLA_ROOT:-$(cd "${DEPLOY_DIR}/.." && pwd)}"
PRETRAINED_PATH="${PRETRAINED_PATH:-}"
if [ -z "${PRETRAINED_PATH}" ]; then
    echo "[ERROR] PRETRAINED_PATH is required." >&2
    echo "        Set PRETRAINED_PATH=/path/to/jaka/checkpoint-N" >&2
    exit 1
fi

VLM_PATH="${VLM_PATH:-/data/rbc/VLM/Qwen3-VL-4B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-31002}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-labvla}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
OUTPUT_CHUNK_SIZE="${OUTPUT_CHUNK_SIZE:-}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-}"
MAX_MESSAGE_SIZE="${MAX_MESSAGE_SIZE:-16777216}"
AUTH_TOKEN="${LABVLA_WS_AUTH_TOKEN:-${AUTH_TOKEN:-}}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
NORM_STATS_PATH="${NORM_STATS_PATH:-}"
TRAINING_REPO_ID="${TRAINING_REPO_ID:-}"
SERVE_ENTRYPOINT="${SERVE_ENTRYPOINT:-serve_jaka.py}"

if [ ! -d "${PRETRAINED_PATH}" ]; then
    echo "[ERROR] checkpoint directory not found: ${PRETRAINED_PATH}" >&2
    exit 1
fi
MODEL_ROOT="${PRETRAINED_PATH}"
if [ ! -f "${MODEL_ROOT}/model.safetensors" ] && [ -f "${MODEL_ROOT}/pretrained_model/model.safetensors" ]; then
    MODEL_ROOT="${MODEL_ROOT}/pretrained_model"
fi
if [ ! -d "${VLM_PATH}" ] && [ ! -f "${MODEL_ROOT}/vlm_config.json" ]; then
    echo "[ERROR] VLM path not found: ${VLM_PATH}" >&2
    echo "        and checkpoint has no self-contained vlm_config.json bundle." >&2
    exit 1
fi

case "${HOST}" in
    127.0.0.1|localhost|::1) ;;
    *)
        if [ -z "${AUTH_TOKEN}" ]; then
            echo "[ERROR] HOST=${HOST} requires LABVLA_WS_AUTH_TOKEN/AUTH_TOKEN." >&2
            exit 2
        fi
        ;;
esac

# Conda's shell hook may reference PS1 internally. Keep strict mode for the
# deployment itself, but disable nounset only during hook evaluation.
set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
set -u
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export LABVLA_ROOT
export PYTHONPATH="${LABVLA_ROOT}:${LABVLA_ROOT}/src"

cmd=(
    python "${DEPLOY_DIR}/${SERVE_ENTRYPOINT}"
    --pretrained_path "${PRETRAINED_PATH}"
    --vlm_path "${VLM_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${DEVICE}"
    --chunk_size "${CHUNK_SIZE}"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --action_mode "auto"
    --max_workers "${MAX_WORKERS}"
    --max_message_size "${MAX_MESSAGE_SIZE}"
)
[ -n "${OUTPUT_CHUNK_SIZE}" ] && cmd+=(--output_chunk_size "${OUTPUT_CHUNK_SIZE}")
[ -n "${MAX_INFLIGHT}" ] && cmd+=(--max_inflight "${MAX_INFLIGHT}")
[ -n "${AUTH_TOKEN}" ] && cmd+=(--auth_token "${AUTH_TOKEN}")
[ -n "${DEFAULT_PROMPT}" ] && cmd+=(--default_prompt "${DEFAULT_PROMPT}")
[ -n "${NORM_STATS_PATH}" ] && cmd+=(--norm_stats_path "${NORM_STATS_PATH}")
[ -n "${TRAINING_REPO_ID}" ] && cmd+=(--repo_id "${TRAINING_REPO_ID}")

echo "[INFO] JAKA deployment: ${HOST}:${PORT}, GPU=${CUDA_DEVICE}"
exec "${cmd[@]}"
