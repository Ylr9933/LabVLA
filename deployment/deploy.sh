#!/usr/bin/env bash
set -euo pipefail

DeployDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ProjRoot="$(cd "${DeployDir}/.." && pwd)"

# ── Paths (edit before running) ──
VlmPath="/path/to/Qwen3-VL-4B-Instruct"
PretrainedPath="/path/to/checkpoint"

if [ -z "${PretrainedPath}" ] || [ "${PretrainedPath}" = "/path/to/checkpoint" ]; then
    echo "[ERROR] PretrainedPath is required. Set it inside the script before running." >&2
    exit 1
fi

# ── Server ──
Host="127.0.0.1"
Port=31002
Device="cuda"
CudaDevice="0"
MaxMessageSize=16777216
MaxWorkers=4
AuthToken=""

# ── Model ──
ChunkSize=50
OutputChunkSize=""
NumInferenceSteps=10
ActionMode="auto"
DefaultPrompt=""
NormStatsPath=""
TrainingRepoId=""

NormalizeArmJoints=true
NormalizeGripper=true
GripperNormMode="q01_q99"
SnapGripperToBinary=false
GripperMaxWidth=0.04
GripperCanonicalDim=7

# ── Environment ──
export PYTHONPATH="src:."
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="${CudaDevice}"
export LABVLA_ROOT="${ProjRoot}"

# ── Validate ──
if [ ! -d "${PretrainedPath}" ]; then
    echo "[ERROR] Checkpoint not found: ${PretrainedPath}" >&2
    exit 1
fi

if [ -f "${PretrainedPath}/model.safetensors" ]; then
    ModelPath="${PretrainedPath}"
elif [ -f "${PretrainedPath}/pretrained_model/model.safetensors" ]; then
    ModelPath="${PretrainedPath}/pretrained_model"
else
    echo "[ERROR] model.safetensors not found in ${PretrainedPath} or ${PretrainedPath}/pretrained_model" >&2
    exit 1
fi

if [ ! -f "${ModelPath}/vlm_config.json" ] && [ ! -d "${VlmPath}" ]; then
    echo "[ERROR] VLM path not found: ${VlmPath} (and checkpoint has no self-contained vlm_config.json bundle)" >&2
    exit 1
fi

if [ ! -d "${ProjRoot}/src" ]; then
    echo "[ERROR] ProjRoot/src not found: ${ProjRoot}/src" >&2
    exit 1
fi

case "${Host}" in
    127.0.0.1|localhost|::1) ;;
    *)
        if [ -z "${AuthToken}" ]; then
            echo "[ERROR] Refusing Host=${Host} without AuthToken." >&2
            echo "        Use Host=127.0.0.1 for local-only serving or set AuthToken." >&2
            exit 2
        fi
        ;;
esac

# ── Launch ──
cd "${ProjRoot}"

echo "============================================================"
echo " LabVLA Deployment Server"
echo "============================================================"
echo "  Checkpoint : ${PretrainedPath}"
echo "  VLM        : ${VlmPath}"
echo "  Device     : ${Device} (CUDA_VISIBLE_DEVICES=${CudaDevice})"
echo "  Host:Port  : ${Host}:${Port}"
echo "  Workers    : ${MaxWorkers}"
echo "  Action mode: ${ActionMode}"
echo "============================================================"

Cmd=(
    python "${DeployDir}/serve_labvla.py"
    --pretrained_path "${PretrainedPath}"
    --vlm_path "${VlmPath}"
    --host "${Host}"
    --port "${Port}"
    --device "${Device}"
    --chunk_size "${ChunkSize}"
    --num_inference_steps "${NumInferenceSteps}"
    --action_mode "${ActionMode}"
    --normalize_arm_joints "${NormalizeArmJoints}"
    --normalize_gripper "${NormalizeGripper}"
    --gripper_norm_mode "${GripperNormMode}"
    --snap_gripper_to_binary "${SnapGripperToBinary}"
    --gripper_max_width "${GripperMaxWidth}"
    --gripper_canonical_dim "${GripperCanonicalDim}"
    --max_message_size "${MaxMessageSize}"
    --max_workers "${MaxWorkers}"
)

[ -n "${OutputChunkSize}" ] && Cmd+=(--output_chunk_size "${OutputChunkSize}")
[ -n "${DefaultPrompt}" ]   && Cmd+=(--default_prompt "${DefaultPrompt}")
[ -n "${NormStatsPath}" ]   && Cmd+=(--norm_stats_path "${NormStatsPath}")
[ -n "${TrainingRepoId}" ]  && Cmd+=(--repo_id "${TrainingRepoId}")
[ -n "${AuthToken}" ]       && Cmd+=(--auth_token "${AuthToken}")

exec "${Cmd[@]}"
