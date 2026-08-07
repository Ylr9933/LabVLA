#!/usr/bin/env bash
set -euo pipefail

ProjRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Assets
VlmPretrainedPath="/data/rbc/VLM/Qwen3-VL-4B-Instruct"
AccelerateBin="/data/rbc/miniconda3/envs/labvla/bin/accelerate"
DataRoot="/data1/xuezirui/data_newest/jaka_10hz_clean_v1"
TrainRoot="${DataRoot}/train"
OutputDir="${ProjRoot}/outputs"
DeepspeedConfig="${ProjRoot}/configs/deepspeed_zero2.json"
PretrainedCkpt="/data1/xuezirui/LabVLA-5B-Base-ckpt"
DatasetSchema="${ProjRoot}/schemas/jaka_v21.py:SCHEMA"
ExternalStatsPath="${TrainRoot}/meta/stats.json"

for RequiredPath in \
    "${AccelerateBin}" \
    "${VlmPretrainedPath}" \
    "${TrainRoot}/meta/info.json" \
    "${ExternalStatsPath}" \
    "${PretrainedCkpt}" \
    "${DatasetSchema%%:*}" \
    "${DeepspeedConfig}"; do
    if [ ! -e "${RequiredPath}" ]; then
        echo "[ERROR] required path missing: ${RequiredPath}" >&2
        exit 1
    fi
done

# Two currently idle A100s. GPU 1 is occupied by another user's simulator.
NumGpus=2
MainProcessPort=29654
export CUDA_VISIBLE_DEVICES="0,3"

# Model and action geometry
Dtype="bfloat16"
DitNumLayers=18
DitNumHeads=8
DitHeadDim=128
ChunkSize=20
MaxStateDim=32
MaxActionDim=32
ImageHeight=224
ImageWidth=224

# Training: global batch = 48 x 2 = 96.
BatchSize=48
GradientAccumulationSteps=1
NumWorkers=8
TotalSteps=4000
SaveFreq=500
LogFreq=50
Seed=42
Lr="5e-5"
VlmLr="5e-5"
DitLr="5e-5"
WeightDecay=0.01
GradClipNorm=1.0
WarmupSteps=500
DecaySteps="${TotalSteps}"
DecayLr="2.5e-6"

FreezeVisionEncoder=true
TrainExpertOnly=true
GradientCheckpointing=true
GcVisualEncoder=true
GcLanguageModel=false
GcDit=false
AttnImplementation="flash_attention_2"

ActionMode="delta"
NormalizeArmJoints=true
NormalizeGripper=true
GripperNormMode="q01_q99"
SnapGripperToBinary=false
GripperMaxWidth=0.04
GripperCanonicalDim=7

TrainingPhase="posttrain"
KnowledgeIsolation=false
UseFastTokenizer=false
FastTokenizerPath="/path/to/fast"
KiMseWeight=10.0

JobName="labvla_finetune_jaka_clean10hz_c20_$(date +%Y%m%d_%H%M%S)"

export PYTHONPATH="src:."
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_ALGO=Ring
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export MALLOC_ARENA_MAX=2

cd "${ProjRoot}"
exec "${AccelerateBin}" launch \
    --num_processes "${NumGpus}" \
    --num_machines 1 \
    --main_process_port "${MainProcessPort}" \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${DeepspeedConfig}" \
    scripts/train.py \
    --vlm_pretrained_path "${VlmPretrainedPath}" \
    --dtype "${Dtype}" \
    --dit_num_layers "${DitNumLayers}" \
    --dit_num_heads "${DitNumHeads}" \
    --dit_head_dim "${DitHeadDim}" \
    --chunk_size "${ChunkSize}" \
    --max_state_dim "${MaxStateDim}" \
    --max_action_dim "${MaxActionDim}" \
    --batch_size "${BatchSize}" \
    --gradient_accumulation_steps "${GradientAccumulationSteps}" \
    --num_workers "${NumWorkers}" \
    --steps "${TotalSteps}" \
    --save_freq "${SaveFreq}" \
    --max_keep_ckpts 8 \
    --log_freq "${LogFreq}" \
    --seed "${Seed}" \
    --image_height "${ImageHeight}" \
    --image_width "${ImageWidth}" \
    --lr "${Lr}" \
    --vlm_lr "${VlmLr}" \
    --dit_lr "${DitLr}" \
    --weight_decay "${WeightDecay}" \
    --grad_clip_norm "${GradClipNorm}" \
    --warmup_steps "${WarmupSteps}" \
    --decay_steps "${DecaySteps}" \
    --decay_lr "${DecayLr}" \
    --freeze_vision_encoder "${FreezeVisionEncoder}" \
    --train_expert_only "${TrainExpertOnly}" \
    --gradient_checkpointing "${GradientCheckpointing}" \
    --gc_visual_encoder "${GcVisualEncoder}" \
    --gc_language_model "${GcLanguageModel}" \
    --gc_dit "${GcDit}" \
    --attn_implementation "${AttnImplementation}" \
    --action_mode "${ActionMode}" \
    --normalize_arm_joints "${NormalizeArmJoints}" \
    --normalize_gripper "${NormalizeGripper}" \
    --gripper_norm_mode "${GripperNormMode}" \
    --snap_gripper_to_binary "${SnapGripperToBinary}" \
    --gripper_max_width "${GripperMaxWidth}" \
    --gripper_canonical_dim "${GripperCanonicalDim}" \
    --training_phase "${TrainingPhase}" \
    --knowledge_isolation "${KnowledgeIsolation}" \
    --use_fast_tokenizer "${UseFastTokenizer}" \
    --fast_tokenizer_path "${FastTokenizerPath}" \
    --ki_mse_weight "${KiMseWeight}" \
    --repo_ids train \
    --data_root "${DataRoot}" \
    --dataset_schema "${DatasetSchema}" \
    --external_stats_path "${ExternalStatsPath}" \
    --output_dir "${OutputDir}" \
    --job_name "${JobName}" \
    --resume "${PretrainedCkpt}" \
    --load_weights_only true
