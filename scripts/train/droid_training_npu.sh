#!/bin/bash
# DreamZero DROID Training Script for Ascend NPU (8-card FSDP + LoRA)
#
# Usage:
#   bash scripts/train/droid_training_npu.sh
#
# Prerequisites:
#   - DROID dataset at DROID_DATA_ROOT
#   - Wan2.1-I2V-14B-480P weights at WAN_CKPT_DIR
#   - umt5-xxl tokenizer at TOKENIZER_DIR

export HYDRA_FULL_ERROR=1
export DREAMZERO_DEVICE=npu
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512

# HCCL config for single-node multi-die
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=128

# NOTE: If Dataloader Bus error (shared memory exhausted), restart container
# with --shm-size=16g, then change dataloader_num_workers back to 1.

# ============ USER CONFIGURATION ============
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"/data/droid"}
OUTPUT_DIR=${OUTPUT_DIR:-"/data/droid/checkpoints/dreamzero_droid_npu"}
NUM_GPUS=${NUM_GPUS:-8}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-1000}
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
FSDP_CONFIG=${FSDP_CONFIG:-"/workspace/fsdp_config_v27.json"}
# =============================================

# Auto-download weights
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Downloading Wan2.1-I2V-14B-480P..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "Downloading umt5-xxl tokenizer..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    exit 1
fi

echo "=== NPU FSDP Training Configuration ==="
echo "NUM_GPUS:       $NUM_GPUS"
echo "DATA_ROOT:      $DROID_DATA_ROOT"
echo "OUTPUT_DIR:     $OUTPUT_DIR"
echo "MAX_STEPS:      $MAX_STEPS"
echo "SAVE_STEPS:     $SAVE_STEPS"
echo "WAN_CKPT:       $WAN_CKPT_DIR"
echo "Device:         $DREAMZERO_DEVICE"
echo "Dist Backend:   hccl"
echo "Strategy:       FSDP full_shard auto_wrap"
echo "FSDP Config:    $FSDP_CONFIG"
echo "=================================="

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=$MAX_STEPS \
    save_steps=$SAVE_STEPS \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=false \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    training_args.fsdp=full_shard\ auto_wrap \
    training_args.fsdp_transformer_layer_cls_to_wrap=CausalWanAttentionBlock \
    training_args.fsdp_config=$FSDP_CONFIG \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
