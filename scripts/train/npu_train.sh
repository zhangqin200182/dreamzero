#!/bin/bash
# DreamZero NPU Training Launcher
# Runs inside aura-a3 container on Ascend NPU
set -e

export DREAMZERO_DEVICE=npu
export HYDRA_FULL_ERROR=1
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

echo "=========================================="
echo " DreamZero NPU Training Setup"
echo "=========================================="

# ---------- install deps (once per container) ----------
echo "[1/4] Checking dependencies..."
MISSING=""
python3 -c "import diffusers" 2>/dev/null || MISSING="$MISSING diffusers"
python3 -c "import dm_tree" 2>/dev/null || MISSING="$MISSING dm-tree"
python3 -c "import imageio" 2>/dev/null || MISSING="$MISSING imageio"

if [ -n "$MISSING" ]; then
    echo "  Installing:$MISSING ..."
    pip install --no-cache-dir -q \
        diffusers==0.30.2 decord2 dm-tree albumentations==1.4.18 \
        imageio==2.34.2 imageio-ffmpeg msgpack-numpy ftfy \
        redis tyro matplotlib termcolor lmdb \
        meshcat meshcat-shapes timm gymnasium pygame h5py
    echo "  Done."
else
    echo "  All present."
fi

# ---------- detect device ----------
echo "[2/4] Device detection..."
python3 -c "
from groot.vla.common.utils.device import *
print(f'  DEVICE: {DEVICE_TYPE} | {DEVICE_STR}')
print(f'  accel: {is_accelerator_available()} | dist: {get_dist_backend()}')
print(f'  count: {get_device_count()} | FA: {gpu_supports_flash_attention()}')
"

# ---------- model weights ----------
WAN_CKPT=${WAN_CKPT_DIR:-/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER=${TOKENIZER_DIR:-/checkpoints/umt5-xxl}
DATA=${DROID_DATA_ROOT:-/data/droid_mirror}
OUTPUT=${OUTPUT_DIR:-/checkpoints/dreamzero_npu}
NUM_GPUS=${NUM_GPUS:-8}

echo "[3/4] Checking weights & data..."
if [ ! -d "$WAN_CKPT" ]; then
    echo "  Downloading Wan2.1-I2V-14B-480P to $WAN_CKPT ..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT"
fi
if [ ! -d "$TOKENIZER" ]; then
    echo "  Downloading umt5-xxl to $TOKENIZER ..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER"
fi
if [ ! -d "$DATA" ]; then
    echo "  ERROR: Dataset not found at $DATA"
    echo "  Download: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir $DATA"
    exit 1
fi

echo "[4/4] Launching training..."
echo "  GPUs: $NUM_GPUS | Output: $OUTPUT | Steps: ${MAX_STEPS:-10}"

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
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT \
    per_device_train_batch_size=1 \
    max_steps=${MAX_STEPS:-10} \
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
    save_strategy=no \
    "training_args.fsdp=full_shard auto_wrap" \
    training_args.fsdp_transformer_layer_cls_to_wrap=CausalWanAttentionBlock \
    droid_data_root=$DATA \
    dit_version=$WAN_CKPT \
    text_encoder_pretrained_path=$WAN_CKPT/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER

echo "Training complete!"
