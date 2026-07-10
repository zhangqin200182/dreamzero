#!/bin/bash
set -e
cd /workspace/dreamzero
export HYDRA_FULL_ERROR=1 DREAMZERO_DEVICE=npu
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_BUFFSIZE=128

# NOTE: If Dataloader Bus error, restart container with --shm-size=16g,
# then change dataloader_num_workers back to 1 for faster data loading.

echo "=== 16-GPU + TensorBoard ==="

torchrun --nproc_per_node 16 --standalone \
    groot/vla/experiment/experiment.py \
    report_to=tensorboard \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 action_horizon=24 num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.warmup_ratio=0.05 \
    output_dir=/data/droid/checkpoints/dreamzero_droid_npu_16gpu_v2 \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=2 \
    max_steps=100000 save_steps=200 \
    weight_decay=1e-5 save_total_limit=50 upload_checkpoints=false \
    bf16=true tf32=false eval_bf16=true \
    dataloader_pin_memory=false dataloader_num_workers=1 \
    image_resolution_width=320 image_resolution_height=176 \
    save_lora_only=true max_chunk_size=4 frame_seqlen=880 \
    save_strategy=steps \
    training_args.fsdp=full_shard\ auto_wrap \
    training_args.fsdp_transformer_layer_cls_to_wrap=CausalWanAttentionBlock \
    training_args.fsdp_config=/workspace/fsdp_config_v27.json \
    droid_data_root=/data/droid \
    dit_version=/checkpoints/Wan2.1-I2V-14B-480P \
    text_encoder_pretrained_path=/checkpoints/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=/checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=/checkpoints/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth \
    tokenizer_path=/checkpoints/umt5-xxl
