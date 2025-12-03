#!/usr/bin/env bash
set -euo pipefail

# Adjust these for your machine
GPUS=4
MASTER_PORT=29500

# Base Qwen3-0.6B model on HF
BASE_MODEL="Qwen/Qwen3-0.6B-Base"

# Meta file we created
META_JSON="shell/playground/data/meta/meta_cnn_qwen3_0_6b.json"

# Deepspeed ZeRO config (you can start with stage 2 for 0.6B)
ZERO_CFG="zero_stage2_config.json"

OUTPUT_DIR="outputs/qwen3_0_6b_cnn_sdlm"

mkdir -p "${OUTPUT_DIR}"

deepspeed \
  --num_gpus ${GPUS} \
  --master_port ${MASTER_PORT} \
  sdlm/train.py \
  --model_name_or_path "${BASE_MODEL}" \
  --meta_file "${META_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --block_size 4 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --num_train_epochs 3 \
  --logging_steps 50 \
  --save_steps 2000 \
  --save_total_limit 3 \
  --evaluation_strategy "steps" \
  --eval_steps 2000 \
  --bf16 True \
  --deepspeed "${ZERO_CFG}" \
  --attn_implementation "sdpa" \
  --causal_attn False \
  --ddp_timeout 36000
