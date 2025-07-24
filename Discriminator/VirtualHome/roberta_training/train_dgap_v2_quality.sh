#!/bin/bash

# DGAP-v2 CriticQ (Quality) Training Script  
# Based on paper: AdamW, lr=1e-5, warmup=0.1, batch_size=32

export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed --num_gpus=4 ds_train_dgap_v2_critics.py \
    --deepspeed dgap_v2_deepspeed_config.json \
    --model_name_or_path roberta-base \
    --critic_type quality \
    --output_dir ./models/dgap_v2_quality \
    --overwrite_output_dir \
    --do_train \
    --num_train_epochs 8 \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_dir ./logs/dgap_v2_quality \
    --logging_steps 50 \
    --save_steps 500 \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --disable_tqdm false \
    --report_to wandb \
    --run_name dgap_v2_quality_$(date +%Y%m%d_%H%M%S) \
    --seed 42 \
    --bf16 true 