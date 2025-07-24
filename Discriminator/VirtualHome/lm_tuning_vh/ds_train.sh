#!/bin/bash


# 设置环境变量
USE_TF=0

# 使用3GPU避开4GPU evaluation bug
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 解决PyTorch 2.6 weights_only问题（老版本PyTorch不需要）
# export TORCH_WEIGHTS_ONLY=False
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 解决多进程内存共享权限问题
#export TORCH_MULTIPROCESSING_SHARE_STRATEGY=file_system
#export OMP_NUM_THREADS=1
EXPERT_DATA_PATH="/home/msj/planning/d2c/VirtualHome/virtualhome/dataset/flant5_training_data/vh_expert_37k_t5.jsonl"
OUTPUT_DIR="/home/msj/planning/d2c/Discriminator/VirtualHome/lm_tuning_vh/outputs/new_flan_t5_vh_expert"
CACHE_DIR="/home/msj/planning/d2c/Discriminator/VirtualHome/lm_tuning_vh/cache"
deepspeed --master_port 29510 \
		./ds_train.py \
	--cache_dir $CACHE_DIR \
        --model_name_or_path ./flan-t5-large \
        --output_dir $OUTPUT_DIR \
        --do_train \
	--do_eval \
	--save_total_limit 5 \
        --train_file $EXPERT_DATA_PATH \
	--validation_file $EXPERT_DATA_PATH \
	--predict_with_generate 0 \
        --learning_rate 5e-5 \
	--adam_eps 1e-06 \
        --max_source_length 512 \
        --max_target_length 64 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --eval_accumulation_steps 2 \
        --dataloader_pin_memory False \
	--metric_for_best_model eval_loss \
	--greater_is_better False \
	--deepspeed zero_2_bf16.json \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 8 \
	--early_stopping_patience 5 \
	--logging_steps 50 \
	--save_strategy steps \
	--eval_strategy steps \
	--save_steps 400 \
	--eval_steps 400 \
	--seed 42 \
	--report_to wandb \
	--run_name flan_t5_vh \
	--warmup_steps 100 \
	--weight_decay 0.01

