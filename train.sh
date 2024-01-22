# 定义模型名称和微调类型
MODEL_NAME='unsloth/llama-2-7b'
FINETUNING_TYPE="lora"

BASE_MODEL_NAME="${MODEL_NAME##*/}"
# 使用`date`命令生成时间戳
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

# 拼接路径
SAVE_DIR="saves/${BASE_MODEL_NAME}/${FINETUNING_TYPE}/train_${TIMESTAMP}"

# 执行训练命令，使用生成的路径作为output_dir参数
CUDA_VISIBLE_DEVICES=1,2,3,6 accelerate launch --config_file config.yaml src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_NAME} \
    --dataset alpaca_zh\
    --dataloader_persistent_workers \
    --dataloader_num_workers 8\
    --template default \
    --use_unsloth \
    --finetuning_type ${FINETUNING_TYPE}  \
    --lora_target q_proj,v_proj \
    --lora_rank 8\
    --lora_dropout 0.0\
    --output_dir ${SAVE_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size 20 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --save_steps 3000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
