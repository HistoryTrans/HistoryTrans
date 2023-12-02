#! /usr/bin/env bash
clear
source /etc/network_turbo
ps aux | grep finetune.py | awk '{print $2}' | xargs kill -9

set -ex

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
MAX_SOURCE_LEN=256
MAX_TARGET_LEN=136
DEV_BATCH_SIZE=20
GRAD_ACCUMULARION_STEPS=2
MAX_STEP=4000
SAVE_INTERVAL=800

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=ancient_gen_pt

BASE_MODEL_PATH="THUDM/chatglm3-6b"
DATASET_PATH=formatted_data/train.jsonl
# DATASET_PATH=formatted_data/val.jsonl
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${PRE_SEQ_LEN}-${LR}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
