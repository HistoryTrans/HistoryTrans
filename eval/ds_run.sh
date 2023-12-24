# 使用DeepSpeed加速推理的模型评估脚本

# 注意事项！
# 1.安装DeepSpeed: pip install deepspeed
# 2.确保run_eval.sh文件具有执行权限: chmod +x ds_run.sh
# 3.如果需要强行终止脚本，请使用: ps aux | grep ds_eval.py | awk '{print $2}' | xargs kill -9

#!/bin/bash
clear

# 设置模型路径
MODEL_PATH='/root/.cache/huggingface/hub/models--THUDM--chatglm3-6b'
# 设置评估数据文件路径
EVAL_FILE="../data/version4/sampled_1000_merged_output_20230812_190843.json"
# 设置输出路径
OUTPUT_PATH="./data/eval_results/"
# 设置批处理大小
BATCH_SIZE=1
# 设置模型并行的GPU数量
MP_SIZE=1
# 设置生成文本的最大长度
MAX_LENGTH=1024
# 设置生成过程中的Top-p参数
TOP_P=0.8
# 设置生成过程中的温度参数
TEMPERATURE=0.1


echo "Start evaluating..."

# 运行评估脚本
python ds_eval.py --model_path $MODEL_PATH --eval_file $EVAL_FILE --output_path $OUTPUT_PATH --batch_size $BATCH_SIZE --mp_size $MP_SIZE --max_length $MAX_LENGTH --top_p $TOP_P --temperature $TEMPERATURE

echo "Evaluation finished."