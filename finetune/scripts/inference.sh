clear
source /etc/network_turbo
ps aux | grep inference.py | awk '{print $2}' | xargs kill -9


PT_CHECKPOINT="./output/ancient_gen_pt-20231106-173414-128-2e-2/checkpoint-1000"

python inference.py \
    --pt-checkpoint $PT_CHECKPOINT \
    --model THUDM/chatglm3-6b 