# 基于大型语言模型的古文翻译

```bash
# 在开始之前，请确保安装了以下依赖：
pip install rouge_chinese nltk jieba datasets
```

## 数据准备

要将数据集处理成所需格式，请运行以下命令：

```bash
chmod +x ./scripts/format_ancient_gen.py
./scripts/format_ancient_gen.py --path "ancient_data/train.json"
./scripts/format_ancient_gen.py --path "ancient_data/val.json"
```

## 模型微调
```bash
# tmux会话管理
tmux new -s my_training_session
tmux attach -t my_training_session

# 网络优化
source /etc/network_turbo
```

- P-Tuning v2 微调

如保存微调过程的日志，并且带有时间戳，可以使用以下命令：

```bash
# 查看显存情况
nvidia-smi -l 2

# 运行
chmod +x ./scripts/finetune_pt.sh
./scripts/finetune_pt.sh 2>&1 | ts '%F %T' | tee finetune_log.txt
```
终止之前的训练进程：

```bash
ps aux | grep finetune.py | awk '{print $2}' | xargs kill -9
```


### 推理验证

对于输入输出格式的微调，可使用 `inference.py` 进行基本的推理验证。

```bash
chmod +x ./scripts/inference.sh
./scripts/inference.sh
```

```bash
python inference.py \
    --tokenizer THUDM/chatglm3-6b \
    --model "path to finetuned model checkpoint" 
```