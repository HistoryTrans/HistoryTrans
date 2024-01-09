---
license: mit
task_categories:
- translation
language:
- zh
tags:
- translation
- 古文翻译
- 文言文翻译
pretty_name: 古文翻译数据集
size_categories:
- 100M<n<1B
---
# HistoryTrans

HistoryTrans 是一个古文翻译数据集，通过数据预处理和质量控制，来提高古文翻译的质量和实用性。

参考我们的项目主页[HistoryTrans古文翻译](https://github.com/HistoryTrans/HistoryTrans).

## 数据集详细信息

### 数据集来源

- **主体:** [Classical-Modern](https://github.com/NiuTrans/Classical-Modern)
- **额外补充:** ：《二十四史》和《清史稿》中提取

### 数据集结构

数据集包含以下 JSONL 文件：

- `train_01_04.jsonl`: 训练集，主要用于训练翻译模型。
- `val_01_04.jsonl`: 验证集，用于训练过程中的模型微调和评估。
- `test_01_04.jsonl`: 测试集，用于评估最终模型性能。

每个 JSON 对象包括：

- `inputs`: 原始古文
- `truth`: 准确翻译

例如：

```json
{"inputs": "昕曰： 回纥之功，唐已报之矣。", "truth": "萧昕反驳说： 回纥的功劳，唐朝已经报答了。"}
{"inputs": "然县令所犯在恩前，中人所犯在恩后。", "truth": "但是县令所犯罪过在施恩大赦之前，宦官所犯罪过在施恩赦免之后。"}
```
