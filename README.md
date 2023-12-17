# HistoryTrans: 文脉

## 项目简介

HistoryTrans 项目致力于创建和改进古文翻译的数据集。我们结合手动收集的数据和已有的公开数据集，通过严格的数据预处理和质量控制，确保数据集的高标准和实用性。本项目特别强调使用 chatglm3 对数据进行微调，从而提高翻译质量。

## 目录结构

```text
HistoryTrans
├── data
│   ├── merge.ipynb
│   ├── util
│   └── version3
├── eval
│   ├── data
│   └── eval_results
└── finetune
    ├── scripts
```

## 数据集 data

- **合并数据**：使用 `merge.ipynb` 合并不同来源的数据。
- **版本4**：`merged_output_20230812_190843.json`， **数据条目**: 977,172

## 评估 eval

- 在 `eval` 文件夹中，可以找到评估所用的数据 (`eval_data.json`) 和评估脚本 (`eval.ipynb`)。
- 评估结果存放在 `eval_results` 文件夹下，特别包括使用 chatglm3 微调的结果。

## 微调 finetune

- 本项目提供了完整的微调脚本，包括预处理工具和模型训练工具。
- 微调相关的脚本和指导可以在 `finetune` 文件夹中找到。

## 许可

本项目采用 [适当的开源许可]，详见 `LICENSE` 文件。
