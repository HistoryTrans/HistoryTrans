# 古文翻译项目数据集 Version 4

`merged_output_20230812_190843.json`，此数据集主要用于古文翻译和研究

## 基本信息

- **文件名**: `merged_output_20230812_190843.json`
- **数据条目**: 977,172
- **版本**: Version 4
- **分布时间**: 2023-08-12 19:08:43

## 数据集说明

- **提取自**: 《二十四史》和《清史稿》+公开数据集 (NiuTrans/Classical-Modern)
- **格式**: 数据集以JSON格式组织，每个条目包含“input”(原文)和“output”(翻译文)字段，以及“terms”字段(保留字段，用于记录原文中的专有名词)
- `sampled_10000_merged_output_20230812_190843`文件是从`merged_output_20230812_190843`中随机抽取的10000条数据，用于测试

## 样例数据

```json
[
    {
        "input": "臣闻伐国事重，古人所难，功虽可立，必须经略而举。",
        "output": "倘若天降大雨，也许可以凭借水路交通，运输粮草增加兵力，为大举进攻作好计划。",
        "terms": []
    },
    {
        "input": "德钧益惭。",
        "output": "赵德钧更加羞惭。",
        "terms": []
    },
    {
        "input": "以山南东道节度使来瑱为兵部尚书，同中书门下平章事，节度如故。",
        "output": "以山南东道节度使来王员为兵部尚书、同中书门下平章事，节度使仍然像原来一样兼任。",
        "terms": []
    }
]
```