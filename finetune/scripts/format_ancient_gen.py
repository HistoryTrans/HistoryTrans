#! /usr/bin/env python

import json
from collections import Counter
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--path", type=str, required=True)

args = parser.parse_args()

with open(args.path) as f:
    # data = [json.loads(line) for line in f]
    # 解析json的列表
    data = json.load(f)

train_examples = [{
    "prompt": x['input'],
    "response": x['output'],
} for x in data]

os.makedirs("formatted_data", exist_ok=True)

# 获取原来不带后缀的文件名
filename = os.path.splitext(os.path.basename(args.path))[0]
with open(f"formatted_data/{filename}.jsonl", "w") as f:
    for e in train_examples:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
