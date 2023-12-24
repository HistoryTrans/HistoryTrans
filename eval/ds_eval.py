import argparse
import json
import os
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import deepspeed

# 解析命令行参数
parser = argparse.ArgumentParser(description='古文翻译模型评估脚本')
parser.add_argument('--model_path', type=str, required=True, help='模型路径')
parser.add_argument('--eval_file', type=str, required=True, help='评估数据文件路径')
parser.add_argument('--output_path', type=str, default='./eval_results/', help='评估结果输出路径')
parser.add_argument('--batch_size', type=int, default=60, help='批处理大小')
parser.add_argument('--mp_size', type=int, default=1, help='模型并行的GPU数量')
parser.add_argument('--max_length', type=int, default=1024, help='生成文本的最大长度')
parser.add_argument('--top_p', type=float, default=0.8, help='生成过程中的Top-p值')
parser.add_argument('--temperature', type=float, default=0.1, help='生成过程中的温度参数')
args = parser.parse_args()

print("命令行参数：", args)

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
print("模型加载完成。")

# 使用DeepSpeed进行推理加速
ds_model = deepspeed.init_inference(
    model,
    mp_size=args.mp_size,
    dtype=torch.half,
    replace_with_kernel_inject=True
)
print("DeepSpeed初始化完成。")

# 提示构造
PROMPT_DICT = {
    "prompt_input": (
        "下面是一段文言文文本，请直接将它翻译成白话文。\n"
        "#文言文文本:\n{input}\n"
        "#白话文文本:\n"
    )
}

# 生成批量文本输出
def generate_output_from_text_batch(text_list):
    batch_prompts = [PROMPT_DICT["prompt_input"].format(input=text) for text in text_list]
    inputs = tokenizer.batch_encode_plus(batch_prompts, return_tensors='pt', padding=True, max_length=512, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}  # 移动输入到GPU

    outputs = ds_model.module.generate(
        input_ids=inputs['input_ids'],
        max_length=args.max_length,
        top_p=args.top_p,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id
    )

    results = []
    for j, output in enumerate(outputs):
        decoded_output = tokenizer.decode(output, skip_special_tokens=True)
        translation = decoded_output.replace(batch_prompts[j], "").strip()
        results.append(translation.replace("[gMASK]sop ", ""))
    return results

# 计算BLEU分数
def calculate_bleu(reference, candidate):
    reference = [list(reference)]
    candidate = list(candidate)
    return sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)

# 秒数转换为小时、分钟和秒的格式
def seconds_to_hms(seconds):
    """将秒数转换为小时、分钟和秒的格式。"""
    parts = []
    hours = seconds // 3600
    if hours > 0:
        parts.append(f"{int(hours)}小时")
    minutes = (seconds % 3600) // 60
    if minutes > 0:
        parts.append(f"{int(minutes)}分钟")
    seconds = seconds % 60
    if seconds > 0 or len(parts) == 0:
        parts.append(f"{int(seconds)}秒")
    return ''.join(parts)

def evaluate_in_batches(data, batch_size, evaluation_results):
    total_samples = len(data)
    total_bleu_score = 0
    start_time = time.time()

    print(f"开始评估，共 {total_samples} 个样本.")

    for i in tqdm(range(0, total_samples, batch_size), desc="评估进度"):
        batch_data = data[i:i + batch_size]
        batch_inputs = [example["input"] for example in batch_data]
        batch_truths = [[example["output"]] for example in batch_data]
        batch_results = generate_output_from_text_batch(batch_inputs)

        for j, result_str in enumerate(batch_results):
            max_bleu_score = max(calculate_bleu(truth, result_str) for truth in batch_truths[j])
            total_bleu_score += max_bleu_score

            # 记录每个样本的评估结果
            evaluation_results["samples"].append({
                "input": batch_inputs[j],
                "output": batch_truths[j][0],
                "translation": result_str,
                "BLEU": max_bleu_score
            })

    # 计算平均BLEU分数和总评估时长
    average_bleu_score = total_bleu_score / total_samples
    total_time = time.time() - start_time
    evaluation_results["scores"]["average_BLEU"] = average_bleu_score
    evaluation_results["infos"]["evaluation_duration"] = seconds_to_hms(total_time)
    print(f"\nBatch={batch_size}")
    print(f"评估完成，平均BLEU分数为: {average_bleu_score:.3f}, 总耗时: {seconds_to_hms(total_time)}")

# 主函数
def main():
    # 先进行一次推理，避免第一次推理时耗时过长并测试模型是否能正常工作
    print("正在进行一次推理测试...")
    text_list = ["老吾老，以及人之老；幼吾幼，以及人之幼", "古之学者必有师。师者， 所以传道受业解惑也。"]
    results = generate_output_from_text_batch(text_list)
    print("原文:", text_list)
    print("结果:", results)   


    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # 读取评估数据
    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 进行评估
    evaluation_results = {
        "scores": {"average_BLEU": 0},
        "infos": {
            "evaluation_time": time.strftime("%Y/%m/%d %H:%M:%S", time.localtime()),
            "model_name_or_path": args.model_path,
            "eva_file_path": args.eval_file,
            "total_samples": len(data),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "top_p": args.top_p,
            "temperature": args.temperature
        },
        "samples": []
    }
    evaluate_in_batches(data, args.batch_size, evaluation_results)

    # 保存评估结果
    result_file = os.path.join(args.output_path, f"eval_resu_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

    print(f"评估结果已保存到 {result_file}")


if __name__ == "__main__":
    main()
