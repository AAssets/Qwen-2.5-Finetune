#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inference.py

功能：
1. 从 Qwen/Qwen2.5-0.5B 加载基座模型（可含bitsandbytes量化配置）。
2. 从 outputs/checkpoint-1/checkpoint-50 中加载 LoRA 微调权重。
3. 将 LoRA 权重合并到基座模型（merge_and_unload）。
4. 只打印『完整输出』，并通过后处理去除常见干扰词。
5. （可选）保存合并后模型到本地目录。
"""

import re
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

def clean_text(text: str) -> str:
    """
    简单后处理：去除常见的无用标记或词汇，如 system/user/assistant/acco/bigot。
    可根据实际需求再添加更多规则。
    """
    remove_words = [r'\bacco\b', r'\bbigot\b', r'\bbigotry\b']
    cleaned = text
    for w in remove_words:
        # 使用正则替换整个单词（不区分大小写）
        cleaned = re.sub(w, '', cleaned, flags=re.IGNORECASE)
    # 去掉多余空行或空格，可以根据需要再微调
    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)   # 连续空行压缩
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)      # 多余空格压缩
    return cleaned.strip()

def main():
    ############################################################################
    # 1. 配置路径
    ############################################################################
    lora_output_dir = "outputs/checkpoint-1/checkpoint-50"  # 最终 LoRA 权重所在目录
    base_model_id   = "Qwen/Qwen2.5-0.5B"                   # 与训练时一致
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############################################################################
    # 2. 如果使用 QLoRA量化，则配置 bitsandbytes
    ############################################################################
    # 如果你只想 float16 推理，可以注释掉下述 BitsAndBytesConfig 部分
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    ############################################################################
    # 3. 加载 LoRA 配置 & 基座模型
    ############################################################################
    print("Loading LoRA config from:", lora_output_dir)
    lora_config = PeftConfig.from_pretrained(lora_output_dir)

    print("Loading base model:", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quant_config,  # 若不想量化可删掉
    )

    # 加载分词器
    print("Loading tokenizer from:", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # 设置 tokenizer 的 pad_token = eos_token, 以便使用 attention_mask
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 用 PeftModel 包裹基座模型，加载 LoRA 权重
    print("Applying LoRA from:", lora_output_dir)
    model = PeftModel.from_pretrained(base_model, lora_output_dir)

    ############################################################################
    # 4. 合并 LoRA 权重到基座模型
    ############################################################################
    print("Merging LoRA into base model ...")
    model = model.merge_and_unload()
    model.eval().to(device)
    print("Model is ready on device:", device)

    ############################################################################
    # 5. (可选) 保存合并后模型
    ############################################################################
    merged_model_dir = "outputs/merged-Qwen2.5-0.5B"
    print(f"Saving merged model to: {merged_model_dir}")
    model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)

    ############################################################################
    # 6. 准备示例推理 - 中文和英文
    ############################################################################
    prompt_zh = "请问，感冒的常见症状有哪些？"
    messages_zh = [
        {"role": "system", "content": "你是一个智能医疗助理。只回答与感冒症状相关的信息，答案简洁明了。"},
        {"role": "user", "content": prompt_zh}
    ]

    prompt_en = "What are the common symptoms of a cold?"
    messages_en = [
        {"role": "system", "content": "You are a medical assistant. Only answer with cold symptoms. Keep it short."},
        {"role": "user", "content": prompt_en}
    ]

    # 针对Qwen：apply_chat_template可自动拼接对话格式
    text_zh = tokenizer.apply_chat_template(messages_zh, tokenize=False, add_generation_prompt=True)
    text_en = tokenizer.apply_chat_template(messages_en, tokenize=False, add_generation_prompt=True)

    # 编码 -> 含 attention_mask
    inputs_zh = tokenizer(
        [text_zh], return_tensors="pt",
        padding=True, truncation=True, max_length=1024
    ).to(device)

    inputs_en = tokenizer(
        [text_en], return_tensors="pt",
        padding=True, truncation=True, max_length=1024
    ).to(device)

    ############################################################################
    # 7. 生成参数 (尽量减少随意性的设置)
    ############################################################################
    gen_kwargs = {
        "do_sample": False,     # 关闭随机采样
        "num_beams": 4,         # 使用beam search
        "max_new_tokens": 128,
        "no_repeat_ngram_size": 4,
        "repetition_penalty": 1.5,
    }

    ############################################################################
    # 8. 推理 - 中文（只打印完整输出）
    ############################################################################
    print("\n===== [中文推理] =====")
    with torch.no_grad():
        outputs_zh = model.generate(
            input_ids=inputs_zh["input_ids"],
            attention_mask=inputs_zh["attention_mask"],
            **gen_kwargs
        )
    full_zh = tokenizer.decode(outputs_zh[0], skip_special_tokens=True)
    # 后处理：去除 system、assistant、acco 等无关词
    full_zh = clean_text(full_zh)
    print("【中文完整输出】：", full_zh)

    ############################################################################
    # 9. 推理 - 英文（只打印完整输出）
    ############################################################################
    print("\n===== [英文推理] =====")
    with torch.no_grad():
        outputs_en = model.generate(
            input_ids=inputs_en["input_ids"],
            attention_mask=inputs_en["attention_mask"],
            **gen_kwargs
        )
    full_en = tokenizer.decode(outputs_en[0], skip_special_tokens=True)
    full_en = clean_text(full_en)
    print("【英文完整输出】：", full_en)


if __name__ == "__main__":
    main()
