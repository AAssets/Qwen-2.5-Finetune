import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 🛠️ **1. 加载数据集**
def load_sample_data():
    """
    加载数据集并提取 1-5 条样本。
    """
    data_files = {
        "train": [
            "medical/finetune/train_zh_0.json", 
            "medical/finetune/train_en_1.json"
        ], 
        "validation": [
            "medical/finetune/valid_zh_0.json", 
            "medical/finetune/valid_en_1.json"
        ],
        "test": [
            "medical/finetune/test_zh_0.json", 
            "medical/finetune/test_en_1.json"
        ]
    }
    
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    # 提取前5条样本
    sample_data = dataset.select(range(1))
    print("✅ 已成功加载 1 条样本数据。")
    
    return sample_data


# 🛠️ **2. 加载模型和分词器**
def load_model_and_tokenizer():
    """
    加载模型和分词器。
    """
    model_name = "Qwen/Qwen2.5-0.5B"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用 float16 精度
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    
    print("✅ 模型和分词器已成功加载。")
    return model, tokenizer


# 🛠️ **3. 数据预处理函数**
def preprocess_function(examples, tokenizer, max_seq_length=1024):
    """
    优化版数据预处理函数（修复重复 token 和遮蔽问题）
    """
    PROMPT_DICT = {
        "prompt_no_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n""",
        "prompt_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n""",
    }

    prompts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        
        if input_text and input_text.strip():
            prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input_text)
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        
        prompts.append(prompt)

    texts = [prompt + examples["output"][i] for i, prompt in enumerate(prompts)]

    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False  # 避免重复添加特殊标记
    )

    # 处理 labels
    labels = tokenized_inputs["input_ids"].clone()
    for i, prompt in enumerate(prompts):
        prompt_length = len(tokenizer(prompt, truncation=True, max_length=max_seq_length)["input_ids"])
        labels[i][:prompt_length] = -100  # 遮蔽输入部分
        labels[i][tokenized_inputs["attention_mask"][i] == 0] = -100  # 遮蔽 padding 部分

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels
    }



# 🚀 **4. 测试主流程**
if __name__ == "__main__":
    print("📊 **开始数据预处理测试**\n")
    
    # 1. 加载示例数据
    sample_data = load_sample_data()
    
    # 2. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 3. 预处理示例数据
    processed_data = preprocess_function(sample_data, tokenizer, max_seq_length=512)
    
    # 4. 打印处理结果
    for i in range(len(sample_data["instruction"])):
        print(f"\n📝 **示例 {i+1}:**")
        print(f"Instruction: {sample_data['instruction'][i]}")
        print(f"Input: {sample_data['input'][i]}")
        print(f"Output: {sample_data['output'][i]}")
        
        print("\n🔗 **Processed Text:**")
        decoded_input_ids = tokenizer.decode(processed_data["input_ids"][i], skip_special_tokens=False)
        print(decoded_input_ids)
        
        print("\n🛡️ **Attention Mask:**")
        print(processed_data["attention_mask"][i])
        
        print("\n🏷️ **Labels:**")
        print(processed_data["labels"][i])
    
    print("\n✅ **数据预处理测试完成！**")
