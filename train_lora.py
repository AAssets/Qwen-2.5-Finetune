# 1. 加载必要的库
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
# 去掉 BitsAndBytesConfig 的 import
# from modelscope import AutoModelForCausalLM, AutoTokenizer  # 如果还需要可保留
from peft import LoraConfig, get_peft_model  #, prepare_model_for_kbit_training
from swanlab.integration.transformers import SwanLabCallback

# 2. 加载数据集
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

dataset = load_dataset('json', data_files=data_files)

# 3. **去掉 int4量化配置** (QLoRA)，保持普通FP16
# （原先的 BitsAndBytesConfig 相关内容不再需要）
# quantization_config = BitsAndBytesConfig(...)

# 4. 加载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # 去掉: quantization_config=quantization_config
    torch_dtype=torch.float16,  # 使用FP16精度
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, padding_side="right",use_fast=False
)

# 5. 预处理数据集
def preprocess_function(examples, tokenizer, max_seq_length=256):
    input_ids, attention_mask, labels = [], [], []
    
    for instruction, human_input, assistant_output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        # 构建输入文本
        if human_input:
            input_text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{human_input}<|im_end|>\n<|im_start|>assistant\n"
        else:
            input_text = f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
        
        # 对输入和输出分别进行分词
        input_tokenizer = tokenizer(
            input_text,
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        output_tokenizer = tokenizer(
            assistant_output,
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        
        # 合并 input 和 output
        input_ids_sample = (
            input_tokenizer["input_ids"] 
            + output_tokenizer["input_ids"] 
            + [tokenizer.eos_token_id]
        )
        attention_mask_sample = (
            input_tokenizer["attention_mask"] 
            + output_tokenizer["attention_mask"] 
            + [1]
        )
        labels_sample = (
            [-100] * len(input_tokenizer["input_ids"]) 
            + output_tokenizer["input_ids"] 
            + [tokenizer.eos_token_id]
        )
        
        # 截断至 max_seq_length
        input_ids.append(input_ids_sample[:max_seq_length])
        attention_mask.append(attention_mask_sample[:max_seq_length])
        labels.append(labels_sample[:max_seq_length])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 应用 preprocess_function
dataset = dataset.map(
    preprocess_function,
    fn_kwargs={"tokenizer": tokenizer, "max_seq_length": 256},
    batched=True,
    num_proc=4,
    remove_columns=["instruction", "input", "output"]
)

# **去掉** prepare_model_for_kbit_training(model) (原QLoRA专用)
# model = prepare_model_for_kbit_training(model)

# 6. 设置 LoRA 参数 (保留不变)
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. 设置训练参数
training_args = TrainingArguments(
    remove_unused_columns=False,
    output_dir="outputs/checkpoint-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,  # 保持FP16
    warmup_steps=10,
    max_steps=50,
    logging_steps=1,
    save_steps=10,
    optim="adamw_hf",   # 这里改成普通 "adamw_hf" 或 "adamw_torch", 不再使用 "paged_adamw_8bit"
    evaluation_strategy="steps",  
    save_strategy="steps",  
    load_best_model_at_end=True, 
    gradient_checkpointing=True,  
    ddp_find_unused_parameters=False
)

swanlab_config = {
    "dataset": dataset,
    "peft": "Lora"  
}
swanlab_callback = SwanLabCallback(
    project="finetune",
    experiment_name=model_name,
    description="中英文医疗数据集指令微调 - LoRA(无量化)",
    workspace=None,
    config=swanlab_config,
)

# 8. 开始训练
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    tokenizer=tokenizer,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="max_length", return_tensors="pt", label_pad_token_id=-100),
    callbacks=[swanlab_callback],
)
model.config.use_cache = False
trainer.train()

# 9. 保存训练结果
trainer.save_model(trainer.args.output_dir)