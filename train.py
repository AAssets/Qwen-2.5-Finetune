# 1. 加载必要的库
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

# 3. int4 量化配置 

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # 或者 load_in_8bit=True，根据需要设置
    llm_int8_enable_fp32_cpu_offload=True,
    bnb_4bit_compute_dtype=torch.float16, # 虽然我们以4位加载和存储模型，但我们在需要时会部分反量化他，并以16位精度进行计算
    bnb_4bit_quant_type="nf4", # nf量化类型
    bnb_4bit_use_double_quant=True, # 双重量化，量化一次后再量化，进一步解决显存
)

# 4. 加载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B"  # ModelScope 的模型 ID

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,  # 应用量化配置
    torch_dtype=torch.float16,  # 使用 float16 精度
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, padding_side="right",use_fast=False
)

# 5. 预处理数据集
def preprocess_function(examples):
    # 定义提示词模板
    PROMPT_DICT = {
        "prompt_no_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n""",
        "prompt_input": """<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n""",
    }

    # 将 instruction 和 input 拼接成 prompt
    prompts = []
    for instruction, input in zip(examples["instruction"], examples["input"]):
        if input:
            prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        prompts.append(prompt)

    # 将 prompt 和 output 拼接成一个字符串
    texts = []
    for prompt, output in zip(prompts, examples["output"]):
        text = prompt + output  # 将 prompt 和 output 拼接
        texts.append(text)

    # 使用 tokenizer 对 texts 进行分词
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt",  
    )

    # 只返回 input_ids 和 attention_mask
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
    }

dataset = dataset.map(preprocess_function, batched=True, num_proc=4, remove_columns=["instruction", "input", "output"],)

model = prepare_model_for_kbit_training(model)
# 6. 设置 LoRA 参数
lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 7. 设置训练参数
training_args = TrainingArguments(
    remove_unused_columns=False,
    output_dir="outputs/checkpoint-1",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    fp16=True,
    warmup_steps=10,
    max_steps=50,
    logging_steps=1,
    save_steps=10,
    optim="paged_adamw_8bit",  # 使用 Paged Optimizers
    evaluation_strategy="steps",  
    save_strategy="steps",  
    load_best_model_at_end=True,  # 训练结束后加载最佳模型
    gradient_checkpointing=True,  # 启用梯度检查点
    ddp_find_unused_parameters=False
)

swanlab_config = {
        "dataset": dataset,
        "peft":"Qlora"
    }
swanlab_callback = SwanLabCallback(
    project="finetune",
    experiment_name=model_name,
    description="中英文医疗数据集指令微调",
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
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[swanlab_callback],
)
model.config.use_cache = False
trainer.train()

# 9. 保存训练结果
trainer.save_model(trainer.args.output_dir)

# import torch
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 获取训练输出目录
# peft_model_dir = trainer.args.output_dir

# # 从训练模型目录加载配置
# config = PeftConfig.from_pretrained(peft_model_dir)

# # 加载基础模型
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
#     return_dict=True,
#     device_map=device,
#     torch_dtype=torch.float16,
#     quantization_config=quantization_config
# )

# # 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# # 加载 LoRA 模型
# model = PeftModel.from_pretrained(model, peft_model_dir)

# # 打印模型信息
# print(model)

# # 中文推理示例
# prompt_zh = "请问，感冒的常见症状有哪些？"
# messages_zh = [
#     {"role": "system", "content": "你是一个智能医疗助理."},
#     {"role": "user", "content": prompt_zh}
# ]

# # 英文推理示例
# prompt_en = "What are the common symptoms of a cold?"
# messages_en = [
#     {"role": "system", "content": "You are a medical assistant."},
#     {"role": "user", "content": prompt_en}
# ]

# # 应用聊天模板
# text_zh = tokenizer.apply_chat_template(messages_zh, tokenize=False, add_generation_prompt=True)
# text_en = tokenizer.apply_chat_template(messages_en, tokenize=False, add_generation_prompt=True)

# # 准备模型输入
# model_inputs_zh = tokenizer([text_zh], return_tensors="pt").to(model.device)
# model_inputs_en = tokenizer([text_en], return_tensors="pt").to(model.device)

# # 生成参数设置
# gen_kwargs = {"max_length": 512, "do_sample": True, "top_k": 1}

# # 中文推理
# with torch.no_grad():
#     outputs_zh = model.generate(**model_inputs_zh, **gen_kwargs)
#     outputs_zh = outputs_zh[:, model_inputs_zh['input_ids'].shape[1]:]
#     print("中文模型输出:", tokenizer.decode(outputs_zh[0], skip_special_tokens=True))

# # 英文推理
# with torch.no_grad():
#     outputs_en = model.generate(**model_inputs_en, **gen_kwargs)
#     outputs_en = outputs_en[:, model_inputs_en['input_ids'].shape[1]:]
#     print("英文模型输出:", tokenizer.decode(outputs_en[0], skip_special_tokens=True))