from typing import Dict, List
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from datasets import Dataset
import wandb
from modelscope import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["MODELSCOPE_CACHE"] = "/root/autodl-tmp"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def prepare_dataset(data_path: str, tokenizer) -> Dataset:
    """准备数据集,将输入输出拼接成单个序列并进行tokenize"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        full_text = f"Question: {item['question']}\nLet me think step by step:\n{item['thought_process']}\n\nAnswer: {item['answer']}"
        formatted_data.append({'text': full_text})
    
    # 创建基础数据集
    dataset = Dataset.from_list(formatted_data)
    
    # 定义tokenize函数
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding=False,
            return_special_tokens_mask=True
        )
    
    # 对数据集进行tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=['text'],
        batched=True
    )
    
    return tokenized_dataset

def train():
    wandb.init(project="24-game-solver")
    
    # 修改模型加载部分
    model_name = "Qwen/Qwen2.5-14B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        revision=None
    )
    
    # 确保设置特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        revision=None
    )
    
    # 确保模型参数可训练
    model.train()
    model.enable_input_require_grads()  # 启用输入梯度
    
    # 优化LoRA配置
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=256,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,  # 确保不是推理模式
    )
    model = get_peft_model(model, peft_config)
    
    # 确保所有需要训练的参数都设置正确
    for name, param in model.named_parameters():
        if 'lora' in name:  # 只训练 LoRA 参数
            param.requires_grad = True
            param.data = param.data.to(torch.float32)
        else:
            param.requires_grad = False
    
    # 打印可训练参数信息
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    # 准备数据集
    train_dataset = prepare_dataset("24_game_train.json", tokenizer)
    eval_dataset = prepare_dataset("24_game_test.json", tokenizer)
    
    # 修改训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_grad_norm=0.5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        report_to="wandb",
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        group_by_length=True,
        ddp_find_unused_parameters=False,
        torch_compile=False
    )
    
    # 使用DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("final_model")
    wandb.finish()

if __name__ == "__main__":
    train()