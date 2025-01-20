from typing import Dict, List
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig
from datasets import Dataset
import wandb
from modelscope import AutoTokenizer, AutoModelForCausalLM

def prepare_dataset(data_path: str, tokenizer) -> Dataset:
    """准备数据集,将输入输出拼接成单个序列并进行tokenize"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        full_text = f"Question: {item['question']}\nLet me think step by step:\n{item['long_cot']}\n\nAnswer: {item['answer']}"
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
    model_name = "qwen/Qwen2.5-0.5B"
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
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    # 准备数据集
    train_dataset = prepare_dataset("24_game_train.json", tokenizer)
    eval_dataset = prepare_dataset("24_game_test.json", tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.1,
        warmup_ratio=0.2,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",
        remove_unused_columns=False,
        lr_scheduler_type="cosine_with_restarts",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # 使用DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因为是因果语言模型,所以不使用MLM
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