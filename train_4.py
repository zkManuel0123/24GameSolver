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
    
    # 优化LoRA配置
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,  # 增加LoRA秩以提升模型容量
        lora_alpha=128,  # 相应调整alpha
        lora_dropout=0.05,  # 降低dropout提高训练稳定性
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # 增加训练参数
    )
    model = get_peft_model(model, peft_config)
    
    # 准备数据集
    train_dataset = prepare_dataset("24_game_train.json", tokenizer)
    eval_dataset = prepare_dataset("24_game_test.json", tokenizer)
    
    # 优化训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=15,  # 增加训练轮次
        per_device_train_batch_size=32,  # 利用大显存提升批次大小
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # 由于批次变大，减少梯度累积
        learning_rate=2e-4,  # 略微提高学习率
        max_grad_norm=1.0,
        weight_decay=0.05,  # 适当减小权重衰减
        warmup_ratio=0.1,  # 减少预热比例加快训练
        logging_dir='./logs',
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=50,  # 更频繁的评估
        save_strategy="steps",
        save_steps=50,
        report_to="wandb",
        remove_unused_columns=False,
        lr_scheduler_type="cosine_with_restarts",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,  # 启用混合精度训练
        gradient_checkpointing=True,  # 启用梯度检查点以节省显存
        dataloader_num_workers=4  # 增加数据加载的工作进程
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