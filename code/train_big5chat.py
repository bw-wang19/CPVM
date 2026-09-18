import wandb
import os
import sys
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import logging as logger
import torch
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset

from CPVM.code.utils.arguments import *
from CPVM.code.utils.process import *
from CPVM.code.utils.model import *


def main():
    # --- 2. 解析参数 ---
    # HfArgumentParser 可以同时读取命令行参数和 yaml/json 文件
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments, Big5ChatArguments))
    
    # 如果命令行传入了 .yaml/.json 文件，直接解析
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        model_args, data_args, lora_args, training_args, big5chat_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则尝试解析命令行参数
        model_args, data_args, lora_args, training_args, big5chat_args = parser.parse_args_into_dataclasses()

    # 设置日志
    logger.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logger.StreamHandler(sys.stdout)],
        level=logger.INFO if training_args.local_rank in [-1, 0] else logger.WARN,
    )
    
    # 设置随机种子
    set_seed(training_args.seed)
    split_seed = (
        training_args.data_seed
        if training_args.data_seed is not None
        else training_args.seed
    )
    
    # 加载模型
    logger.info(f"Loading model from {model_args.model_name_or_path}...")
    model, tokenizer = load_model_tokenizer(model_args, lora_args)
    model.train()
    
    if training_args.local_rank == 0:
        # Print trainable parameters info
        if model_args.finetune_type == "lora":
            model.print_trainable_parameters()
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Full fine-tuning: trainable params: {trainable_params:,d} || all params: {total_params:,d} || trainable%: {100 * trainable_params / total_params:.2f}%")

    print("Loading dataset...")
    # --- 5. 数据处理 ---
    dataset = Big5Chat(data_args.data_path)

    
    # 人格特质筛选条件
    
    if big5chat_args.trait is None or big5chat_args.level is None:
        raise ValueError("Both 'trait' and 'level' must be specified in Big5ChatArguments.")
    else:
        trait_filter = {
            'trait': big5chat_args.trait,
            'level': big5chat_args.level     
        }
        print(f"Filtering dataset for trait: {trait_filter}")
    
    # 多进程处理数据
    with training_args.main_process_first(desc="Processing dataset"):
        
        # 1. 对原始数据按照original_index进行划分，保证训练集和验证集的分布一致
        dataset_split = Big5Chat.split_by_original_index(
            dataset,
            val_size=data_args.val_ratio,
            seed=split_seed,
        )
        
        # 2. train/val 使用相同的 trait-level 过滤规则
        dataset_filtered = dataset_split.filter(
            Big5Chat.filter_func,
            num_proc=32,
            fn_kwargs={
                'trait_filter': trait_filter, 
                'big5chat_args': big5chat_args,
            }
                       
        )
        
        # 检查拒绝输出
        dataset_filtered = Big5Chat.refusal_filter(
            dataset_filtered,
            data_args.filter_refusals
        )
        
        # 3. 数据tokenization
        tokenized_dataset_split  = dataset_filtered.map(
            Big5Chat.sft_process_func, 
            num_proc=32, 
            fn_kwargs={
                'tokenizer': tokenizer, 
                'max_length': data_args.max_length,
                'big5chat_args': big5chat_args,
            }      
        )

    print(
        '''
        --------------
        Dataset Loaded
        --------------
        '''
    )

    
    # --- 6. 初始化 Trainer ---

    print('Trainer initializing...')
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ft_tag = model_args.finetune_type  # "lora" or "full"
    run_name = f"{model_args.model_name_or_path.split('/')[-1]}-sft-{ft_tag}-big5chat-{trait_filter['trait']}-{trait_filter['level']}-ep{training_args.num_train_epochs}"
    training_args.run_name = run_name
    output_subdir = "adapters" if model_args.finetune_type == "lora" else "full"
    training_args.output_dir = f"{training_args.output_dir}/{output_subdir}/{run_name}"
    print(f"Output directory: {training_args.output_dir}")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_split['train'],
        eval_dataset=tokenized_dataset_split['val'],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,           # 传入 model 是为了自动获取 label_pad_token_id (默认-100)
            padding=True,          # 开启动态补齐，按照 Batch 里最长的补
            pad_to_multiple_of=8   # (可选) 补齐到 8 的倍数，对 NVIDIA 显卡计算更友好
            )
    )
    print('''
          -------------------
          Trainer initialized
          -------------------
          '''
        )
    # --- 7. 开始训练 ---
    print('''
          -------------------
          Training starting...
          ------------------- 
          '''
        )
    
    if training_args.local_rank == 0:
        print("Starting training...")
        
    trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    print('''
          -------------------
          Training finished
          ------------------- 
          '''
        )
    # --- 8. 保存 ---
    print('Saving model...')
    trainer.save_model() # LoRA mode saves adapter only; full mode saves the entire model
    if training_args.local_rank == 0:
        print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()