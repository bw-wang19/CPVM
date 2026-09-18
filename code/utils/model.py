import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

def load_model_tokenizer(model_args, lora_args):
    # --- 3. 加载 Tokenizer 和 模型 ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
        use_fast=False # 避免某些 FastTokenizer 的 Rust 报错
    )
    # 处理 Pad Token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # 极端情况，添加一个新的 pad token
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    # 加载模型：BF16 原生加载，不量化
    # 这里的 device_map 不要设为 "auto"，DDP 环境下会自动分配到 local_rank
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        # use_cache=False 是训练必须的
        use_cache=False,
    )

    # 这里的 resize 是为了防止前面如果加了特殊的 pad token
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Choose fine-tuning strategy based on finetune_type
    finetune_type = getattr(model_args, 'finetune_type', 'lora')

    if finetune_type == "lora":
        # Apply LoRA adapter
        if lora_args:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_args.lora_rank,
                lora_alpha=lora_args.lora_alpha,
                lora_dropout=lora_args.lora_dropout,
                target_modules=lora_args.target_modules
            )
            model = get_peft_model(model, peft_config)
    elif finetune_type == "full":
        # Full fine-tuning: ensure all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported finetune_type: '{finetune_type}'. Must be 'lora' or 'full'.")

    return model, tokenizer