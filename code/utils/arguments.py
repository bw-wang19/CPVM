from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed
)
from dataclasses import dataclass, field
from typing import Optional, List
import argparse
import yaml
import os


# 我们把参数分为三类：模型参数、数据参数、LoRA参数
# TrainingArguments 直接使用库自带的

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "模型路径或HuggingFace ID"})
    finetune_type: str = field(default="lora", metadata={"help": "Fine-tuning type: 'lora' or 'full'"})
    resume_wandb_run_id: Optional[str] = field(default=None, metadata={"help": "WandB run ID for resuming training"})
@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "训练数据路径 (json)"})
    max_length: int = field(default=1024, metadata={"help": "最大序列长度"})
    val_ratio: float = field(default=0.1, metadata={"help": "验证集比例"})
    filter_refusals: bool = field(
        default=True,
        metadata={"help": "Remove known dataset-generation refusal templates."},
    )
    preview_samples: int = field(
        default=2,
        metadata={"help": "Number of formatted preference pairs to preview."},
    )

@dataclass
class DPODataArguments:
    data_path: str
    val_ratio: float = 0.1
    filter_refusals: bool = True
    preview_samples: int = 2

@dataclass
class LoraArguments:
    lora_rank: int = field(default=8, metadata={"help": "LoRA Rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA Alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA Dropout"})
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"], metadata={"help": "LoRA 作用模块"})

@dataclass
class Big5ChatArguments:
    trait: str = field(default=None, metadata={"help": "人格特质, 例如:\n 'conscientiousness',\n 'openness',\n 'extraversion',\n 'agreeableness',\n 'neuroticism'"})
    level: str = field(default=None, metadata={"help": "人格特质水平: 'low' or 'high'"})
    use_original_prompt: bool = field(default=False, metadata={"help": "是否使用原始的 prompt 模板"})
    dialogue_prompt: str = field(default=None, metadata={"help": "对话 prompt 模板"})

def parse_args_yaml(desc:str=None):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the yaml configuration file."
    )

    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}: {config}")
    print(f"Config: {config}")
    return config

