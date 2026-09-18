import sys
import yaml
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
import os
from CPVM.code.utils.arguments import *

def get_torch_dtype(dtype_str):
    """将字符串类型转换为 torch 数据类型"""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto"
    }
    return dtype_map.get(dtype_str, torch.bfloat16)

def merge_lora(
    base_model_path: str, 
    lora_path: str, 
    output_path: str, 
    device_map: str, 
    dtype: str, 
    safe_serialization: bool, 
    max_shard_size: str,
    **kwargs
    ):

    print(f"Loading base model from {base_model_path}...")
    torch_dtype = get_torch_dtype(dtype)
    # 1. 加载底座模型
    # 注意：即使你训练时用了量化(load_in_4bit)，合并时强烈建议用 float16 或 bfloat16 加载底座
    # 否则无法进行权重相加的数学操作
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype, # 或 torch.float16，取决于你的显卡支持
        device_map=device_map,          # 自动分配显存，或者设为 "cpu" 防止显存不足
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print(f"Loading LoRA adapter from {lora_path}...")

    # 2. 加载 LoRA Adapter
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights...")

    # 3. 核心步骤：合并并卸载 LoRA
    # 这会将 LoRA 的权重 W = W_base + B*A 计算进去，并把 LoRA 结构移除
    model = model.merge_and_unload()

    output_path = os.path.join(output_path, f"{lora_path.split('/')[-1]}-merged")

    print(f"Saving merged model to {output_path}...")

    # 4. 保存合并后的模型
    model.save_pretrained(
        output_path, 
        safe_serialization=safe_serialization, # 保存为 .safetensors 格式（推荐）
        max_shard_size=max_shard_size    # 切分权重文件大小
    )

    # 5. 保存 Tokenizer
    # 这一步很重要，否则你加载模型时没 tokenizer 用
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"Done! Merged model saved in {output_path}.")

def main():
    
    # 读取 YAML 配置
    cfg = parse_args_yaml('Merge LoRA weights into base model using YAML config.')

    merge_lora(**cfg)

if __name__ == "__main__":
    main()