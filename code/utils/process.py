from datasets import Dataset, load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
    set_seed
)
from CPVM.code.utils.arguments import *
import random
from typing import Any, Dict
import re

REFUSAL_PATTERN = re.compile(
    r"^(?:"
    r"i cannot\s+(?:generate|create|fulfill|provide|assist|help|comply|write|produce)"
    r"|i can't\s+(?:generate|create|fulfill|provide|assist|help|comply|write|produce)"
    r"|i am unable to\s+(?:generate|create|fulfill|provide|assist|help|comply|write|produce)"
    r"|as an ai(?:\s+language model)?\b"
    r")",
    flags=re.IGNORECASE,
)


def nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()

def is_refusal(value: Any) -> bool:
    text = nonempty_text(value)
    return bool(text and REFUSAL_PATTERN.match(text))


class Big5Chat(DatasetDict):
    def __init__(self, path:str='/mnt/nvme2/models/amadeus/dataset_zoo/big5_chat', data: DatasetDict = None, **kwargs) -> DatasetDict:
        
        if data is not None and isinstance(data, DatasetDict):
            super().__init__(data)
            return
        
        try: 
            data = load_dataset(path)
        except:
            print('load from remote repository')
            data = load_dataset('wenkai-li/big5_chat')
        super().__init__(data)


    @classmethod
    def refusal_filter(
        cls,
        dataset: Dataset,
        filter_refusals: bool = True,
    ):
        '''
        Filter out rows that contain known refusal templates in either the prompt or output.
        '''
        
        if not filter_refusals:
            return dataset

        def is_refusal_row(row: dict[str, Any]) -> bool:
            prompt_text = nonempty_text(row.get("train_input"))
            output_text = nonempty_text(row.get("train_output"))
            return is_refusal(prompt_text) or is_refusal(output_text)

        return dataset.filter(lambda row: not is_refusal_row(row), num_proc=32)

    @classmethod
    def sft_process_func(cls, example, **kwargs):
        """
        标准 SFT 数据处理：
        1. 使用 apply_chat_template 拼接文本
        2. Tokenize
        3. 构造 Labels (将 User 部分设为 -100)
        """
        tokenizer = kwargs['tokenizer']
        max_length = kwargs['max_length']
        big5chat_args = kwargs['big5chat_args']
        
        sys_prompt = example["train_instruction"] if big5chat_args.use_original_prompt else big5chat_args.dialogue_prompt
        instruction = example['train_input']
        output = example["train_output"]           

        # 构造对话列表 (Standard Chat Format)

        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": instruction}
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True  ## add_generation_prompt=true会在末尾自动加上<|imstart|>assistant\n
        )
        
        prompt_ids = tokenizer(
            prompt_text, 
            max_length=max_length, 
            truncation=True, 
            add_special_tokens=False,
            return_attention_mask=False
            )["input_ids"]
        output_ids = tokenizer(
            output, 
            max_length=max_length, 
            truncation=True, 
            add_special_tokens=False,
            return_attention_mask=False
            )["input_ids"]
        # 2. Tokenize 完整文本
        input_ids = prompt_ids + output_ids + [tokenizer.eos_token_id]
        # 截断 (Truncation)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
                
        if len(labels) != len(input_ids):
            # 如果截断发生在 prompt 内部，整个样本其实就是无效的（没有回答），但为了代码鲁棒：
             labels = [-100] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids) # 显式返回 mask，虽然 collator 会补，但这样更安全
        }


    @classmethod
    def filter_func(cls, example, **kwargs):
        for key, value in kwargs['trait_filter'].items():
            if example[key].lower() != value.lower():
                return False

        sys_prompt = example.get("train_instruction")
        instruction = example.get("train_input")
        output = example.get("train_output")

        # 必须都不为 None，且转为字符串后不为空
        if not (instruction and output):
            return False
        
        if kwargs['big5chat_args'].use_original_prompt and sys_prompt is None:
            return False
        
        # 确保是字符串类型（防止 excel 里的数字进来）
        if not isinstance(output, str):
            return False
        return True

    @classmethod
    def split(cls, datasetdict, val_size) -> DatasetDict:
        dataset_train = datasetdict['train']
        dataset_split = dataset_train.train_test_split(test_size=val_size)
        return DatasetDict({
            'train': dataset_split['train'],
            'val': dataset_split['test']
        })
        
    @classmethod
    def split_by_original_index(
        cls,
        datasetdict,
        val_size: float,
        seed: int,
        data_args: DataArguments = None,
    ) -> DatasetDict:
        """
        严格按照 original_index 划分。

        同一个 original_index 下的所有 trait/level 行，
        必须全部进入 train 或全部进入 val。
        """
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")

        dataset = datasetdict["train"]

        if "original_index" not in dataset.column_names:
            raise ValueError(
                "Dataset does not contain required column: original_index"
            )

        # 转成字符串，避免 numpy.int64 / int / JSON 类型差异
        all_original_ids = sorted({
            str(original_id)
            for original_id in dataset["original_index"]
        })

        if len(all_original_ids) < 2:
            raise ValueError("Not enough unique original_index values to split")

        # 使用独立 RNG，不依赖进程的全局随机状态
        shuffled_ids = all_original_ids.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled_ids)

        n_val = int(round(len(shuffled_ids) * val_size))
        n_val = max(1, min(n_val, len(shuffled_ids) - 1))

        val_original_ids = set(shuffled_ids[:n_val])
        train_original_ids = set(shuffled_ids[n_val:])

        # 基本完整性检查
        assert train_original_ids.isdisjoint(val_original_ids)
        assert len(train_original_ids) + len(val_original_ids) == len(
            all_original_ids
        )

        train_positions = []
        val_positions = []

        for row_index, original_id in enumerate(dataset["original_index"]):
            original_id = str(original_id)

            if original_id in val_original_ids:
                val_positions.append(row_index)
            elif original_id in train_original_ids:
                train_positions.append(row_index)
            else:
                raise RuntimeError(
                    f"original_index {original_id} is missing from split"
                )

        result = DatasetDict({
            "train": dataset.select(train_positions),
            "val": dataset.select(val_positions),
        })

        print(
            f"Split by original_index: "
            f"seed={seed}, "
            f"unique_train_ids={len(train_original_ids)}, "
            f"unique_val_ids={len(val_original_ids)}, "
            f"train_rows={len(result['train'])}, "
            f"val_rows={len(result['val'])}"
        )

        return result
    
    @classmethod
    def build_dpo_dataset(
        cls,
        dataset: Dataset,
        chosen_level: str,
        big5chat_args: Big5ChatArguments,
        filter_refusals: bool = True,
    ) -> Dataset:
        """
        Build TRL's explicit-prompt conversational preference format.
        """
        def index_trait_rows(
            dataset: Dataset,
        ) -> dict[str, dict[str, dict[str, Any]]]:
            """Index one trait as ``original_index -> level -> row``."""
            grouped: dict[str, dict[str, dict[str, Any]]] = {}
            for row in dataset:
                original_id = str(row["original_index"])
                level = row.get("level")
                trait = row.get("trait")
                levels = grouped.setdefault(original_id, {})
                if level in levels:
                    raise ValueError(
                        f"Duplicate ({original_id}, {trait}, {level}) row in BIG5-CHAT."
                    )
                levels[level] = row

            return grouped
        
        grouped_rows = index_trait_rows(dataset)
        
        records: list[dict[str, Any]] = []
        opposite_level = "low" if chosen_level == "high" else "high"
        
        for original_index in grouped_rows.keys():
            levels = grouped_rows.get(original_index)
            try:
                high_row= levels["high"]
                low_row = levels["low"]
                prompt_text = nonempty_text(high_row.get("train_input"))
            except KeyError:
                continue
            high_output = nonempty_text(high_row.get("train_output"))
            low_output = nonempty_text(low_row.get("train_output"))
            
            # 检查拒绝输出
            
            if filter_refusals and (
                is_refusal(prompt_text)
                or is_refusal(high_output)
                or is_refusal(low_output)
            ):
                continue
            
            chosen_row = levels[chosen_level]
            opposite_row = levels[opposite_level]
            if big5chat_args.use_original_prompt:
                # Both completions deliberately receive the target-level instruction,
                # so the DPO prompt is still identical. This reproduces the
                # prompt-conditioned setup, but is not suitable for testing whether
                # the trait has been internalized without an explicit personality cue.
                system_prompt = nonempty_text(chosen_row.get("train_instruction"))
            else:
                system_prompt = nonempty_text(big5chat_args.dialogue_prompt)
            
            records.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text},
                    ],
                    "chosen": [
                        {"role": "assistant", "content": chosen_row['train_output']},
                    ],
                    "rejected": [
                        {"role": "assistant", "content": opposite_row['train_output']},
                    ],
                }
            )
        return Dataset.from_list(records)
            