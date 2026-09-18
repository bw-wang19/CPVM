import json
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List, Dict
import os
import argparse
from CPVM.code.utils.arguments import *


class BFIJsonTester:
    def __init__(
        self, 
        model_path: str, 
        json_path: str = '/mnt/nvme2/models/amadeus/dataset_zoo/BFI', 
        use_vllm: bool = False,
        reverse: bool = False,
        few_shot: bool = True,
        gpu_memory_utilization: float = 0.9, 
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        system_prompt: str = None,
        trait_prompt: str = None,
        question_version: int = 1,
        adapter_path: str = None,
        **kwargs
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path or None
        self.use_vllm = use_vllm
        self.json_path = json_path
        self.temperature = temperature
        self.reverse = reverse
        self.few_shot = few_shot
        self.system_prompt = system_prompt
        self.trait_prompt = trait_prompt
        self.question_version = question_version
        self.cfg = {
            "model_path": model_path,
            "adapter_path": self.adapter_path,
            "json_path": json_path,
            "use_vllm": use_vllm,
            "reverse": reverse,
            "few_shot": few_shot,
            "gpu_memory_utilization": gpu_memory_utilization,
            "temperature": temperature,
            "tensor_parallel_size": tensor_parallel_size,
            "system_prompt": system_prompt,
            "trait_prompt": trait_prompt,
            "question_version": question_version
        }
        self.cfg = {**self.cfg, **kwargs}
        print(f"Loaded config: {self.cfg}")
        # Load Tokenizer
        print(f"Loading Tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.lora_request = None

        if self.use_vllm:
            from vllm import LLM, SamplingParams

            # vLLM cannot consume an in-memory merged Transformers model, so it
            # applies the same adapter natively without writing a merged model.
            print(f"🚀 Initializing vLLM Engine from {model_path}(TP:{tensor_parallel_size})...")
            llm_kwargs = {
                "model": model_path,
                "trust_remote_code": True,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            }
            if self.adapter_path:
                llm_kwargs["enable_lora"] = True

            self.llm = LLM(**llm_kwargs)

            if self.adapter_path:
                from vllm.lora.request import LoRARequest

                print(f"Loading LoRA adapter from {self.adapter_path} with vLLM...")
                self.lora_request = LoRARequest(
                    "test_adapter",
                    1,
                    self.adapter_path,
                )

            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=4096,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
        else:
            print("🐢 Initializing HuggingFace Transformers (Native Mode)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
            )
            if self.adapter_path:
                from peft import PeftModel

                print(f"Merging LoRA adapter from {self.adapter_path}...")
                model = PeftModel.from_pretrained(model, self.adapter_path)
                model = model.merge_and_unload()
            self.model = model.eval()

        self.questions = self.load_questions()
        print(f"Loaded {len(self.questions)} questions from {self.json_path}")

    def load_questions(self):
        bfi_path = self.json_path
        if self.question_version == 1:
            # Load BFI-version1 with 44 questions
            bfi_path = os.path.join(bfi_path, "bfi_44.json")
        elif self.question_version == 2:
            # Load BFI-version2 with 60 questions
            bfi_path = os.path.join(bfi_path, "bfi_2.json")
        else:
            raise ValueError(f"Invalid question_version: {self.question_version}. Must be 1 or 2.")
        with open(bfi_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def build_prompt(self, questions:List[Dict]):
        """
        构造 Few-Shot Prompt, 强制模型学会 JSON 格式
        """
        
        # add system prompt and trait prompt
        if self.system_prompt is None:
            self.system_prompt = ""
        if self.trait_prompt is None:
            self.trait_prompt = ""
        system_prompt = self.trait_prompt + self.system_prompt
        
        # add format prompt
        format_prompt = (
            "Please output the result in JSON format: [{'id': <id>, 'score': <score>}, ...], "
            "id indicates the question ID, score indicates the score, which should be an integer between 1 and 5.\n"
        )
        # Few-Shot Examples (少样本示例)
        # 用一些无关痛痒的例子教会模型格式
        examples = [
            {
                "role": "user", 
                "content": "[{'id': 998, 'text': 'I see Myself as Someone Who Likes Apples'}, {'id': 999, 'text': 'I see Myself as Someone Who Hates Raining'}]"
            },
            {
                "role": "assistant", 
                "content": "[{'id': 998, 'score': 5}, {'id': 999, 'score': 2}]"
            }
        ]

        # 当前题目
        content_list = []
        for question in questions:
            content_list.append(
                {
                    "id": question['id'],
                    "text": f"I see Myself as Someone Who {question['text_en'] if not self.reverse else question['reverse_text_en']}"

                }
            )
        current_query = {
            "role": "user",
            "content": json.dumps(content_list)
        }
        if self.few_shot:
            all_prompt = f"{system_prompt}<FORMAT>\n{format_prompt}<EXAMPLES>\n{json.dumps(examples)}"
        else:
            all_prompt = f"{system_prompt}<FORMAT>\n{format_prompt}"
        messages = [{"role": "system", "content": all_prompt}] + [current_query]
        self.all_prompt = all_prompt

        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, # will be tokenized later
            add_generation_prompt=True
        )

    def parse_json_response(self, response_text):
        """
        重写后的正则提取：针对 List[Dict] 格式
        """
        text = response_text.strip()
        
        # 1. 移除可能的 Markdown 代码块标记 (```json ... ```)
        match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1)
            
        # 2. 核心正则：提取最外层的方括号 [...] 及其内容
        # re.DOTALL 允许 . 匹配换行符
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        
        if match:
            json_str = match.group()
            try:
                # 尝试标准 JSON 解析
                return json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    # 3. 容错处理：如果 JSON 解析失败（通常是因为单引号），尝试 ast.literal_eval
                    # LLM 经常输出 {'id': 1} 而不是 {"id": 1}，json.loads 会挂，但 literal_eval 能过
                    return ast.literal_eval(json_str)
                except:
                    # 4. 暴力替换单引号（兜底方案）
                    try:
                        fixed_str = json_str.replace("'", '"')
                        return json.loads(fixed_str)
                    except:
                        pass
        
        print(f"⚠️ Regex failed to extract JSON list from: {response_text[:100]}...")
        return []

    # inference with vllm
    def inference_vllm(self, prompts: List[str]) -> List[str]:
        print(f"Running vLLM inference on {len(prompts)} items...")
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            lora_request=self.lora_request,
        )
        return [output.outputs[0].text for output in outputs]
    
    # inference with transformers
    def inference_hf(self, prompts: List[str]) -> List[str]:
        """Transformers 推理后端"""
        print(f"Running HF inference on {len(prompts)} items...")
        results = []
        for prompt in tqdm(prompts, desc="Inference"):
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=self.temperature,
                    do_sample=False if self.temperature == 0 else True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            # 解码
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(response)
        return results

    def run_test(self):
        question_ids = self.cfg['question_ids']
        all_in_one = self.cfg['all_in_one']
        if question_ids:
            questions = [item for item in self.questions if item['id'] in question_ids]
        else:
            questions = self.questions
        
        print(f"Preparing to test {len(questions)} questions...")

        if all_in_one:
            prompts = [self.build_prompt(questions)]
        else:
            prompts = [self.build_prompt([item]) for item in questions]

        # 动态选择后端
        if self.use_vllm:
            output_texts = self.inference_vllm(prompts)
        else:
            output_texts = self.inference_hf(prompts)

        scores, outputs = self.process_output(output_texts)
        self.print_report(results=scores, outputs=outputs)
    
    def process_output(self, outputs:List[str]):
        results = {}
        question_map = {q['id']: q for q in self.questions}
        for output in outputs:   
            # 解析 JSON
            data = self.parse_json_response(output)
            if not isinstance(data, list):
                continue
            for item in data:
                if item and "score" in item and "id" in item:
                    # 限制范围 (防止幻觉出 6 或 0)
                    raw_score = max(1, min(5, int(item["score"])))
                    try:
                        q = question_map[item["id"]]
                    except:
                        print(f'item index out of range: {item}')
                        continue
                    # 反向计分处理
                    final_score = raw_score if q['scoring'] == 'positive' else (6 - raw_score)
                    if q['trait'] not in results:
                        results[q['trait']] = []
                    results[q['trait']].append(final_score)
                else:
                    print(f"⚠️ Format Error (Item: {item})")

        return results, outputs

    def print_report(self, results, outputs):
        print("\n=== BFI REPORT ===")
        print(f"{'Trait':<20} | {'Score':<5}")
        print("-" * 30)
        for trait, scores in results.items():
            if scores:
                avg = np.mean(scores)
                print(f"{trait:<20} | {avg:.2f}")
            else:
                print(f"{trait:<20} | N/A")
        save_path = self.cfg['output_path']
        if save_path:
            scores = {k: np.mean(v) if v else 0 for k, v in results.items()}
            summary = {
                "scores": scores, 
                "results": results,
                "outputs": outputs,
                }
            summary = {**summary, **self.cfg}
            model_source = Path(self.adapter_path or self.model_path)
            model_name = model_source.name
            if model_name.startswith("checkpoint-"):
                model_name = f"{model_source.parent.name}-{model_name}"
            save_path = os.path.join(save_path, f"bfi-v{self.question_version}-{model_name}{'-reverse' if self.reverse else ''}.json")
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            print(f"Report saved to {save_path}")


def main():
    cfg = parse_args_yaml("BFI test config -> *.yaml")
    bfi_tester = BFIJsonTester(**cfg)
    bfi_tester.run_test()

if __name__ == "__main__":
    main()
