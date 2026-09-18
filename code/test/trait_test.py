# reference from https://github.com/pull-ups/TRAIT.git

import copy
from concurrent.futures import ThreadPoolExecutor
import inspect
import math
import json
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import ordered_trait_names, run_models


def get_prompt(
    prompt_type,
    reverse,
    user_query,
    response_high1,
    response_high2,
    response_low1,
    response_low2,
):
    if reverse:
        option1, option2 = response_low1, response_high1
        option3, option4 = response_low2, response_high2
    else:
        option1, option2 = response_high1, response_low1
        option3, option4 = response_high2, response_low2

    if prompt_type == 1:
        return dedent(
            f"""\
            Given a situation: "{user_query}"
            Please choose from the following options to identify which suggestion you would like to give.
            Options:
            A. {option1}
            B. {option2}
            C. {option3}
            D. {option4}

            Answer:
            """
        )

    elif prompt_type == 2:
        return dedent(
            f"""\
            Now I will briefly describe a situation. Please read a given situation and tell me which option that you want to suggest.
            Write your response among the following options:
            1 = {option1}
            2 = {option2}
            3 = {option3}
            4 = {option4}
            Please answer the suggestion, even if you are not completely sure of your response.

            Situation: {user_query}

            Response: """
        )

    elif prompt_type == 3:
        return dedent(
            f"""\
            Question: {user_query}
            A: {option1}
            B: {option2}
            C: {option3}
            D: {option4}
            Answer: """
        )

    else:
        raise ValueError(
            f"Unsupported prompt type: {prompt_type}, select from 1, 2, or 3."
        )


FEW_SHOT_EXAMPLES = [
    ("Which of the following is an even number?", ["4", "3", "5", "7"], 0),
    (
        "Which of the following animals is a mammal?",
        ["Trout", "Dolphin", "Lizard", "Eagle"],
        1,
    ),
    (
        "Which color is commonly made by mixing blue and yellow?",
        ["Red", "Purple", "Green", "Black"],
        2,
    ),
    (
        "Which season comes immediately after autumn?",
        ["Spring", "Summer", "Autumn", "Winter"],
        3,
    ),
]


class TRAITTester:
    def __init__(
        self,
        model_path,
        dataset_path,
        output_path,
        use_vllm=False,
        dtype="auto",
        device_map="auto",
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        data_parallel_size=1,
        prompt_type=1,
        few_shot=True,
        system_prompt=None,
        trait_prompt=None,
        enable_thinking=False,
        batch_size=4,
        max_length=4096,
        personalities=None,
        max_samples=None,
        adapter_path=None,
        test_name=None,
        preloaded_model=None,
        preloaded_tokenizer=None,
        result_model_name=None,
    ):
        # 将 YAML 参数原样写入结果，方便复现实验。
        self.cfg = dict(locals())
        self.cfg.pop("self")
        self.cfg.pop("preloaded_model")
        self.cfg.pop("preloaded_tokenizer")

        self.model_path = model_path
        self.adapter_path = adapter_path or None
        self.cfg["adapter_path"] = self.adapter_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.use_vllm = use_vllm
        self.dtype = dtype
        self.device_map = device_map
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.prompt_type = prompt_type
        self.few_shot = few_shot
        self.system_prompt = system_prompt or ""
        self.trait_prompt = trait_prompt or ""
        self.enable_thinking = enable_thinking
        self.batch_size = batch_size
        self.max_length = max_length
        self.personalities = personalities
        self.max_samples = max_samples
        self.test_name = test_name
        self.preloaded_model = preloaded_model
        self.preloaded_tokenizer = preloaded_tokenizer
        self.result_model_name = result_model_name

        self.option_labels = (
            ["1", "2", "3", "4"] if prompt_type == 2 else ["A", "B", "C", "D"]
        )
        self.few_shot_messages = self.build_few_shot_messages()
        self.questions = self.load_questions()
        self.result_path = self.build_result_path()
        self.load_model()

    def load_questions(self):
        dataset = load_dataset(self.dataset_path)
        personalities = self.personalities or list(dataset.keys())

        questions = []
        for personality in personalities:
            for index, question in enumerate(dataset[personality]):
                question["dataset_index"] = index
                questions.append(question)

        if self.max_samples is not None:
            questions = questions[: self.max_samples]

        print(f"Loaded {len(questions)} TRAIT questions")
        return questions

    def load_model(self):
        print(f"Loading tokenizer and base/full model from {self.model_path}...")
        self.tokenizer = self.preloaded_tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # TRAIT 比较的是下一个 token，因此选项标签必须是单 token。
        encoded_labels = [
            self.tokenizer.encode(label, add_special_tokens=False)
            for label in self.option_labels
        ]
        if any(len(token_ids) != 1 for token_ids in encoded_labels):
            raise ValueError(
                f"Option labels are not single tokens for this model: "
                f"{dict(zip(self.option_labels, encoded_labels))}"
            )
        self.option_token_ids = [token_ids[0] for token_ids in encoded_labels]
        self.lora_request = None

        if self.preloaded_model is not None and self.use_vllm:
            # vLLM cannot consume an in-memory Transformers model.  Keep the
            # merged weights in memory and use the HF data-parallel path.
            self.use_vllm = False
            self.cfg["use_vllm"] = False

        if self.use_vllm:
            # 只有选择 vLLM 后端时才导入；未安装 vLLM 不影响 HF 模式。
            from vllm import LLM, SamplingParams

            # vLLM cannot consume an in-memory merged Transformers model, so it
            # applies the same adapter natively without writing a merged model.
            llm_kwargs = {
                "model": self.model_path,
                "trust_remote_code": True,
                "dtype": self.dtype,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
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
                temperature=0.0,
                max_tokens=1,
                logprobs=len(self.option_token_ids),
                logprob_token_ids=self.option_token_ids,
            )
            return

        if self.preloaded_model is not None:
            model = self.preloaded_model
            if self.data_parallel_size == 1:
                if self.device_map == "auto":
                    model = model.to(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )
                elif isinstance(self.device_map, str):
                    model = model.to(self.device_map)
        else:
            load_device_map = (
                {"": "cpu"} if self.data_parallel_size > 1 else self.device_map
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=self.dtype,
                device_map=load_device_map,
            )
            if self.adapter_path:
                from peft import PeftModel

                print(f"Merging LoRA adapter from {self.adapter_path}...")
                model = PeftModel.from_pretrained(model, self.adapter_path)
                model = model.merge_and_unload()

        self.models = self.build_data_parallel_models(model)
        self.model = self.models[0]
        self.input_devices = [
            replica.get_input_embeddings().weight.device for replica in self.models
        ]
        self.input_device = self.input_devices[0]

        # Qwen 支持只返回最后一个位置的 logits；其他模型走普通 forward。
        self.keep_last_logit = (
            "logits_to_keep" in inspect.signature(self.model.forward).parameters
        )

    def build_data_parallel_models(self, model):
        """Place persistent model replicas on the requested visible GPUs."""

        if self.data_parallel_size == 1:
            return [model.eval()]
        if torch.cuda.device_count() < self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size={self.data_parallel_size}, but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )

        print(
            f"Building {self.data_parallel_size} persistent Transformers "
            "data-parallel replicas"
        )
        model.requires_grad_(False)
        replicas = [None] * self.data_parallel_size
        for rank in range(1, self.data_parallel_size):
            replicas[rank] = copy.deepcopy(model).to(f"cuda:{rank}").eval()
        replicas[0] = model.to("cuda:0").eval()
        return replicas

    def build_result_path(self):
        if self.result_model_name:
            self.model_name = self.result_model_name
            return Path(self.output_path) / f"trait-{self.model_name}.json"
        model_source = Path(self.adapter_path or self.model_path)
        self.model_name = model_source.name
        # 全参模型和 LoRA adapter 都可能直接指向 checkpoint-* 目录。
        # 加上父训练目录名，避免不同实验的同一步数发生重名。
        if self.model_name.startswith("checkpoint-"):
            self.model_name = f"{model_source.parent.name}-{self.model_name}"
        return Path(self.output_path) / f"trait-{self.model_name}.json"

    def build_few_shot_messages(self):
        messages = []
        for question, options, answer_position in FEW_SHOT_EXAMPLES:
            # normal 模板中的位置依次是 high1、low1、high2、low2。
            # 示例是客观题，四个正确答案分别位于 A/B/C/D。
            prompt = get_prompt(
                self.prompt_type,
                False,
                question,
                options[0],
                options[2],
                options[1],
                options[3],
            )
            messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": self.option_labels[answer_position],
                    },
                ]
            )
        return messages

    def build_messages(self, question, reverse):
        messages = []
        system_prompt = "\n".join(
            part for part in [self.trait_prompt, self.system_prompt] if part
        )
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.few_shot:
            messages.extend(self.few_shot_messages)

        prompt = get_prompt(
            self.prompt_type,
            reverse,
            question["question"],
            question["response_high1"],
            question["response_high2"],
            question["response_low1"],
            question["response_low2"],
        )
        messages.append({"role": "user", "content": prompt})
        return messages

    def render_prompt(self, question, reverse):
        return self.tokenizer.apply_chat_template(
            self.build_messages(question, reverse),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def get_hf_option_probabilities(self, model, input_device, inputs):
        """Run one Transformers replica on one already-tokenized input shard."""

        if input_device.type == "cuda":
            torch.cuda.set_device(input_device)
        model_inputs = {
            name: values.to(input_device) for name, values in inputs.items()
        }
        if self.keep_last_logit:
            model_inputs["logits_to_keep"] = 1

        with torch.inference_mode():
            next_token_logits = model(**model_inputs).logits[:, -1, :]

        option_ids = torch.tensor(
            self.option_token_ids,
            dtype=torch.long,
            device=next_token_logits.device,
        )
        option_logits = next_token_logits.index_select(-1, option_ids).float()
        return torch.softmax(option_logits, dim=-1).cpu().tolist()

    def get_option_probabilities(self, prompts):
        """Score prompts with vLLM or concurrent Transformers replicas."""

        if self.use_vllm:
            outputs = self.llm.generate(
                prompts,
                self.sampling_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )
            probabilities = []
            for output in outputs:
                token_logprobs = output.outputs[0].logprobs[0]
                values = [
                    math.exp(token_logprobs[token_id].logprob)
                    for token_id in self.option_token_ids
                ]
                total = sum(values)
                probabilities.append([value / total for value in values])
            return probabilities

        inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if len(self.models) == 1:
            return self.get_hf_option_probabilities(
                self.models[0], self.input_devices[0], inputs
            )

        # Interleaving keeps both GPUs busy even for one question, whose normal
        # and reversed prompts form a two-item inference batch.
        indices_by_rank = [
            torch.arange(rank, len(prompts), len(self.models))
            for rank in range(len(self.models))
        ]
        ordered = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=len(self.models)) as pool:
            jobs = []
            for model, input_device, indices in zip(
                self.models, self.input_devices, indices_by_rank
            ):
                shard = {
                    name: values.index_select(0, indices)
                    for name, values in inputs.items()
                }
                jobs.append(
                    (
                        indices,
                        pool.submit(
                            self.get_hf_option_probabilities,
                            model,
                            input_device,
                            shard,
                        ),
                    )
                )

            for indices, job in jobs:
                for position, probabilities in zip(indices.tolist(), job.result()):
                    ordered[position] = probabilities
        return ordered

    def test_batch(self, questions):
        prompts = []
        for question in questions:
            prompts.append(self.render_prompt(question, reverse=False))
            prompts.append(self.render_prompt(question, reverse=True))

        probabilities = self.get_option_probabilities(prompts)
        results = []

        for index, question in enumerate(questions):
            normal = probabilities[2 * index]
            reversed_order = probabilities[2 * index + 1]

            # normal:   A=high1, B=low1,  C=high2, D=low2
            # reversed: A=low1,  B=high1, C=low2,  D=high2
            scores = {
                "high1": (normal[0] + reversed_order[1]) / 2,
                "high2": (normal[2] + reversed_order[3]) / 2,
                "low1": (normal[1] + reversed_order[0]) / 2,
                "low2": (normal[3] + reversed_order[2]) / 2,
            }
            selected = max(scores, key=scores.get)
            prediction = "high" if selected.startswith("high") else "low"

            results.append(
                {
                    "personality": question["personality"],
                    "prediction": prediction,
                    "high_probability": scores["high1"] + scores["high2"],
                }
            )

        return results

    @staticmethod
    def summarize(results):
        counts = {}
        high_probabilities = {}
        for result in results:
            personality = result["personality"]
            counts.setdefault(personality, {"high": 0, "low": 0})
            counts[personality][result["prediction"]] += 1
            high_probabilities.setdefault(personality, []).append(
                result["high_probability"]
            )

        personalities = ordered_trait_names(counts)
        scores = {
            personality: counts[personality]["high"]
            / (counts[personality]["high"] + counts[personality]["low"])
            * 100
            for personality in personalities
        }
        counts = {personality: counts[personality] for personality in personalities}
        mean_high_probability = {
            personality: sum(high_probabilities[personality])
            / len(high_probabilities[personality])
            for personality in personalities
        }
        return scores, counts, mean_high_probability

    def save_report(self, results):
        scores, counts, mean_high_probability = self.summarize(results)

        if self.result_path.exists():
            with open(self.result_path, "r", encoding="utf-8") as file:
                report = json.load(file)
        else:
            report = {
                "model_name": self.model_name,
                "model_path": self.model_path,
                "adapter_path": self.adapter_path,
                "tests": [],
            }

        test_number = len(report["tests"]) + 1
        settings = {
            key: value
            for key, value in self.cfg.items()
            if key
            not in {
                "model_path",
                "adapter_path",
                "output_path",
                "test_name",
            }
        }
        test_record = {
            "test_number": test_number,
            "test_id": f"test-{test_number:03d}",
            "test_name": self.test_name,
            "tested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "settings": settings,
            "num_questions": len(results),
            "scores": scores,
            "counts": counts,
            "mean_high_probability": mean_high_probability,
        }
        report["tests"].append(test_record)

        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.result_path, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        return test_record

    def run_test(self):
        results = []
        print(f"Testing {len(self.questions)} questions")

        batch_starts = range(0, len(self.questions), self.batch_size)
        for start in tqdm(batch_starts, desc="TRAIT inference", unit="batch"):
            batch = self.questions[start : start + self.batch_size]
            results.extend(self.test_batch(batch))

        test_record = self.save_report(results)
        scores = test_record["scores"]
        counts = test_record["counts"]

        print("\n=== TRAIT REPORT ===")
        for personality, score in scores.items():
            total = counts[personality]["high"] + counts[personality]["low"]
            print(f"{personality:<20}: {score:>6.2f}  (N={total})")
        print(f"Report saved to {self.result_path}")

        return test_record


def run(config):
    """Evaluate one model and return its TRAIT test record."""
    return TRAITTester(**config).run_test()


def main():
    cfg = parse_args_yaml("TRAIT test config -> *.yaml")
    run_models(cfg, run, "TRAIT")


if __name__ == "__main__":
    main()
