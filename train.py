"""
Autoresearch routing SFT script for Gemma 4 on a single NVIDIA GPU.

Usage:
    uv run train.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Hyperparameters (the agent is expected to tune these)
# ---------------------------------------------------------------------------

CACHE_DIR = Path(os.path.expanduser("~/.cache/routing"))
OUTPUT_DIR = Path("./gemma4-adapter")
MODEL_ID = "google/gemma-4-E4B-it"

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 512
NUM_EPOCHS = 3
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.0
BF16 = True
SEED = 42

LOAD_IN_4BIT = True
GRADIENT_CHECKPOINTING = True
ATTN_IMPLEMENTATION = "sdpa"
OPTIMIZER = "paged_adamw_8bit"

# ---------------------------------------------------------------------------
# Fixed evaluation helpers (do not change)
# ---------------------------------------------------------------------------

ROLE_MAP = {"model": "assistant"}
OLD_THINK_RE = re.compile(r"<\|think\|>(.*?)(?:<\|/think\|>|</\|think\|>)", re.DOTALL)
NEW_THINK_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)
TOOL_RE = re.compile(r'"name"\s*:\s*"(find_[^"]+)"')


def strip_thinking(text: str) -> str:
    text = OLD_THINK_RE.sub("", text)
    text = NEW_THINK_RE.sub("", text)
    return text.lstrip()


def normalize_messages(messages: list[dict], strip_history_thoughts: bool) -> list[dict]:
    normalized = []
    last_index = len(messages) - 1
    for index, message in enumerate(messages):
        role = ROLE_MAP.get(message["role"], message["role"])
        content = message["content"].replace("\r\n", "\n")
        if strip_history_thoughts and role == "assistant" and index != last_index:
            content = strip_thinking(content)
        normalized.append({"role": role, "content": content})
    return normalized


def render_messages(processor, messages: list[dict], add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )


@dataclass
class EvalMetrics:
    overall_accuracy: float
    tool_accuracy: float
    notool_accuracy: float
    tool_correct: int
    tool_total: int
    notool_correct: int
    notool_total: int


@torch.no_grad()
def evaluate_routing(model, processor, eval_data: list[dict]) -> EvalMetrics:
    model.eval()
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device

    tool_correct = 0
    tool_total = 0
    notool_correct = 0
    notool_total = 0

    prompt_limit = max(1, MAX_SEQ_LEN - MAX_NEW_TOKENS)

    for example in eval_data:
        gt_content = example["messages"][-1]["content"]
        gt_tool_match = TOOL_RE.search(gt_content)
        is_tool_query = gt_tool_match is not None

        prompt_messages = normalize_messages(example["messages"][:-1], strip_history_thoughts=True)
        prompt_text = render_messages(processor, prompt_messages, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        if inputs["input_ids"].shape[1] > prompt_limit:
            for key in inputs:
                inputs[key] = inputs[key][:, -prompt_limit:]
        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        completion = outputs[0][inputs["input_ids"].shape[1]:].detach().cpu()
        pred_text = processor.decode(completion, skip_special_tokens=False)
        pred_tool_match = TOOL_RE.search(pred_text)

        if is_tool_query:
            tool_total += 1
            if pred_tool_match and pred_tool_match.group(1) == gt_tool_match.group(1):
                tool_correct += 1
        else:
            notool_total += 1
            if pred_tool_match is None:
                notool_correct += 1

    total = tool_total + notool_total
    overall = (tool_correct + notool_correct) / max(total, 1)
    tool_accuracy = tool_correct / max(tool_total, 1)
    notool_accuracy = notool_correct / max(notool_total, 1)

    print("--- Routing Evaluation ---")
    print(f"Tool queries: {tool_correct}/{tool_total} = {tool_accuracy * 100:.1f}%")
    print(f"Non-tool queries: {notool_correct}/{notool_total} = {notool_accuracy * 100:.1f}%")
    print(f"Overall accuracy: {overall * 100:.2f}%")

    return EvalMetrics(
        overall_accuracy=overall,
        tool_accuracy=tool_accuracy,
        notool_accuracy=notool_accuracy,
        tool_correct=tool_correct,
        tool_total=tool_total,
        notool_correct=notool_correct,
        notool_total=notool_total,
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def load_cached_examples(split: str) -> list[dict]:
    path = CACHE_DIR / f"{split}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached {split} data at {path}. Run `uv run prepare_routing.py` first."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_metadata() -> dict:
    path = CACHE_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


class RoutingSFTDataset(Dataset):
    def __init__(self, examples: list[dict], processor, max_seq_len: int):
        self.examples = examples
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_len = max_seq_len
        self.features = [self._tokenize_example(example) for example in examples]

    def _tokenize_example(self, example: dict) -> dict:
        prompt_messages = normalize_messages(example["messages"][:-1], strip_history_thoughts=True)
        full_messages = normalize_messages(example["messages"], strip_history_thoughts=True)

        prompt_text = render_messages(self.processor, prompt_messages, add_generation_prompt=True)
        full_text = render_messages(self.processor, full_messages, add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        overflow = max(0, len(full_ids) - self.max_seq_len)
        if overflow:
            full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)

        labels = list(full_ids)
        for index in range(min(prompt_len, len(labels))):
            labels[index] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


class RoutingCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model / trainer setup
# ---------------------------------------------------------------------------


def build_quantization_config() -> BitsAndBytesConfig | None:
    if not LOAD_IN_4BIT:
        return None
    compute_dtype = torch.bfloat16 if BF16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def make_model_and_processor():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this training script.")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = build_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map={"": torch.cuda.current_device()} if LOAD_IN_4BIT else None,
        attn_implementation=ATTN_IMPLEMENTATION,
    )

    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        )
    elif GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, processor


def make_training_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=BF16,
        fp16=not BF16,
        logging_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=MAX_GRAD_NORM,
        optim=OPTIMIZER,
        seed=SEED,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this training script.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()

    metadata = load_metadata()
    train_examples = load_cached_examples("train")
    eval_examples = load_cached_examples("eval")

    print(f"Model: {MODEL_ID}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")
    if metadata:
        print("Cached dataset summary:")
        print(json.dumps(metadata, indent=2))

    model, processor = make_model_and_processor()
    train_dataset = RoutingSFTDataset(train_examples, processor, MAX_SEQ_LEN)
    collator = RoutingCollator(processor.tokenizer)

    training_args = make_training_args()
    print("Training configuration:")
    print(json.dumps(training_args.to_dict(), indent=2, default=str))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    training_seconds = time.time() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    trainer.model.config.use_cache = True
    eval_metrics = evaluate_routing(trainer.model, processor, eval_examples)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"overall_accuracy: {eval_metrics.overall_accuracy:.6f}")
    print(f"tool_accuracy:    {eval_metrics.tool_accuracy:.6f}")
    print(f"notool_accuracy:  {eval_metrics.notool_accuracy:.6f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"num_epochs:       {NUM_EPOCHS}")
    print(f"num_train:        {len(train_examples)}")
    print(f"num_eval:         {len(eval_examples)}")
    print(f"lora_r:           {LORA_R}")
    print(f"lora_targets:     {','.join(LORA_TARGETS)}")


if __name__ == "__main__":
    main()
