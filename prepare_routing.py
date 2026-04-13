"""
One-time data preparation for Gemma 4 routing SFT.

Usage:
    uv run prepare_routing.py

Reads the routing JSONL files from ./data, normalizes Gemma 4 thinking traces,
filters malformed rows, and caches cleaned train/eval examples in ~/.cache/routing.
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_FILE = DATA_DIR / "dysko_gemma4_train.jsonl"
EVAL_FILE = DATA_DIR / "dysko_gemma4_eval.jsonl"
CACHE_DIR = Path(os.path.expanduser("~/.cache/routing"))

OLD_THINK_RE = re.compile(r"<\|think\|>(.*?)(?:<\|/think\|>|</\|think\|>)", re.DOTALL)
TOOL_RE = re.compile(r'"name"\s*:\s*"(find_[^"]+)"')


def convert_think_tokens(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        body = match.group(1).strip("\n")
        if body:
            return f"<|channel>thought\n{body}\n<channel|>"
        return "<|channel>thought\n<channel|>"

    return OLD_THINK_RE.sub(repl, text)


def normalize_example(example: dict) -> dict | None:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    if not all(isinstance(message.get("content"), str) for message in messages):
        return None

    cleaned = copy.deepcopy(example)
    for message in cleaned["messages"]:
        message["content"] = message["content"].replace("\r\n", "\n")
        if message.get("role") == "model":
            message["content"] = convert_think_tokens(message["content"])
    return cleaned


def load_and_clean(path: Path) -> tuple[list[dict], dict]:
    raw_examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cleaned_examples = []
    tool_counter: Counter[str] = Counter()
    non_tool_examples = 0
    dropped_examples = 0

    for example in raw_examples:
        cleaned = normalize_example(example)
        if cleaned is None:
            dropped_examples += 1
            continue

        final_content = cleaned["messages"][-1]["content"]
        tool_match = TOOL_RE.search(final_content)
        if tool_match:
            tool_counter[tool_match.group(1)] += 1
        else:
            non_tool_examples += 1
        cleaned_examples.append(cleaned)

    old_think_remaining = sum(
        1
        for example in cleaned_examples
        for message in example["messages"]
        if message["role"] == "model" and "<|think|>" in message["content"]
    )
    assert old_think_remaining == 0, f"{path.name}: found unconverted <|think|> blocks"

    summary = {
        "file": path.name,
        "total": len(raw_examples),
        "usable": len(cleaned_examples),
        "dropped": dropped_examples,
        "tool_examples": sum(tool_counter.values()),
        "non_tool_examples": non_tool_examples,
        "tool_breakdown": dict(tool_counter),
    }
    return cleaned_examples, summary


def main() -> None:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing training file: {TRAIN_FILE}")
    if not EVAL_FILE.exists():
        raise FileNotFoundError(f"Missing eval file: {EVAL_FILE}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    train_examples, train_summary = load_and_clean(TRAIN_FILE)
    eval_examples, eval_summary = load_and_clean(EVAL_FILE)

    with (CACHE_DIR / "train.pkl").open("wb") as handle:
        pickle.dump(train_examples, handle)
    with (CACHE_DIR / "eval.pkl").open("wb") as handle:
        pickle.dump(eval_examples, handle)

    metadata = {
        "train": train_summary,
        "eval": eval_summary,
        "cache_dir": str(CACHE_DIR),
    }
    (CACHE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved cleaned data to {CACHE_DIR}")
    for split_name, summary in [("Train", train_summary), ("Eval", eval_summary)]:
        print(f"{split_name}: {summary['usable']}/{summary['total']} usable, dropped {summary['dropped']}")
        print(f"  Tool examples: {summary['tool_examples']}")
        print(f"  Non-tool examples: {summary['non_tool_examples']}")
        print(f"  Tool breakdown: {summary['tool_breakdown']}")


if __name__ == "__main__":
    main()
