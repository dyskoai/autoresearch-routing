# GPU Prep Checklist

Use this on the machine where Codex will run the experiments.

## 1. Confirm You Are On The GPU Box

Run:

```bash
pwd
hostname
nvidia-smi
```

Good signs:
- `nvidia-smi` works
- you see one NVIDIA GPU
- GPU memory is visible

Bad signs:
- `nvidia-smi: command not found`
- no GPU listed
- CUDA driver error

## 2. Confirm Python Sees CUDA

Run:

```bash
uv run python -c "import torch; print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count()); print('name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Good output looks like:

```text
cuda: True
count: 1
name: NVIDIA ...
```

If `cuda: False`, stop here.

## 3. Confirm You Are In The Correct Repo

Run:

```bash
ls
test -f train.py && test -f prepare_routing.py && test -f program.md && echo OK
```

You should be inside the tracked repo root, the one containing:
- `train.py`
- `prepare_routing.py`
- `program.md`
- `data/`

## 4. Install Dependencies

Run:

```bash
uv sync
```

This should install the added training stack:
- `transformers`
- `accelerate`
- `peft`
- `bitsandbytes`
- `sentencepiece`

## 5. Confirm The Model Stack Imports

Run:

```bash
uv run python -c "import torch, transformers, peft, bitsandbytes, accelerate; print('imports ok')"
```

Expected:

```text
imports ok
```

## 6. Prepare The Routing Cache

Run:

```bash
uv run prepare_routing.py
```

Good output should include:
- `Saved cleaned data to .../.cache/routing`
- `Train: 297/350 usable, dropped 53`
- `Eval: 74/78 usable, dropped 4`

Then check the cache:

```bash
ls ~/.cache/routing
```

Expected files:
- `train.pkl`
- `eval.pkl`
- `metadata.json`

## 7. Smoke-Test The Training Script

Run:

```bash
uv run train.py
```

This is the real baseline run, not a fake test. Let it start.

Good signs in the log:
- model loads successfully
- trainable LoRA parameter summary prints
- training begins without immediate CUDA OOM
- final summary prints `overall_accuracy`

Expected final lines look like:

```text
---
overall_accuracy: ...
tool_accuracy:    ...
notool_accuracy:  ...
peak_vram_mb:     ...
training_seconds: ...
```

## 8. Decide If The Session Is Good To Go

The session is ready if all of these are true:
- `nvidia-smi` works
- `torch.cuda.is_available()` is `True`
- `uv sync` completed
- `prepare_routing.py` created the cache
- `train.py` completed one run and printed `overall_accuracy`

If all five pass, Codex can start autonomous experiment runs.

## 9. Kick Off Autonomous Runs

After the baseline succeeds, open Codex in this repo and point it to:

- `program.md`
- `data/main-tuning.md`

Then tell it to start a new unattended run on the GPU machine.

## Common Failure Points

- `nvidia-smi` fails: wrong machine or broken NVIDIA driver
- model download fails: no Hugging Face access on that machine
- `bitsandbytes` import fails: CUDA/toolchain mismatch
- OOM on startup: GPU too small for current settings
- `overall_accuracy` missing: training crashed, inspect `run.log`

