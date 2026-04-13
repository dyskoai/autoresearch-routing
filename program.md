# autoresearch-routing

This repo adapts `karpathy/autoresearch` from 5-minute GPT pretraining to autonomous single-GPU routing SFT for `google/gemma-4-E4B-it`.

## Setup

When a human asks you to start a new unattended run, do the following before the first experiment:

1. Pick a fresh run tag based on the date, for example `apr13-gemma4`.
2. Create a dedicated branch from the current default branch:
   - `git checkout -b autoresearch-routing/<tag>`
3. Read the in-scope files for context:
   - `data/main-tuning.md`
   - `prepare_routing.py`
   - `train.py`
   - `pyproject.toml`
4. Verify the machine has a usable NVIDIA GPU:
   - `nvidia-smi`
   - `uv run python -c "import torch; print(torch.cuda.is_available())"`
5. Verify dependencies are installed. If imports fail, run `uv sync`.
6. Verify the cached routing data exists in `~/.cache/routing/`:
   - If `train.pkl` or `eval.pkl` is missing, run `uv run prepare_routing.py`
7. Initialize `results.tsv` with the header row if it does not exist yet:
   - `commit	accuracy	memory_gb	status	description`
8. Start the baseline run immediately once setup is valid. Do not wait for extra confirmation.

## Ground Rules

You are running autonomous research on a single GPU. The main objective is to maximize `overall_accuracy` from the fixed evaluator in `train.py`.

What you can modify:
- `train.py`

What you must not modify during experiments:
- `prepare_routing.py`
- the fixed routing evaluator in `train.py`
- the dataset files in `data/`

The first run must always be the untouched baseline.

## Metric

The score to optimize is:

`overall_accuracy = (correct_tool_routes + correct_non_tool_responses) / total_eval_examples`

Higher is better.

Current cached split after filtering malformed rows:
- Train: 297 usable examples
- Eval: 74 usable examples
- Eval tool queries: 49
- Eval non-tool queries: 25

Important limitation:
- The current eval set has no `find_pairing_suggestions` examples after filtering.
- The train set has only 1 pairing example.
- Do not claim pairing quality improved unless you also spot-check it manually with direct prompts.

## Output Format

Each run ends with a summary like this:

```
---
overall_accuracy: 0.923100
tool_accuracy:    0.938800
notool_accuracy:  0.896600
peak_vram_mb:     11240.0
training_seconds: 1820.5
num_epochs:       3
num_train:        297
num_eval:         74
lora_r:           16
lora_targets:     q_proj,k_proj,v_proj,o_proj
```

Extract the key metrics from the log with:

```
grep "^overall_accuracy:\|^peak_vram_mb:" run.log
```

## Logging Results

After every experiment, append one line to `results.tsv` using tab separation:

```
commit	accuracy	memory_gb	status	description
```

Columns:
1. short git commit hash
2. `overall_accuracy` as a float, or `0.000000` on crash
3. peak memory in GB with one decimal place, or `0.0` on crash
4. `keep`, `discard`, or `crash`
5. short description of the experiment

Example:

```
commit	accuracy	memory_gb	status	description
abc1234	0.662162	14.8	keep	baseline qlora config
def5678	0.689189	15.1	keep	raise lora rank to 32
ghi9012	0.675676	15.0	discard	add mlp targets with same lr
jkl3456	0.000000	0.0	crash	flash attention 2 without dependency
```

## What To Tune

Prioritize simple, high-signal changes:
- LoRA rank, alpha, dropout
- LoRA target modules, including MLP projections
- learning rate and scheduler
- batch size and gradient accumulation
- number of epochs
- max sequence length
- gradient checkpointing
- attention implementation when the dependency is already available

Prefer changes that improve accuracy without materially inflating VRAM or adding brittle code.

## The Experiment Loop

LOOP FOREVER:

1. Inspect the current git commit and recent `results.tsv`.
2. Pick one concrete idea for `train.py`.
3. Edit `train.py`.
4. Commit the experiment.
5. Run:
   - `uv run train.py > run.log 2>&1`
6. Read out the result:
   - `grep "^overall_accuracy:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed:
   - inspect `tail -n 80 run.log`
   - fix obvious bugs and retry once if the idea is still valid
   - otherwise log `crash` and move on
8. Append the result to `results.tsv`.
9. If `overall_accuracy` improved, keep the commit and continue from there.
10. If accuracy is flat or worse, reset the branch back to the previous keep-point and continue with a new idea.

## Runtime Expectations

This is full SFT, not a 5-minute toy loop.

Expected run time:
- roughly 20 to 60 minutes per experiment on a practical single GPU

Timeout rule:
- if a run exceeds 90 minutes, treat it as failed unless the log shows steady progress and you have a specific reason to wait longer

## Autonomy

Once the loop begins, do not pause for human confirmation. Keep running experiments until manually stopped.
