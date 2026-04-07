# Plan: Adapt Autoresearch Framework for Gemma 4 E4B Routing SFT

## Context
The user wants to replace GPT-120B-OSS with `google/gemma-4-E4B-it` (4.5B effective params) fine-tuned on 350 fashion routing conversations. Instead of writing a standalone training script, we adapt the existing **autoresearch framework** — which already has an autonomous loop where Claude modifies `train.py`, runs experiments, checks results, and iterates forever. We just need to:
1. Preprocess the data (one-time, human step)
2. Replace `train.py` with a Gemma 4 SFT baseline
3. Update `program.md` for the new task and metric

---

## Pre-Requisite: GPU Access via VS Code Remote SSH

Claude Code runs Bash commands on whatever machine VS Code is connected to. To give Claude access to a GPU, connect VS Code to a remote GPU cloud instance via Remote SSH — then all terminal commands (including training runs) execute on that machine.

### 1 — Rent a GPU Instance

**RunPod** (recommended — reliable, BF16 native):
1. Go to runpod.io → Secure Cloud → filter **L4 24 GB** (~$0.44/hr) or **RTX 3090** (~$0.44/hr)
2. Select the **PyTorch 2.x / CUDA 12.x** template (torch + CUDA pre-installed)
3. Deploy → copy the SSH command shown (e.g. `ssh root@XX.XX.XX.XX -p XXXXX -i ~/.ssh/id_ed25519`)

**Vast.ai** (cheapest — RTX 3090 ~$0.25/hr):
1. Filter: RTX 3090, CUDA ≥ 12.1, Ubuntu 22.04
2. Rent → get SSH host/port/key from the instance dashboard

### 2 — Install Remote-SSH in VS Code

In VS Code Extensions panel, search and install: **Remote - SSH** (by Microsoft)

### 3 — Connect VS Code to the GPU Machine

1. Press `Ctrl+Shift+P` → type `Remote-SSH: Connect to Host`
2. Paste your SSH connection string (e.g. `root@XX.XX.XX.XX -p XXXXX`)
3. VS Code reopens — status bar bottom-left shows `SSH: <host>`
4. Open the project folder on the remote machine: `File → Open Folder`

Claude Code (VS Code extension) now runs all Bash commands on the remote GPU machine.

### 4 — Transfer Project Files to the Instance

From your Windows machine (run in local terminal, not the SSH terminal):
```bash
scp -P <PORT> "C:\Users\SHILPI DARBARI\autosearch-routing\dysko_gemma4_train.jsonl" root@<IP>:~/autosearch-routing/
scp -P <PORT> "C:\Users\SHILPI DARBARI\autosearch-routing\dysko_gemma4_eval.jsonl"  root@<IP>:~/autosearch-routing/
# Or clone/copy the full repo
```

### 5 — Verify GPU Access

Once connected, ask Claude Code to run:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"
```

If both commands succeed, Claude Code has full GPU access and can run all subsequent steps autonomously.

### 6 — Install New Dependencies (once, on the GPU machine)

```bash
pip install transformers>=4.51.3 trl==0.17.0 peft==0.15.1 \
            bitsandbytes==0.45.5 accelerate==1.6.0 datasets==3.5.0 \
            sentencepiece==0.2.0 protobuf scipy
```

---

## Environment

| Item | Value |
|---|---|
| Python | 3.10 (`.python-version`) |
| CUDA | 12.8 (`pyproject.toml`, torch cu128) |
| Model | `google/gemma-4-E4B-it` |
| GPU (low cost) | **L4 24 GB on RunPod** (~$0.44/hr) — or RTX 3090 on Vast.ai (~$0.25/hr) |

---

## Dataset Facts

- **Train**: 350 raw → ~300 usable (filter 50 with `user.content = NaN`)
- **Eval**: 78 total — **~49 tool-call queries**, **~29 non-tool queries** (greetings, follow-ups)
- Roles: `system / user / model` — already Gemma-native, no remapping needed
- Tool imbalance: `find_products_directly` 133, `find_styling_ideas` 59, `find_pairing_suggestions` 1

### Thinking Token Mismatch
Dataset uses `<|think|>...<|/think|>` but the model natively produces `<|channel>thought\n...<channel|>`.
Must convert during preprocessing (Step 1 below).

---

## Evaluation Metric: Two-Part Accuracy

The autoresearch metric must cover **both** query types:

```
overall_accuracy = (correct_tool_routes + correct_non_tool_responses) / 78
```

- **Tool queries (~49)**: Did the model generate the right tool name (`find_products_directly`, `find_styling_ideas`, `find_pairing_suggestions`)?
- **Non-tool queries (~29)**: Did the model correctly generate NO `<tool_call>` block? (Greetings like "hi", follow-ups like "thanks, that's great")

This replaces `val_bpb` as the autoresearch metric.

---

## Files to Create/Modify

| File | Action | Note |
|---|---|---|
| `autoresearch-routing/prepare_routing.py` | **Create** | One-time preprocessing (like `prepare.py`) |
| `autoresearch-routing/train.py` | **Replace** | Gemma 4 QLoRA SFT baseline (agent modifies this) |
| `autoresearch-routing/program.md` | **Update** | New metric, task description, what agent can change |
| `autoresearch-routing/pyproject.toml` | **No change** | Existing torch 2.9.1+cu128 is fine |
| `autoresearch-routing/prepare.py` | **No change** | Leave as-is |

---

## Step 1 — `prepare_routing.py` (One-Time, Human Runs This)

Located at: `autoresearch-routing/prepare_routing.py`

Purpose: Preprocess JSONL once, save processed data for the training loop.

```python
"""
One-time data preparation for Gemma 4 routing SFT.
Run once before starting the autoresearch loop: python prepare_routing.py
Saves processed data to ~/.cache/routing/
"""
import os, json, re, pickle

TRAIN_FILE = "../dysko_gemma4_train.jsonl"
EVAL_FILE  = "../dysko_gemma4_eval.jsonl"
CACHE_DIR  = os.path.expanduser("~/.cache/routing")

THINK_RE = re.compile(r"<\|think\|>(.*?)<\|/think\|>", re.DOTALL)

def convert_think_tokens(text):
    """Convert dataset think format to Gemma 4 E4B native format."""
    return THINK_RE.sub(r"<|channel>thought\n\1<channel|>", text)

def load_and_clean(path):
    with open(path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]
    cleaned = []
    for ex in raw:
        # Filter examples where any message content is not a string (NaN follow-ups)
        if not all(isinstance(m["content"], str) for m in ex["messages"]):
            continue
        # Convert thinking tokens in model messages
        for m in ex["messages"]:
            if m["role"] == "model":
                m["content"] = convert_think_tokens(m["content"])
        cleaned.append(ex)
    return cleaned

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    train = load_and_clean(TRAIN_FILE)
    eval_ = load_and_clean(EVAL_FILE)
    print(f"Train: {len(train)} examples (filtered {350 - len(train)} NaN)")
    print(f"Eval:  {len(eval_)} examples")

    # Verify think tokens converted correctly
    model_msgs = [m["content"] for ex in train for m in ex["messages"] if m["role"] == "model"]
    think_count = sum(1 for c in model_msgs if "<|channel>thought" in c)
    old_think   = sum(1 for c in model_msgs if "<|think|>" in c)
    print(f"Think blocks converted: {think_count}, old format remaining: {old_think}")
    assert old_think == 0, "Some <|think|> tokens not converted!"

    with open(os.path.join(CACHE_DIR, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    with open(os.path.join(CACHE_DIR, "eval.pkl"), "wb") as f:
        pickle.dump(eval_, f)
    print(f"Saved to {CACHE_DIR}")
```

---

## Step 2 — `train.py` Replacement (Gemma 4 SFT Baseline)

The agent will modify this file each experiment, just like the original autoresearch.

**What the agent CAN modify** (same philosophy as before):
- LoRA rank (`LORA_R`), alpha (`LORA_ALPHA`), dropout
- Target modules (`LORA_TARGETS`) — can add `gate_proj`, `up_proj`, `down_proj` for MLP LoRA
- Learning rate, scheduler, warmup
- Batch size, gradient accumulation
- Number of epochs
- Gradient checkpointing on/off
- Flash attention on/off (if installed)

**Configuration block** (top of train.py, agent edits these):
```python
CACHE_DIR    = os.path.expanduser("~/.cache/routing")
OUTPUT_DIR   = "./gemma4-adapter"
MODEL_ID     = "google/gemma-4-E4B-it"

LORA_R       = 16
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_SEQ_LEN  = 8192
NUM_EPOCHS   = 3
LR           = 2e-4
BATCH_SIZE   = 1
GRAD_ACCUM   = 4
BF16         = True
```

**Evaluation function** (fixed — agent cannot change this):
```python
TOOL_RE = re.compile(r'"name":\s*"(find_\w+)"')

@torch.no_grad()
def evaluate_routing(model, processor, eval_data):
    """
    Two-part accuracy:
    - Tool queries: correct tool name selected
    - Non-tool queries: no <tool_call> generated (greetings, follow-ups)
    Returns overall_accuracy (higher = better).
    """
    model.eval()
    tokenizer = processor.tokenizer
    tool_correct, tool_total = 0, 0
    notool_correct, notool_total = 0, 0

    for ex in eval_data:
        gt_content = ex["messages"][-1]["content"]
        gt_tool = TOOL_RE.search(gt_content)
        is_tool_query = gt_tool is not None

        prompt_msgs = ex["messages"][:-1]
        prompt = processor.apply_chat_template(
            prompt_msgs, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LEN - 256
        ).to(model.device)

        out = model.generate(
            **inputs, max_new_tokens=512,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        pred_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:])
        pred_tool = TOOL_RE.search(pred_text)

        if is_tool_query:
            tool_total += 1
            if pred_tool and pred_tool.group(1) == gt_tool.group(1):
                tool_correct += 1
        else:
            notool_total += 1
            if pred_tool is None:
                notool_correct += 1

    overall = (tool_correct + notool_correct) / (tool_total + notool_total)
    print(f"--- Routing Evaluation ---")
    print(f"Tool queries:    {tool_correct}/{tool_total} = {tool_correct/max(tool_total,1)*100:.1f}%")
    print(f"Non-tool (greet/follow-up): {notool_correct}/{notool_total} = {notool_correct/max(notool_total,1)*100:.1f}%")
    print(f"Overall accuracy: {overall*100:.2f}%")
    return overall
```

**Output format** (printed at end of each run):
```
---
overall_accuracy: 0.923100
tool_accuracy:    0.938800
notool_accuracy:  0.896600
peak_vram_mb:     11240.0
training_seconds: 1820.5
num_epochs:       3
num_train:        298
lora_r:           16
lora_targets:     q_proj,k_proj,v_proj,o_proj
```

---

## Step 3 — Update `program.md`

Key changes from the original:
- **Metric**: `overall_accuracy` (higher = better, range 0–1) replaces `val_bpb`
- **Time budget**: Remove fixed 5-min budget. Each run = full SFT (typically 20–45 min on L4)
- **What to tune**: LoRA hyperparams, LR, batch size, target modules, epochs — NOT model architecture
- **Results TSV columns**: `commit  accuracy  memory_gb  status  description`
- **Eval** is the fixed `evaluate_routing()` function — agent cannot modify it

---

## Autoresearch Loop Behavior

Once `python prepare_routing.py` has been run once:

```
Agent loop:
1. Read current train.py config block
2. Hypothesize: try higher LORA_R=32? add MLP targets? lower LR?
3. Edit train.py
4. git commit
5. uv run train.py > run.log 2>&1
6. grep "^overall_accuracy:\|^peak_vram_mb:" run.log
7. If accuracy improved → keep commit, advance branch
8. If equal or worse → git reset, log discard
9. NEVER STOP
```

Each run takes ~20–45 min (vs 5 min for GPT pre-training), so expect ~15–30 experiments overnight.

---

## Greeting / Non-Tool Testing (Manual Verification After Training)

**Should produce a tool call:**
- "show me red kurtas under 1000" → `find_products_directly`, color=Red, category=Kurtas
- "black formal shirts for men" → `find_products_directly`, gender=Men
- "how do I style this white shirt?" → `find_styling_ideas`
- "what goes with my black jeans?" → `find_pairing_suggestions`

**Should NOT produce a tool call (greetings/follow-ups):**
- "hi" → friendly greeting, no `<tool_call>`
- "hello, what can you help me with?" → intro response, no `<tool_call>`
- "thanks, that looks great!" → acknowledgment, no `<tool_call>`
- "perfect, I love it" → positive response, no `<tool_call>`

**Edge cases:**
- "hi, show me red dresses" → mixed greeting+query → SHOULD produce `find_products_directly`
- Verify output JSON is valid and parseable
- Verify `journey_name` is always 2–4 words

---

## Critical Files

| File | Role |
|---|---|
| `autosearch-routing/dysko_gemma4_train.jsonl` | Raw training data |
| `autosearch-routing/dysko_gemma4_eval.jsonl` | Raw eval data |
| `autoresearch-routing/prepare_routing.py` | Create: one-time preprocessing |
| `autoresearch-routing/train.py` | Replace: Gemma 4 SFT baseline |
| `autoresearch-routing/program.md` | Update: new metric + task |
| `autoresearch-routing/pyproject.toml` | Install new deps alongside existing |

---

## Verification

1. Run `python prepare_routing.py` → `~/.cache/routing/train.pkl` and `eval.pkl` created, 0 old think tokens remaining
2. Run `uv run train.py` baseline → prints `overall_accuracy`, no crash
3. Confirm `tool_accuracy` and `notool_accuracy` both printed
4. Autoresearch loop begins; first commit should be the baseline
5. After 5+ experiments, check `results.tsv` for improvement trend
