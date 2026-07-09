# 144M-Param LLM — Pretraining Pipeline

A from-scratch decoder-only transformer (144M parameters) trained on ~5B tokens.  
Architecture: RoPE + GQA (12q/4kv heads) + SwiGLU + RMSNorm + tied embeddings.  
Tokenizer: Llama-2 (32K vocab).

---

## Project Structure

```
llm144m/
├── config.py                  Single source of truth for all hyperparameters
├── train.py                   Main pretraining script
├── lr_schedule.py             Cosine LR with warmup
├── checkpoint.py              Rolling-window + best-val + time-based checkpointing
├── logger.py                  CSV logger (+ optional wandb)
│
├── model/
│   ├── rmsnorm.py             RMSNorm (fp32 variance upcast)
│   ├── rope.py                Rotary position embeddings (θ=10000)
│   ├── attention.py           GQA via F.scaled_dot_product_attention(enable_gqa=True)
│   ├── ffn.py                 SwiGLU FFN
│   └── transformer.py         Full 144M-param decoder-only transformer
│
├── data/
│   ├── dataloader.py          np.memmap sharded dataloader
│   ├── prepare_fineweb.py     FineWeb → .bin shards  (GPU pod, datatrove)
│   └── prepare_stories.py     TinyStories → .bin shards  (GPU pod, wget)
│
├── validate/
│   ├── validate_components.py Per-component allclose vs HF LlamaModel
│   └── validate_forward.py    Full forward pass allclose vs HF LlamaForCausalLM
│
├── scripts/
│   ├── smoke_test.sh          100-step sanity check (loss < ln(32000))
│   └── run_pretrain.sh        Full 9,537-step run
│
├── requirements.txt
└── .env               Copy to .env, fill in HF_TOKEN
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set HF token (needed for Llama-2 tokenizer)
cp .env
# Edit .env and set HF_TOKEN=hf_...
```

---

## Execution Order (GPU Pod)

### Step 1 — Prepare data

```bash
# FineWeb (~4.7B tokens → ~47 shards × 100M tokens)
python data/prepare_fineweb.py

# TinyStories (~300M tokens → ~3 shards)
python data/prepare_stories.py
```

> Both scripts run entirely on the pod — no local download needed.  
> FineWeb uses datatrove streaming; stories use wget against the HF resolve URL.  
> Expected output: `data/shards/train_*.bin`, `data/shards/stories_*.bin`, `data/val.bin`

### Step 2 — Validate model architecture

```bash
# Per-component checks (RMSNorm, FFN, GQA+RoPE) vs randomly-initialized HF LlamaModel
python validate/validate_components.py

# Full forward pass check
python validate/validate_forward.py
```

> No HF token required for validation — uses randomly-initialized LlamaConfig at our dimensions.  
> Both must exit 0 before training.

### Step 3 — Smoke test

```bash
bash scripts/smoke_test.sh
```

> Runs 100 steps and asserts final loss < ln(32000) ≈ 10.37.  
> If it fails: check model init, LR, or data loading — do NOT proceed to full run.

### Step 4 — Full pretraining

```bash
bash scripts/run_pretrain.sh
```

> Expected runtime:  
> • H100 with `torch.compile` + TF32: **~4.25–4.28 hours**  
> • RTX Pro 6000 (no compile, baseline): ~10.5 hours  
> Checkpoints saved to `checkpoints/` every 500 steps (rolling last-3 + best-val).

### Resuming after interruption

```bash
python train.py --resume checkpoints/step_005000.pt
```

---

## Key Config Values

| Parameter | Value |
|---|---|
| d_model | 768 |
| n_layers | 16 |
| n_heads / n_kv_heads | 12 / 4 (GQA 3:1) |
| ffn_hidden_dim | 2560 |
| vocab_size | 32000 (Llama-2) |
| seq_len | 2048 |
| total_steps | 9,537 |
| effective_batch_tokens | 524,288 (16 × 16 × 2048) |
| peak_lr / min_lr | 6e-4 / 6e-5 |
| warmup_steps | 350 |
| precision | bf16 (forward/backward), fp32 (grads + norms + loss) |
| compile | torch.compile — 47% speedup on H100 |
| TF32 | enabled — near-free throughput on Ampere+ |

---

## Cloud-Pod Workflow (Multi-Pod / Preemptible)

### Smoke-test data prep (fast pipeline check, ~minutes)

```bash
# Produces 5M train tokens + 200K val tokens instead of full targets
python data/prepare_fineweb.py --smoke_test
python data/prepare_stories.py --smoke_test
```

Everything else (tokenisation, EOS insertion, shard writing, val carve-out) is
identical — only the token targets are overridden at runtime.

### Push shards to HF Hub (after full prep)

```bash
# Set HF_SHARD_REPO in .env first (e.g. myuser/cdx144m-data)
python data/push_to_hub.py
# Optional overrides:
# python data/push_to_hub.py --repo myuser/cdx144m-data --shard_dir data/shards
```

### Pull shards on a new pod (before training)

```bash
python data/pull_from_hub.py
# Optional override:
# python data/pull_from_hub.py --repo myuser/cdx144m-data
```

### Checkpoint push (automatic, non-blocking)

Set `HF_CHECKPOINT_REPO` in `.env` (can be the same repo as `HF_SHARD_REPO`):

```bash
# .env
HF_CHECKPOINT_REPO=myuser/cdx144m-data
```

Once set, `CheckpointManager.save()` spawns a background daemon thread after
each `torch.save()`. The upload runs concurrently — the training loop never waits.
Both `step_*.pt` and `best_val.pt` are uploaded under `checkpoints/` in the repo.
Checkpoints that get pruned locally by the rolling-window logic are **not** pushed.

### Pull latest checkpoint on a new pod (before resuming)

```bash
# Pull the specific checkpoint you want to resume from:
python -c "
from huggingface_hub import hf_hub_download
import os, dotenv
dotenv.load_dotenv()
path = hf_hub_download(
    repo_id=os.environ['HF_CHECKPOINT_REPO'],
    repo_type='dataset',
    filename='checkpoints/step_009536.pt',  # or best_val.pt
    local_dir='.',
)
print('Downloaded to:', path)
"

# Then resume training:
python train.py --resume checkpoints/step_009536.pt
```

> **Tip:** To list all available checkpoints in the repo:
> ```bash
> python -c "
> from huggingface_hub import HfApi; import os, dotenv; dotenv.load_dotenv()
> files = HfApi().list_repo_files(os.environ['HF_CHECKPOINT_REPO'], repo_type='dataset')
> print([f for f in files if f.startswith('checkpoints/')])
> "
> ```

### Full new-pod session sequence

```bash
# 1. Install deps + set tokens
pip install -r requirements.txt
cp .env .env.bak  # keep a copy; fill in HF_TOKEN, HF_SHARD_REPO, HF_CHECKPOINT_REPO

# 2. Pull shards (~GB download)
python data/pull_from_hub.py

# 3. Pull latest checkpoint
python -c "
from huggingface_hub import hf_hub_download
import os, dotenv; dotenv.load_dotenv()
hf_hub_download(os.environ['HF_CHECKPOINT_REPO'], repo_type='dataset',
    filename='checkpoints/best_val.pt', local_dir='.')
"

# 4. Resume training (checkpoints auto-push in background)
python train.py --resume checkpoints/best_val.pt
```

---

## Smoke-Test Pass Criterion

```
final_loss < ln(32000) ≈ 10.37
```

If loss doesn't drop below the random baseline in 100 steps, training is diverging.  
Common causes: bad weight init, LR too high, data loading returning garbage.

---

## SFT (Stage 2)

SFT scaffolding is planned for a separate pass after pretraining is validated and complete.  
See `project_plan2.md` Section 4 for the target dataset spec (20k–30k rows, persona sub-set, EOS-terminated).

