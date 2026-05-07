# Autolearn: Learn by Surprise, Commit by Proof

A self-gated post-training framework for autonomous knowledge acquisition in language models.

Autolearn enables language models to *learn from what they read*—selectively, verifiably, and without external supervision. It detects what a model does not already know (via surprisal), verifies[...]  

**Paper**: [Autolearn: Learn by Surprise, Commit by Proof](https://arxiv.org/abs/2604.01951)

## How It Works

Autolearn operates in three stages:

**Stage 1 — Detect and Ground.** Compute per-token surprisal for each passage. Flag passages exceeding a threshold (μ + λσ) as surprising. This is a single forward pass with zero additional c[...]  

**Stage 2 — Verify, Grade and Annotate.** For each flagged passage, generate a Q&A chain tagged by epistemic origin (`[existing]`, `[mechanism]`, `[implication]`). Check each answer against the [...]  

**Stage 3 — Gated Weight Update.** Adjust AdamW's β₂ proportionally to *k* via β₂ = 0.999 · rᵏ. When *k* = 0, β₂ = 0.999 (standard AdamW, no learning). As *k* increases, β₂ decrea[...]  

## Key Results

- Q&A-format training suppresses token-sequence memorization (perturbation gap below pre-trained baseline, >10 sigma) while enabling knowledge acquisition that standard fine-tuning does not achieve on genuinely novel facts.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+
- PEFT 0.10+
- GPU with 32GB+ VRAM (48GB+ recommended for 14B models)

Install dependencies:

```bash
pip install torch transformers peft safetensors
```

## Repository Structure

```
autolearn/
├── config.py                  # Model, LoRA, and training configuration
├── passages.py                # 60 passages (20 novel, 20 corrupt, 20 known)
├── paraphrases.py             # Semantically equivalent paraphrases for perturbation gap
├── lscp1.py                   # Stage 1: Surprisal-based detection
├── lscp2_1.py                 # Stage 2: Q&A chain generation
├── lscp2_2.py                 # Stage 2: Consistency checking + conviction depth
├── lscp3_1.py                 # Stage 3: Baseline (pre-trained adapter initialization)
├── lscp3_2.py                 # Stage 3: SFT (standard fine-tuning on raw text)
├── lscp3_3.py                 # Stage 3: Autolearn (Q&A training with gated weight update)
├── lscp3_eval.py              # Factual evaluation (perturbation gap + PPL + keyword D.Cor)
├── test_qa_v2.py              # 78 test questions (5 categories, demonstration
└── results/                   # Output directory for adapters and metrics
```

## Quick Start

### Configure

Edit `config.py` to set your model and paths:

```python
MODEL_NAME = "Qwen/Qwen3-14B"
DTYPE = torch.float16
DEVICE = "cuda"
LORA_RANK = 8
LORA_LAYERS = 8
PASSAGE_FILE = "passages"
```

### 2. Stage 1: Detection

```bash
python lscp1.py
```

Computes per-passage surprisal and flags passages above threshold. Outputs `results/stage1_results.json`.

### 3. Stage 2: Self-Verification

```bash
python lscp2_1.py    # Generate Q&A chains
python lscp2_2.py    # Consistency checking, compute conviction depth k
```

Produces verified Q&A pairs with conviction depth for each flagged passage. Outputs `results/stage2_results.json`.

### 4. Stage 3: Training

**Initialize baseline adapter:**

```bash
python lscp3_1.py
```

**SFT baseline (raw text training):**

```bash
python lscp3_2.py --lr 1e-5 --epochs 15 --auto-stop
```

**Autolearn (Q&A training with gated weight update):**

```bash
python lscp3_3.py --lr 1e-5 --r 0.999 --epochs 15 --auto-stop
```

**Autolearn without beta2 gating (Q&A format only):**

```bash
python lscp3_3.py --lr 1e-5 --r 1.0 --epochs 15 --auto-stop
```

The `--auto-stop` flag monitors the perturbation gap and halts training when it rises for two consecutive epochs.

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 0.98 | Decay factor for β₂ schedule |
| `LAMBDA` | 2.0 | Surprisal threshold (μ + λσ) |
| `QA_SCALE` | 3.0 | Scaling constant *c* for N = ⌈S_i · c⌉ |
| `LEARNING_RATE` | 1e-5 | AdamW learning rate |
| `EPOCHS` | 3 | Training epochs per passage |
| `LORA_RANK` | 8 | LoRA adapter rank |

## Output

Stage 3 produces `stage3_results_r{R}.json` containing:

- Baseline, Normal and Autolearn perplexity (target, known, retain)
- Perturbation gap (paraphrase PPL / original PPL)
- Five-way Q&A accuracy (novel-direct, novel-adjacent, corrupt-direct, corrupt-adjacent, unrelated)
- Training metadata (items, time, loss curves)


## Citation

```bibtex
@article{choi2026auto,
  title={Autolearn: Learn by Surprise, Commit by Proof},
  author={Choi, Kang-Sin},
  year={2026}
}
```

## License

MIT
