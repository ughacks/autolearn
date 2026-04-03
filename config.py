"""
LSCP Configuration
==================
Central configuration for all LSCP pipeline stages.
Edit this file to change model, paths, and hyperparameters.

Usage:
    from config import *
"""
from pathlib import Path

# -- Model ---------------------------------------------------------
MODEL_NAME = "mlx-community/Qwen3-14B-4bit"
ADAPTER_PATH = None       # Set to e.g. "results/adapters.npz" for cycle 2+

# -- Paths ---------------------------------------------------------
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
PASSAGE_FILE = "passages"

# -- Stage 1 -------------------------------------------------------
LAMBDA = 2.0              # threshold = mu + lambda * sigma

# -- Stage 2 -------------------------------------------------------
MIN_QA = 5
MAX_QA = 15
QA_GEN_TEMPERATURE = 0.7
CONSISTENCY_TEMPERATURE = 0.1

# -- Stage 3 -------------------------------------------------------
R = 0.98                  # decay factor for beta2 = 0.999 * r^k
LORA_RANK = 8
LORA_LAYERS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
N_EPOCHS = 3
BETA2_DEFAULT = 0.999
BETA2_FLOOR = 0.01
ALPHA_DROP = 0.0          # d_k adjustment (0 = disabled)
BETA_MIN = BETA2_FLOOR
