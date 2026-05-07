"""
LSCP Stage 3-1: Baseline Evaluation
=====================================
No training. Measures baseline PPL and perturbation gap.
Saves cache for use by 3-2 and 3-3.

Backend: PyTorch + Transformers + PEFT (DGX Spark)

Usage:
    python lscp3_1.py
"""

import time, json, math, numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from config import *

STAGE2_FILE = RESULTS_DIR / "stage2_results.json"
OUTPUT_FILE = RESULTS_DIR / "baseline_cache.json"

RETAIN_TEXTS = [
    "Water is a chemical compound with the formula H2O. Each molecule consists of one oxygen atom covalently bonded to two hydrogen atoms.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second, a fundamental constant in physics denoted by the letter c.",
    "Humans have 46 chromosomes arranged in 23 pairs. Each parent contributes one chromosome to each pair through sexual reproduction.",
    "Paris is the capital and largest city of France, situated on the river Seine in the north-central part of the country.",
    "William Shakespeare wrote Romeo and Juliet around 1594 to 1596. It is one of the most famous love tragedies in English literature.",
    "Diamond is the hardest known natural material, scoring 10 on the Mohs hardness scale. It is an allotrope of carbon formed under high pressure.",
    "World War II ended in 1945. Germany surrendered in May and Japan surrendered in August after the atomic bombings of Hiroshima and Nagasaki.",
    "Mercury is the closest planet to the Sun and the smallest planet in the Solar System. Its orbital period is approximately 88 Earth days.",
    "The chemical symbol for gold is Au, derived from the Latin word aurum. Gold is a dense, soft, malleable precious metal with atomic number 79.",
    "Leonardo da Vinci painted the Mona Lisa, believed to depict Lisa Gherardini. It is displayed in the Louvre Museum in Paris.",
]


# ── Model ─────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        dtype=DTYPE,
    )
    print(f"  Model loaded")
    return model, tokenizer


def apply_lora(model):
    from peft import get_peft_model, LoraConfig, TaskType

    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    try:
        num_layers = model.config.num_hidden_layers
    except AttributeError:
        try:
            num_layers = model.config.text_config.num_hidden_layers
        except AttributeError:
            num_layers = NUM_LAYERS  # config.py에서 직접 지정
    target_layers = list(range(num_layers - LORA_LAYERS, num_layers))

    config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 20,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        layers_to_transform=target_layers,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_adapters(model, path):
    model.save_pretrained(str(path))


# ── Perplexity ────────────────────────────────────────────────────

@torch.no_grad()
def measure_perplexity(model, tokenizer, text):
    tokens = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    if tokens.shape[1] < 2:
        return 0.0
    logits = model(tokens).logits[0]
    shift_logits = logits[:-1]
    shift_labels = tokens[0, 1:]
    loss = F.cross_entropy(shift_logits, shift_labels)
    return round(math.exp(loss.item()), 4)


def measure_batch(model, tokenizer, texts, label=""):
    model.eval()
    results = []
    for idx, text in enumerate(texts):
        print(f"\r    [{label}] {idx+1}/{len(texts)}", end="", flush=True)
        results.append(measure_perplexity(model, tokenizer, text))
    mean_ppl = round(float(np.mean(results)), 4)
    print(f"\r    [{label}] done, mean={mean_ppl:.2f}          ")
    return mean_ppl, results


def perturbation_test(model, tokenizer, originals, paraphrases, label=""):
    model.eval()
    detail = []
    for topic, orig_text in originals:
        para_text = paraphrases.get(topic)
        if not para_text:
            continue
        ppl_orig = measure_perplexity(model, tokenizer, orig_text)
        ppl_para = measure_perplexity(model, tokenizer, para_text)
        gap = ppl_para / ppl_orig if ppl_orig > 0 else 0
        detail.append({"topic": topic, "ppl_original": ppl_orig,
                       "ppl_paraphrase": ppl_para, "gap": round(gap, 3)})
    mean_gap = round(float(np.mean([d["gap"] for d in detail])), 3) if detail else 0
    if detail:
        print(f"    [{label}] perturbation: {len(detail)} passages, mean_gap={mean_gap:.3f}")
    return mean_gap, detail


# ── Main ──────────────────────────────────────────────────────────

def main():
    t_global = time.time()

    # Load data
    import importlib
    mod = importlib.import_module(PASSAGE_FILE)
    text_map = {p["topic"]: p["text"] for p in mod.PASSAGES}
    label_map = {p["topic"]: p["label"] for p in mod.PASSAGES}

    s1_file = RESULTS_DIR / "stage1_results.json"
    with open(s1_file) as f:
        s1_data = json.load(f)
    s1_threshold = s1_data.get("analysis", {}).get("threshold", 2.0)
    all_flagged = [(r["topic"], text_map[r["topic"]])
                   for r in s1_data.get("results", [])
                   if r["S_k"] > s1_threshold and r["topic"] in text_map]

    paraphrase_map = {}
    try:
        para_mod = importlib.import_module("paraphrases")
        paraphrase_map = {p["topic"]: p["text"] for p in para_mod.PARAPHRASES}
        print(f"  Paraphrases: {len(paraphrase_map)} loaded")
    except (ImportError, ModuleNotFoundError):
        print(f"  Paraphrases: not found")

    target_texts = all_flagged
    known_texts = [(t, text_map[t]) for t in text_map
                   if label_map.get(t) == "known"][:10]

    print(f"  Eval: {len(target_texts)} target, {len(known_texts)} aligned, "
          f"{len(RETAIN_TEXTS)} retain")

    # Load model + LoRA init
    print(f"\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"  Ready in {time.time()-t0:.1f}s")

    print(f"\nApplying LoRA...")
    model = apply_lora(model)

    # Save init adapters for 3-2 and 3-3
    init_path = RESULTS_DIR / "adapters_init"
    save_adapters(model, init_path)
    print(f"  Init adapters saved: {init_path}")

    # ── Evaluate ──────────────────────────────────────────────────
    print(f"\n{'='*70}\nBASELINE EVALUATION\n{'='*70}")
    model.eval()

    a_t, a_td = measure_batch(model, tokenizer, [t[1] for t in target_texts], "target")
    a_a, a_ad = measure_batch(model, tokenizer, [t[1] for t in known_texts], "known")
    a_r, a_rd = measure_batch(model, tokenizer, RETAIN_TEXTS, "retain")

    a_pt_gap, a_pt_detail = (0, [])
    if paraphrase_map:
        a_pt_gap, a_pt_detail = perturbation_test(
            model, tokenizer, target_texts, paraphrase_map, "baseline")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}\nBASELINE SUMMARY\n{'='*70}")
    print(f"  Target PPL:  {a_t:.2f}")
    print(f"  Known PPL:   {a_a:.2f}")
    print(f"  Retain PPL:  {a_r:.2f}")
    print(f"  Pert gap:    {a_pt_gap:.3f}")

    output = {
        "target_ppl": a_t, "known_ppl": a_a, "retain_ppl": a_r,
        "target_detail": a_td, "aligned_detail": a_ad, "retain_detail": a_rd,
        "perturbation_gap": a_pt_gap, "perturbation_detail": a_pt_detail,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Total: {(time.time()-t_global)/60:.1f} min")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
