"""
LSCP Stage 3: Gated Weight Update via QLoRA
=================================================
Compares baseline (no training) vs LSCP (beta2 = 0.999 * r^k):
  - r=1.0 is equivalent to Standard (uniform beta2=0.999)
  - r<1.0 opens the Variance Lock proportional to verification depth
  - --no-passage skips source windows, training on Q&A + strangeness only

Evaluation via PERPLEXITY (no circular Q&A):
  - Target passages: should decrease (model learned new content)
  - Aligned passages: should stay same (no forgetting)
  - Retain texts: should stay same (general knowledge intact)

Usage:
    python lscp3.py              # default r=0.98, with passage
    python lscp3.py 0.95         # custom r
    python lscp3.py --no-passage # Q&A only, no passage text
    python lscp3.py --no-passage 0.95
    python lscp3.py --eval adapters.npz
"""

import time, json, math, numpy as np
from pathlib import Path

from config import *

STAGE2_FILE = RESULTS_DIR / "stage2_results.json"

# Usage: python3 lscp3.py [r]                    (train + eval)
#        python3 lscp3.py 0.95                    (train with r=0.95)
#        python3 lscp3.py --no-passage             (Q&A only, skip source windows)
#        python3 lscp3.py --no-passage 0.95        (combine flags)
#        python3 lscp3.py --eval adapters.npz     (eval only, no training)
import sys

EVAL_ONLY = False
EVAL_ADAPTER = None
SKIP_SOURCE_WINDOW = False

args = [a for a in sys.argv[1:]]
if "--eval" in args:
    EVAL_ONLY = True
    idx = args.index("--eval")
    args.pop(idx)
    if idx < len(args):
        EVAL_ADAPTER = args.pop(idx)
    else:
        print("ERROR: --eval requires adapter path (e.g. --eval lscp_results/adapters_r095_run1.npz)")
        sys.exit(1)
if "--no-passage" in args:
    SKIP_SOURCE_WINDOW = True
    args.remove("--no-passage")

R = float(args[0]) if args else 0.98
ALPHA_DROP = 0.3
BETA_MIN = 0.01

_r_str = str(R).replace('.', '')
_nopass_str = "_nopass" if SKIP_SOURCE_WINDOW else ""
if EVAL_ONLY:
    _adapter_name = Path(EVAL_ADAPTER).stem
    OUTPUT_FILE = RESULTS_DIR / f"eval_{_adapter_name}.json"
else:
    OUTPUT_FILE = RESULTS_DIR / f"stage3_results_r{_r_str}{_nopass_str}.json"

EPOCHS = 3
GRAD_CLIP = 1.0       # Max gradient norm — prevents update explosion

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

def load_model():
    from mlx_lm import load
    print(f"  Loading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)
    print(f"  Model loaded")
    return model, tokenizer

def apply_lora(model):
    try:
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError:
        try:
            from mlx_lm.tuner import linear_to_lora_layers
        except ImportError:
            from mlx_lm.lora import linear_to_lora_layers
    from mlx.utils import tree_flatten
    model.freeze()
    linear_to_lora_layers(model, LORA_LAYERS, {"rank": LORA_RANK, "scale": 20.0, "dropout": 0.0})
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    n_total = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"  LoRA: {n_train:,} trainable / {n_total:,} total ({n_train/n_total:.4%})")
    model.train()
    return model

def save_adapters(model, path):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    mx.savez(str(path), **dict(tree_flatten(model.trainable_parameters())))

def load_adapters(model, path):
    import mlx.core as mx
    model.load_weights(list(mx.load(str(path)).items()), strict=False)

def measure_perplexity(model, tokenizer, text):
    import mlx.core as mx
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array(tokens[:-1])[None]
    y = mx.array(tokens[1:])
    logits = model(x)[0]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = [-log_probs[i, y[i]].item() for i in range(len(y))]
    return round(math.exp(np.mean(nll)), 4)

def measure_batch(model, tokenizer, texts, label=""):
    results = []
    for idx, text in enumerate(texts):
        print(f"\r    [{label}] {idx+1}/{len(texts)}", end="", flush=True)
        results.append(measure_perplexity(model, tokenizer, text))
    mean_ppl = round(float(np.mean(results)), 4)
    print(f"\r    [{label}] done, mean={mean_ppl:.2f}          ")
    return mean_ppl, results

def perturbation_test(model, tokenizer, originals, paraphrases, label=""):
    """Measure memorization gap: PPL(paraphrase) / PPL(original).

    Gap ≈ 1.0 → understanding (both expressions predicted well)
    Gap >> 1.0 → memorization (only original tokens predicted)

    Args:
        originals:  list of (topic, text) for original passages
        paraphrases: dict of {topic: paraphrase_text}
    Returns:
        mean_gap, detail list
    """
    detail = []
    for topic, orig_text in originals:
        para_text = paraphrases.get(topic)
        if not para_text:
            continue
        ppl_orig = measure_perplexity(model, tokenizer, orig_text)
        ppl_para = measure_perplexity(model, tokenizer, para_text)
        gap = ppl_para / ppl_orig if ppl_orig > 0 else 0
        detail.append({
            "topic": topic,
            "ppl_original": ppl_orig,
            "ppl_paraphrase": ppl_para,
            "gap": round(gap, 3),
        })
    if detail:
        mean_gap = round(float(np.mean([d["gap"] for d in detail])), 3)
        print(f"    [{label}] perturbation: {len(detail)} passages, mean_gap={mean_gap:.3f}")
    else:
        mean_gap = 0
        print(f"    [{label}] perturbation: no paraphrases available")
    return mean_gap, detail

def generate_answer(model, tokenizer, question, max_tokens=100):
    """Generate an answer using mlx_lm.generate."""
    from mlx_lm import generate
    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.0)
        kwargs = {"sampler": sampler}
    except (ImportError, TypeError):
        kwargs = {}
    prompt = f"Q: {question}\nA:"
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=max_tokens, verbose=False, **kwargs).strip()

def evaluate_test_qa(model, tokenizer, test_qa, label=""):
    """Evaluate on human-authored test Q&A.
    Returns (overall_acc, per_field_acc, details).
    per_field_acc is a dict: {field: {label: accuracy}}.
    """
    results = []
    for idx, item in enumerate(test_qa):
        print(f"\r    [{label}] {idx+1}/{len(test_qa)}", end="", flush=True)
        response = generate_answer(model, tokenizer, item["question"])

        expected_lower = item["answer"].lower()
        response_lower = response.lower()
        key_words = [w.strip(".,!?;:'\"()") for w in expected_lower.split()
                     if len(w.strip(".,!?;:'\"()")) > 3]
        if key_words:
            matches = sum(1 for kw in key_words if kw in response_lower)
            score = matches / len(key_words)
        else:
            score = 1.0 if expected_lower[:10] in response_lower else 0.0

        results.append({
            "topic": item["topic"], "label": item["label"],
            "field": item.get("field", "direct"), "type": item["type"],
            "question": item["question"], "expected": item["answer"],
            "response": response[:200], "score": round(score, 3),
            "correct": score >= 0.5,
        })

    overall = sum(1 for r in results if r["correct"]) / len(results) if results else 0

    # Per-field, per-label accuracy
    per_field = {}
    for field in ["direct", "adjacent", "unrelated"]:
        per_field[field] = {}
        for lbl in ["novel", "corrupt", "unrelated"]:
            items = [r for r in results if r["field"] == field and r["label"] == lbl]
            if items:
                per_field[field][lbl] = round(sum(1 for r in items if r["correct"]) / len(items), 4)

    # Legacy: flat novel/corrupt acc (direct only for backward compat)
    novel_items = [r for r in results if r["label"] == "novel"]
    corrupt_items = [r for r in results if r["label"] == "corrupt"]
    novel_acc = sum(1 for r in novel_items if r["correct"]) / len(novel_items) if novel_items else 0
    corrupt_acc = sum(1 for r in corrupt_items if r["correct"]) / len(corrupt_items) if corrupt_items else 0

    # Print summary
    print(f"\r    [{label}] done: overall={overall:.0%} novel={novel_acc:.0%} corrupt={corrupt_acc:.0%}")
    for field, accs in per_field.items():
        parts = [f"{lbl}={acc:.0%}" for lbl, acc in accs.items()]
        if parts:
            print(f"      {field}: {', '.join(parts)}")

    return round(overall, 4), round(novel_acc, 4), round(corrupt_acc, 4), per_field, results

def _apply_grads_and_eval(model, optimizer, loss, grads):
    """Shared gradient clipping, optimizer step, and eval."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map
    if GRAD_CLIP > 0:
        norms = tree_flatten(tree_map(lambda g: (g * g).sum(), grads))
        total_norm = mx.sqrt(sum(v for _, v in norms))
        scale = mx.minimum(mx.array(GRAD_CLIP) / (total_norm + 1e-8), mx.array(1.0))
        grads = tree_map(lambda g: g * scale, grads)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    lv = loss.item()
    if math.isnan(lv):
        print(f"\n  WARNING: NaN loss detected")
        return 0.0
    return lv


def train_on_text(model, optimizer, tokenizer, text):
    """Full next-token prediction on the entire text.
    Used for source_window items: the raw passage text is the knowledge
    source, so every token is a valid prediction target -- identical to
    the pretraining objective.
    """
    import mlx.core as mx
    import mlx.nn as nn
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return 0.0
    x = mx.array(tokens[:-1])[None]
    y = mx.array(tokens[1:])[None]
    def loss_fn(model):
        logits = model(x)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        return nn.losses.cross_entropy(logits_flat, y.reshape(-1)).mean()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    return _apply_grads_and_eval(model, optimizer, loss, grads)


def train_on_qa(model, optimizer, tokenizer, context, question, answer):
    """Answer-only masked training for Q&A items.

    Text layout fed to the model:
        [context]  [Q: question\nA: ]  [answer]
          ^-- no loss    ^-- no loss       ^-- loss here only

    The context is the source_window excerpt for the passage.
    Prepending it ensures the answer is learned in the presence of the
    new knowledge, not in a vacuum.

    Why answer-only masking:
        The question and context are already known (or are just framing).
        Backpropagating through them would update weights for tokens the
        model already predicts well, diluting the learning signal for the
        answer tokens that carry the new information.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prefix = (context.strip() + "\n") if context else ""
    q_text = f"Q: {question}\nA: "
    full_text = prefix + q_text + answer

    # Count tokens in the non-answer prefix to compute mask boundary
    prefix_tokens = tokenizer.encode(prefix + q_text)
    full_tokens   = tokenizer.encode(full_text)

    if len(full_tokens) < 2:
        return 0.0

    answer_start = len(prefix_tokens)  # first token index belonging to answer

    x = mx.array(full_tokens[:-1])[None]
    y = mx.array(full_tokens[1:])

    def loss_fn(model):
        logits = model(x)       # (1, T, vocab)
        logits_flat = logits[0] # (T, vocab)
        T = logits_flat.shape[0]

        per_token_loss = nn.losses.cross_entropy(
            logits_flat, y, reduction="none"
        )   # (T,)

        # y[i] predicts token i+1; answer tokens in y start at answer_start-1
        mask_start = max(0, answer_start - 1)
        mask = mx.concatenate([
            mx.zeros((mask_start,)),
            mx.ones((T - mask_start,)),
        ])  # (T,)

        masked = per_token_loss * mask
        n_answer_tokens = float(max(1, T - mask_start))
        return masked.sum() / n_answer_tokens

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    return _apply_grads_and_eval(model, optimizer, loss, grads)

def compute_beta2(k, d_k):
    """β₂ = max(0.999 · r^k · (1 - α·d_k), β_min). Computed at training time."""
    base = 0.999 * (R ** k)
    adjusted = max(base * (1.0 - ALPHA_DROP * max(0, d_k)), BETA_MIN)
    return round(adjusted, 6)

def build_source_window(tokenizer, text, peaks, w_ext=50):
    """Extract high-surprisal region of passage for training."""
    if not peaks:
        return text
    peak_positions = [p["pos"] for p in peaks]
    min_peak, max_peak = min(peak_positions), max(peak_positions)
    tokens = tokenizer.encode(text)
    window_start = max(0, min_peak - w_ext)
    window_end = min(max_peak + w_ext + 1, len(tokens))
    window_tokens = tokens[window_start:window_end]
    if len(window_tokens) >= len(tokens) * 0.8:
        return text
    return tokenizer.decode(window_tokens)

def prepare_training_data(s2, text_map=None, source_map=None, s1_by_topic=None, tokenizer=None):
    """Build the training item list from Stage 2 output.

    beta2 is computed per training item using k (conviction depth at that item).
    Earlier questions get higher beta2 (weaker learning); later questions
    that survived more verification get lower beta2 (stronger learning).
    All item types (qa, strangeness, source_window) use the same k field.

    Source window text is resolved from passages.py + stage1 peaks if not
    already present in the item (pointer-only format from lscp2_2.py).
    """
    # Resolve source window texts
    source_window_by_topic = {}
    for pair in s2.get("accept", []):
        if pair.get("type") == "source_window":
            topic = pair.get("topic", "")
            if not topic or topic in source_window_by_topic:
                continue
            if "text" in pair and pair["text"]:
                # Legacy format: text already present
                source_window_by_topic[topic] = pair["text"]
            elif text_map and topic in text_map:
                # Pointer format: resolve from passages + peaks
                raw_text = text_map[topic]
                source = source_map.get(topic, "") if source_map else ""
                peaks = s1_by_topic.get(topic, {}).get("peaks", []) if s1_by_topic else []
                sw_text = build_source_window(tokenizer, raw_text, peaks) if tokenizer else raw_text
                if source:
                    sw_text = f"[Source: {source}]\n\n{sw_text}"
                source_window_by_topic[topic] = sw_text

    items = []

    for pair in s2.get("accept", []):
        ptype = pair.get("type", "")
        tier  = pair.get("tier", "")
        topic = pair.get("topic", "")
        d_k   = pair.get("d_k", 0)
        k     = pair.get("k", 0)
        beta2 = compute_beta2(k, d_k)

        if ptype == "source_window":
            if SKIP_SOURCE_WINDOW:
                continue  # skip passage text, learn only from Q&A + strangeness
            sw_text = source_window_by_topic.get(topic, "")
            if not sw_text:
                continue
            items.append({
                "train_mode": "full",
                "text": sw_text,
                "beta2": beta2,
                "topic": topic, "type": "source_window", "tier": "",
                "k": k, "d_k": d_k,
            })

        elif ptype == "qa" and tier in ("mechanism", "implication"):
            context = source_window_by_topic.get(topic, "")
            items.append({
                "train_mode": "qa_masked",
                "context":  context,
                "question": pair["question"],
                "answer":   pair["answer"],
                "beta2": beta2,
                "topic": topic, "type": "qa", "tier": tier,
                "k": k, "d_k": d_k,
            })

        elif ptype == "strangeness":
            context = source_window_by_topic.get(topic, "")
            items.append({
                "train_mode": "qa_masked",
                "context":  context,
                "question": pair["question"],
                "answer":   pair["answer"],
                "beta2": beta2,
                "topic": topic, "type": "strangeness", "tier": "strangeness",
                "k": k, "d_k": d_k,
            })

    return items

def _train_item(model, optimizer, tokenizer, item):
    """Dispatch to the correct training function based on train_mode."""
    mode = item.get("train_mode", "full")
    if mode == "qa_masked":
        return train_on_qa(
            model, optimizer, tokenizer,
            context=item.get("context", ""),
            question=item["question"],
            answer=item["answer"],
        )
    else:  # "full" — source_window and any legacy items
        return train_on_text(model, optimizer, tokenizer, item["text"])


def main():
    import mlx.core as mx
    import mlx.optimizers as optim
    t_global = time.time()

    print(f"Loading Stage 2 results...")
    with open(STAGE2_FILE) as f:
        s2 = json.load(f)

    # Model consistency check
    s2_model = s2.get("model", "")
    if s2_model and s2_model != MODEL_NAME:
        print(f"\n  *** WARNING: Stage 2 used model '{s2_model}' but Stage 3 uses '{MODEL_NAME}' ***")
        print(f"  *** All stages must use the same model for valid results ***\n")

    import importlib
    mod = importlib.import_module(PASSAGE_FILE)
    text_map = {p["topic"]: p["text"] for p in mod.PASSAGES}
    label_map = {p["topic"]: p["label"] for p in mod.PASSAGES}
    source_map = {p["topic"]: p.get("source", "") for p in mod.PASSAGES}

    # Load Stage 1 results for peaks (source window construction)
    s1_by_topic = {}
    s1_file = RESULTS_DIR / "stage1_results.json"
    if s1_file.exists():
        with open(s1_file) as f:
            s1 = json.load(f)
        s1_by_topic = {r["topic"]: r for r in s1.get("results", [])}

    # Load paraphrases for perturbation test (optional)
    paraphrase_map = {}
    try:
        para_mod = importlib.import_module("paraphrases")
        paraphrase_map = {p["topic"]: p["text"] for p in para_mod.PARAPHRASES}
        print(f"  Paraphrases: {len(paraphrase_map)} loaded")
    except (ImportError, ModuleNotFoundError):
        print(f"  Paraphrases: not found (skipping perturbation test)")

    # Load model (needed for tokenizer in source window + training)
    print(f"\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"  Ready in {time.time()-t0:.1f}s")

    train_items = prepare_training_data(s2, text_map=text_map, source_map=source_map,
                                         s1_by_topic=s1_by_topic, tokenizer=tokenizer)
    betas = [t["beta2"] for t in train_items]
    n_sw     = sum(1 for t in train_items if t.get("type") == "source_window")
    n_qa_mec = sum(1 for t in train_items if t.get("type") == "qa" and t.get("tier") == "mechanism")
    n_qa_imp = sum(1 for t in train_items if t.get("type") == "qa" and t.get("tier") == "implication")
    if SKIP_SOURCE_WINDOW:
        print(f"  ** NO-PASSAGE MODE: source windows skipped, Q&A + strangeness only **")
    print(f"  r={R}, beta2 range: [{min(betas):.4f}, {max(betas):.4f}]")
    print(f"  Training items: {len(train_items)} (source_window={n_sw}, mechanism_qa={n_qa_mec}, implication_qa={n_qa_imp})")

    flagged_topics = set()
    for d in s2.get("detail", []):
        if d.get("break_point") is None and d.get("n_passed", 0) > 0:
            flagged_topics.add(d["topic"])

    # All flagged passages (for Normal: train on raw text without filtering)
    all_flagged = [(d["topic"], text_map[d["topic"]]) for d in s2.get("detail", []) if d["topic"] in text_map]
    # Target = all flagged (for PPL evaluation — we want to measure all, not just accepted)
    target_texts = all_flagged
    known_texts = [(t, text_map[t]) for t in text_map if label_map.get(t) == "known" and t not in {t for t,_ in all_flagged}]
    if len(known_texts) < 5:
        known_texts = [(t, text_map[t]) for t in text_map if label_map.get(t) == "known"][:10]

    print(f"  Eval: {len(target_texts)} target, {len(known_texts)} aligned, {len(RETAIN_TEXTS)} retain")

    # Load human-authored test Q&A (independent evaluation)
    from test_qa import TEST_QA
    print(f"  Test Q&A: {len(TEST_QA)} questions (human-authored, independent)")

    print(f"\nApplying LoRA...")
    model = apply_lora(model)
    if ADAPTER_PATH and Path(ADAPTER_PATH).exists():
        load_adapters(model, ADAPTER_PATH)
        print(f"  Previous adapter loaded: {ADAPTER_PATH}")
    init_path = RESULTS_DIR / "adapters_init.npz"
    save_adapters(model, init_path)

    # EVAL-ONLY MODE: load adapter, run all evaluations, save, exit
    if EVAL_ONLY:
        print(f"\n{'='*70}\nEVAL-ONLY: {EVAL_ADAPTER}\n{'='*70}")
        load_adapters(model, EVAL_ADAPTER)
        model.eval()
        print(f"  Adapter loaded: {EVAL_ADAPTER}")

        e_t, e_td = measure_batch(model, tokenizer, [t[1] for t in target_texts], "target")
        e_a, e_ad = measure_batch(model, tokenizer, [t[1] for t in known_texts], "known")
        e_r, e_rd = measure_batch(model, tokenizer, RETAIN_TEXTS, "retain")
        print(f"  Target={e_t:.2f}  Aligned={e_a:.2f}  Retain={e_r:.2f}")

        print(f"  Evaluating test Q&A...")
        e_qa_all, e_qa_novel, e_qa_corrupt, e_qa_fields, e_qa_detail = evaluate_test_qa(model, tokenizer, TEST_QA, "test-QA")

        if paraphrase_map:
            e_pt_gap, e_pt_detail = perturbation_test(model, tokenizer, target_texts, paraphrase_map, "eval")
        else:
            e_pt_gap, e_pt_detail = 0, []

        output = {
            "mode": "eval_only", "adapter": EVAL_ADAPTER,
            "model": MODEL_NAME, "r": R,
            "target_ppl": e_t, "known_ppl": e_a, "retain_ppl": e_r,
            "target_detail": e_td, "aligned_detail": e_ad, "retain_detail": e_rd,
            "test_qa_overall": e_qa_all, "test_qa_novel": e_qa_novel,
            "test_qa_corrupt": e_qa_corrupt, "test_qa_fields": e_qa_fields,
            "test_qa_detail": e_qa_detail,
            "perturbation_gap": e_pt_gap, "perturbation_detail": e_pt_detail,
            "target_topics": [t[0] for t in target_texts],
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Saved: {OUTPUT_FILE}")
        print(f"\n  Total: {(time.time()-t_global)/60:.1f} min")
        return


    # BASELINE — measure once, cache for reuse across r values
    baseline_file = RESULTS_DIR / "baseline_cache.json"
    if baseline_file.exists():
        print(f"\n  Loading cached baseline from {baseline_file}")
        with open(baseline_file) as f:
            bl_cache = json.load(f)
        a_t = bl_cache["target_ppl"]; a_td = bl_cache["target_detail"]
        a_a = bl_cache["known_ppl"]; a_ad = bl_cache["aligned_detail"]
        a_r = bl_cache["retain_ppl"]; a_rd = bl_cache["retain_detail"]
        a_qa_all = bl_cache["test_qa_overall"]
        a_qa_novel = bl_cache["test_qa_novel"]
        a_qa_corrupt = bl_cache["test_qa_corrupt"]
        a_qa_detail = bl_cache["test_qa_detail"]
        a_qa_fields = bl_cache.get("test_qa_fields", {})
        print(f"  Target={a_t:.2f}  Aligned={a_a:.2f}  Retain={a_r:.2f}")
        print(f"  Test Q&A: overall={a_qa_all:.0%} novel={a_qa_novel:.0%} corrupt={a_qa_corrupt:.0%}")
    else:
        print(f"\n{'='*70}\nBASELINE (no training)\n{'='*70}")
        model.eval()
        a_t, a_td = measure_batch(model, tokenizer, [t[1] for t in target_texts], "target")
        a_a, a_ad = measure_batch(model, tokenizer, [t[1] for t in known_texts], "known")
        a_r, a_rd = measure_batch(model, tokenizer, RETAIN_TEXTS, "retain")
        print(f"  Target={a_t:.2f}  Aligned={a_a:.2f}  Retain={a_r:.2f}")
        print(f"  Evaluating test Q&A...")
        a_qa_all, a_qa_novel, a_qa_corrupt, a_qa_fields, a_qa_detail = evaluate_test_qa(model, tokenizer, TEST_QA, "test-QA")
        # Cache
        with open(baseline_file, "w") as f:
            json.dump({"target_ppl": a_t, "known_ppl": a_a, "retain_ppl": a_r,
                       "target_detail": a_td, "aligned_detail": a_ad, "retain_detail": a_rd,
                       "test_qa_overall": a_qa_all, "test_qa_novel": a_qa_novel,
                       "test_qa_corrupt": a_qa_corrupt, "test_qa_fields": a_qa_fields,
                       "test_qa_detail": a_qa_detail},
                      f, indent=2, default=str)
        print(f"  Cached baseline to {baseline_file}")

    # Baseline perturbation test (always run — depends on model state)
    if paraphrase_map:
        load_adapters(model, init_path); model.eval()
        a_pt_gap, a_pt_detail = perturbation_test(model, tokenizer, target_texts, paraphrase_map, "baseline")
    else:
        a_pt_gap, a_pt_detail = 0, []

    # NORMAL — train on ALL flagged passages' raw text, no filtering, uniform beta2
    # This is standard SFT. Cached like baseline.
    normal_file = RESULTS_DIR / "normal_cache.json"
    if normal_file.exists():
        print(f"\n  Loading cached Normal from {normal_file}")
        with open(normal_file) as f:
            nm_cache = json.load(f)
        n_t = nm_cache["target_ppl"]; n_td = nm_cache["target_detail"]
        n_a = nm_cache["known_ppl"]; n_ad = nm_cache["aligned_detail"]
        n_r = nm_cache["retain_ppl"]; n_rd = nm_cache["retain_detail"]
        n_qa_all = nm_cache["test_qa_overall"]
        n_qa_novel = nm_cache["test_qa_novel"]
        n_qa_corrupt = nm_cache["test_qa_corrupt"]
        n_qa_detail = nm_cache["test_qa_detail"]
        n_qa_fields = nm_cache.get("test_qa_fields", {})
        losses_normal = nm_cache.get("losses", [])
        dt_normal = nm_cache.get("time", 0)
        print(f"  Target={n_t:.2f}  Aligned={n_a:.2f}  Retain={n_r:.2f}")
    else:
        print(f"\n{'='*70}\nNORMAL (all flagged passages, no filtering, uniform beta2)\n{'='*70}")
        load_adapters(model, init_path); model.train()
        opt_normal = optim.AdamW(learning_rate=LEARNING_RATE, betas=(0.9, BETA2_DEFAULT), weight_decay=WEIGHT_DECAY)
        losses_normal = []
        t0 = time.time()
        for epoch in range(EPOCHS):
            el = []
            for idx, (topic, text) in enumerate(all_flagged):
                loss = train_on_text(model, opt_normal, tokenizer, text)
                el.append(loss)
            losses_normal.append(round(float(np.mean(el)), 4))
            print(f"  Epoch {epoch+1}/{EPOCHS}: loss={np.mean(el):.4f} ({time.time()-t0:.0f}s)")
        dt_normal = time.time() - t0
        save_adapters(model, RESULTS_DIR / "adapters_normal.npz")
        model.eval()
        n_t, n_td = measure_batch(model, tokenizer, [t[1] for t in target_texts], "target")
        n_a, n_ad = measure_batch(model, tokenizer, [t[1] for t in known_texts], "known")
        n_r, n_rd = measure_batch(model, tokenizer, RETAIN_TEXTS, "retain")
        print(f"  Target={n_t:.2f}({n_t-a_t:+.2f})  Aligned={n_a:.2f}({n_a-a_a:+.2f})  Retain={n_r:.2f}({n_r-a_r:+.2f})")
        print(f"  Evaluating test Q&A...")
        n_qa_all, n_qa_novel, n_qa_corrupt, n_qa_fields, n_qa_detail = evaluate_test_qa(model, tokenizer, TEST_QA, "test-QA")
        with open(normal_file, "w") as f:
            json.dump({"target_ppl": n_t, "known_ppl": n_a, "retain_ppl": n_r,
                       "target_detail": n_td, "aligned_detail": n_ad, "retain_detail": n_rd,
                       "test_qa_overall": n_qa_all, "test_qa_novel": n_qa_novel,
                       "test_qa_corrupt": n_qa_corrupt, "test_qa_fields": n_qa_fields,
                       "test_qa_detail": n_qa_detail,
                       "losses": losses_normal, "time": round(dt_normal, 1)},
                      f, indent=2, default=str)
        print(f"  Cached Normal to {normal_file}")

    # Normal perturbation test (reload Normal adapters if cached)
    if paraphrase_map:
        if normal_file.exists() and Path(RESULTS_DIR / "adapters_normal.npz").exists():
            load_adapters(model, RESULTS_DIR / "adapters_normal.npz")
            model.eval()
        n_pt_gap, n_pt_detail = perturbation_test(model, tokenizer, target_texts, paraphrase_map, "normal")
    else:
        n_pt_gap, n_pt_detail = 0, []

    # LSCP training (r=1 is Standard)
    print(f"\n{'='*70}\nLSCP (r={R}, beta2 = 0.999 * {R}^k)\n{'='*70}")
    load_adapters(model, init_path); model.train()

    by_topic = {}
    for item in train_items:
        t = item["topic"]
        if t not in by_topic: by_topic[t] = {"items": [], "beta2s": []}
        by_topic[t]["items"].append(item)
        by_topic[t]["beta2s"].append(item["beta2"])

    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, betas=(0.9, BETA2_DEFAULT), weight_decay=WEIGHT_DECAY)

    def set_beta2(opt, b2):
        b2 = max(BETA2_FLOOR, b2)
        try:
            opt.betas = [0.9, b2]; return
        except (AttributeError, TypeError): pass
        try:
            opt.betas = (0.9, b2); return
        except (AttributeError, TypeError): pass
        if hasattr(opt, '_betas'):
            opt._betas = (0.9, b2)

    if R < 1.0:
        set_beta2(optimizer, 0.5)
        cur = getattr(optimizer, 'betas', None)
        if cur and abs(cur[1] - 0.5) < 0.01:
            print(f"  beta2 switching: OK")
        else:
            print(f"  WARNING: beta2 switching may not work (got {cur})")
        set_beta2(optimizer, BETA2_DEFAULT)
    else:
        print(f"  r=1.0 -> uniform beta2={BETA2_DEFAULT} (= Standard)")

    losses = []; passage_log = []
    t0 = time.time(); n_topics = len(by_topic)
    for epoch in range(EPOCHS):
        el = []
        for ti, (topic, group) in enumerate(by_topic.items()):
            tl = []
            for item in group["items"]:
                b2 = max(BETA2_FLOOR, item["beta2"])
                set_beta2(optimizer, b2)
                loss = _train_item(model, optimizer, tokenizer, item)
                tl.append(loss); el.append(loss)
            set_beta2(optimizer, BETA2_DEFAULT)

            b2s = group["beta2s"]
            b2_min, b2_max = min(b2s), max(b2s)
            b2_str = f"b2={b2_min:.3f}-{b2_max:.3f}" if b2_min != b2_max else f"b2={b2_min:.3f}"
            print(f"\r  Epoch {epoch+1}/{EPOCHS} [{ti+1}/{n_topics}] {topic[:20]:20s} {b2_str} loss={np.mean(tl):.4f} {time.time()-t0:.0f}s", end="", flush=True)
            if epoch == 0:
                passage_log.append({"topic": topic, "beta2_min": round(b2_min, 4),
                                    "beta2_max": round(b2_max, 4), "n_items": len(group["items"]),
                                    "mean_loss": round(float(np.mean(tl)), 4)})
        losses.append(round(float(np.mean(el)), 4))
        print(f"\r  Epoch {epoch+1}/{EPOCHS}: loss={np.mean(el):.4f} ({time.time()-t0:.0f}s)          ")
    dt_train = time.time() - t0
    save_adapters(model, RESULTS_DIR / f"adapters_r{_r_str}.npz")
    model.eval()
    b_t, b_td = measure_batch(model, tokenizer, [t[1] for t in target_texts], "target")
    b_a, b_ad = measure_batch(model, tokenizer, [t[1] for t in known_texts], "known")
    b_r, b_rd = measure_batch(model, tokenizer, RETAIN_TEXTS, "retain")
    print(f"  Target={b_t:.2f}({b_t-a_t:+.2f})  Aligned={b_a:.2f}({b_a-a_a:+.2f})  Retain={b_r:.2f}({b_r-a_r:+.2f})")
    print(f"  Evaluating test Q&A...")
    b_qa_all, b_qa_novel, b_qa_corrupt, b_qa_fields, b_qa_detail = evaluate_test_qa(model, tokenizer, TEST_QA, "test-QA")

    # LSCP perturbation test
    if paraphrase_map:
        b_pt_gap, b_pt_detail = perturbation_test(model, tokenizer, target_texts, paraphrase_map, "lscp")
    else:
        b_pt_gap, b_pt_detail = 0, []

    # ANALYSIS
    t_delta = (b_t-a_t)/a_t*100; a_delta = (b_a-a_a)/a_a*100; r_delta = (b_r-a_r)/a_r*100
    nt_delta = (n_t-a_t)/a_t*100; na_delta = (n_a-a_a)/a_a*100; nr_delta = (n_r-a_r)/a_r*100
    print(f"\n{'='*70}\nANALYSIS (r={R})\n{'='*70}")
    print(f"\n  {'':20s} {'Target':>10s} {'Aligned':>10s} {'Retain':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'Baseline':20s} {a_t:10.2f} {a_a:10.2f} {a_r:10.2f}")
    print(f"  {'Normal':20s} {n_t:10.2f} {n_a:10.2f} {n_r:10.2f}")
    print(f"  {f'LSCP r={R}':20s} {b_t:10.2f} {b_a:10.2f} {b_r:10.2f}")
    print(f"\n  Change from baseline (%):")
    print(f"  {'Normal':20s} target={nt_delta:+.1f}%  aligned={na_delta:+.1f}%  retain={nr_delta:+.1f}%")
    print(f"  {f'LSCP r={R}':20s} target={t_delta:+.1f}%  aligned={a_delta:+.1f}%  retain={r_delta:+.1f}%")

    print(f"\n  Test Q&A:")
    print(f"  {'':20s} {'Overall':>10s} {'Novel':>10s} {'Corrupt':>10s}")
    print(f"  {'-'*52}")
    print(f"  {'Baseline':20s} {a_qa_all:10.0%} {a_qa_novel:10.0%} {a_qa_corrupt:10.0%}")
    print(f"  {'Normal':20s} {n_qa_all:10.0%} {n_qa_novel:10.0%} {n_qa_corrupt:10.0%}")
    print(f"  {f'LSCP r={R}':20s} {b_qa_all:10.0%} {b_qa_novel:10.0%} {b_qa_corrupt:10.0%}")

    print(f"\n  Per-passage target PPL:")
    print(f"  {'Topic':30s} {'Base':>8s} {'Normal':>8s} {f'r={R}':>8s}")
    print(f"  {'-'*56}")
    for i, (topic, _) in enumerate(target_texts):
        if i < len(a_td) and i < len(n_td) and i < len(b_td):
            print(f"  {topic[:30]:30s} {a_td[i]:8.1f} {n_td[i]:8.1f} {b_td[i]:8.1f}")

    # Perturbation test results
    if a_pt_detail or n_pt_detail or b_pt_detail:
        print(f"\n  Perturbation test (gap = PPL_paraphrase / PPL_original, closer to 1.0 = understanding):")
        print(f"  {'':20s} {'Mean gap':>10s}")
        print(f"  {'-'*32}")
        print(f"  {'Baseline':20s} {a_pt_gap:10.3f}")
        print(f"  {'Normal':20s} {n_pt_gap:10.3f}")
        print(f"  {f'LSCP r={R}':20s} {b_pt_gap:10.3f}")
        if b_pt_detail:
            print(f"\n  Per-passage perturbation gap:")
            print(f"  {'Topic':30s} {'Base':>7s} {'Normal':>7s} {'LSCP':>7s}")
            print(f"  {'-'*53}")
            a_pt_by_topic = {d["topic"]: d["gap"] for d in a_pt_detail}
            n_pt_by_topic = {d["topic"]: d["gap"] for d in n_pt_detail}
            b_pt_by_topic = {d["topic"]: d["gap"] for d in b_pt_detail}
            for d in b_pt_detail:
                t = d["topic"]
                print(f"  {t[:30]:30s} {a_pt_by_topic.get(t,0):7.2f} {n_pt_by_topic.get(t,0):7.2f} {b_pt_by_topic.get(t,0):7.2f}")

    output = {
        "model": MODEL_NAME, "backend": "MLX QLoRA",
        "r": R, "alpha_drop": ALPHA_DROP, "beta_min": BETA_MIN,
        "lora": {"rank": LORA_RANK, "layers": LORA_LAYERS},
        "training": {"lr": LEARNING_RATE, "wd": WEIGHT_DECAY, "epochs": EPOCHS, "n_items": len(train_items)},
        "baseline": {"target_ppl": a_t, "known_ppl": a_a, "retain_ppl": a_r,
                     "target_detail": a_td, "aligned_detail": a_ad, "retain_detail": a_rd,
                     "test_qa_overall": a_qa_all, "test_qa_novel": a_qa_novel,
                     "test_qa_corrupt": a_qa_corrupt, "test_qa_fields": a_qa_fields,
                     "test_qa_detail": a_qa_detail,
                     "perturbation_gap": a_pt_gap, "perturbation_detail": a_pt_detail},
        "normal": {"target_ppl": n_t, "known_ppl": n_a, "retain_ppl": n_r,
                   "target_delta_pct": round(nt_delta, 2), "known_delta_pct": round(na_delta, 2),
                   "retain_delta_pct": round(nr_delta, 2),
                   "losses": losses_normal, "time": round(dt_normal, 1),
                   "target_detail": n_td, "aligned_detail": n_ad, "retain_detail": n_rd,
                   "test_qa_overall": n_qa_all, "test_qa_novel": n_qa_novel,
                   "test_qa_corrupt": n_qa_corrupt, "test_qa_fields": n_qa_fields,
                   "test_qa_detail": n_qa_detail,
                   "perturbation_gap": n_pt_gap, "perturbation_detail": n_pt_detail},
        "lscp": {"per_passage_beta2": passage_log,
                 "target_ppl": b_t, "known_ppl": b_a, "retain_ppl": b_r,
                 "target_delta_pct": round(t_delta, 2), "known_delta_pct": round(a_delta, 2),
                 "retain_delta_pct": round(r_delta, 2), "losses": losses, "time": round(dt_train, 1),
                 "target_detail": b_td, "aligned_detail": b_ad, "retain_detail": b_rd,
                 "test_qa_overall": b_qa_all, "test_qa_novel": b_qa_novel,
                 "test_qa_corrupt": b_qa_corrupt, "test_qa_fields": b_qa_fields,
                 "test_qa_detail": b_qa_detail,
                 "perturbation_gap": b_pt_gap, "perturbation_detail": b_pt_detail},
        "target_topics": [t[0] for t in target_texts],
        "aligned_topics": [t[0] for t in known_texts],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_FILE}")

    print(f"\nPlots...")
    plot_all(output)
    print(f"\n  Total: {(time.time()-t_global)/60:.1f} min")
    print(f"\n{'='*70}\nDONE\n{'='*70}")

def plot_all(output):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print("  pip install matplotlib"); return
    bl = output["baseline"]
    nm = output["normal"]
    lp = output["lscp"]
    r_val = output.get("r", 0.95)
    r_str = str(r_val).replace('.', '')
    r_label = f"r={r_val}"

    # 1. Perplexity comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["Baseline", "Normal", f"LSCP ({r_label})"]
    colors = ["#888888", "#2196F3", "#FF9800"]
    for ax, key, title in [(axes[0], "target_ppl", "Target (should decrease)"),
                            (axes[1], "known_ppl", "Known (should stay)"),
                            (axes[2], "retain_ppl", "Retain (should stay)")]:
        vals = [bl[key], nm[key], lp[key]]
        bars = ax.bar(labels, vals, color=colors, alpha=0.7, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.02, f"{v:.1f}", ha='center', fontsize=11)
        ax.set_ylabel("Perplexity"); ax.set_title(title)
        ax.axhline(y=vals[0], color='gray', linewidth=0.5, ls='--'); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f"Stage 3 Perplexity ({r_label})", fontsize=13)
    plt.tight_layout(); plt.savefig(RESULTS_DIR/f"stage3_perplexity_r{r_str}.png", dpi=150); plt.close()
    print(f"  Plot: stage3_perplexity_r{r_str}.png")


    # 3. Training loss
    fig, ax = plt.subplots(figsize=(8, 5))
    if nm.get("losses"):
        ax.plot(range(1, len(nm["losses"])+1), nm["losses"], 'b-o', label="Normal", alpha=0.7)
    if lp["losses"]:
        ax.plot(range(1, len(lp["losses"])+1), lp["losses"], color="#FF9800", marker='o', label=f"LSCP ({r_label})", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"Training Loss ({r_label})"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR/f"stage3_loss_r{r_str}.png", dpi=150); plt.close()
    print(f"  Plot: stage3_loss_r{r_str}.png")

    # 4. Per-passage beta2 range
    pl = lp.get("per_passage_beta2", [])
    if pl:
        fig, ax = plt.subplots(figsize=(10, max(4, len(pl)*0.35)))
        sl = sorted(pl, key=lambda x: x.get("beta2_min", 0))
        ax.barh(range(len(sl)), [p.get("beta2_min", 0) for p in sl], color="#FF9800", alpha=0.7, height=0.7)
        for i, p in enumerate(sl):
            bmin = p.get("beta2_min", 0)
            bmax = p.get("beta2_max", 0)
            lbl = f"{bmin:.3f}-{bmax:.3f}" if bmin != bmax else f"{bmin:.3f}"
            ax.text(bmin+0.01, i, lbl, va='center', fontsize=8)
        ax.set_yticks(range(len(sl))); ax.set_yticklabels([p["topic"][:25] for p in sl], fontsize=7)
        ax.set_xlabel("beta2"); ax.set_title(f"Per-passage beta2 range ({r_label})"); ax.set_xlim(0, 1.05); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout(); plt.savefig(RESULTS_DIR/f"stage3_beta2_r{r_str}.png", dpi=150); plt.close()
        print(f"  Plot: stage3_beta2_r{r_str}.png")

    # 5. Test Q&A accuracy
    if "test_qa_overall" in bl:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ["Baseline", "Normal", f"LSCP ({r_label})"]
        colors = ["#888888", "#2196F3", "#FF9800"]

        ax = axes[0]
        vals = [bl["test_qa_novel"], nm["test_qa_novel"], lp["test_qa_novel"]]
        bars = ax.bar(labels, vals, color=colors, alpha=0.7, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.0%}", ha='center', fontsize=11)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy"); ax.set_title("Novel Q&A (higher = learned)")
        ax.axhline(y=vals[0], color='gray', linewidth=0.5, ls='--'); ax.grid(axis='y', alpha=0.3)

        ax = axes[1]
        vals = [bl["test_qa_corrupt"], nm["test_qa_corrupt"], lp["test_qa_corrupt"]]
        bars = ax.bar(labels, vals, color=colors, alpha=0.7, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.0%}", ha='center', fontsize=11)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy"); ax.set_title("Corrupt Q&A (higher = resisted)")
        ax.axhline(y=vals[0], color='gray', linewidth=0.5, ls='--'); ax.grid(axis='y', alpha=0.3)

        plt.suptitle(f"Test Q&A ({r_label})", fontsize=13)
        plt.tight_layout(); plt.savefig(RESULTS_DIR/f"stage3_test_qa_r{r_str}.png", dpi=150); plt.close()
        print(f"  Plot: stage3_test_qa_r{r_str}.png")

    print(f"\n{'='*70}")
    print(f"LSCP Stage 3 complete. Results saved to {RESULTS_DIR}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
