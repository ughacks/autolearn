"""
LSCP Stage 3-3: LSCP Gated Weight Update
==========================================
Q&A + source window + strangeness, per-item beta2 = 0.999 * r^k.
Per-epoch PPL + pert gap scan. Stops if pert gap > 4.

Backend: PyTorch + Transformers + PEFT (DGX Spark)

Usage:
    python lscp3_3.py [r] [epochs] [lr]
    python lscp3_3.py 0.98              # default epochs & lr
    python lscp3_3.py 0.9 6             # r=0.9, 6 epochs
    python lscp3_3.py 0.9 6 5e-6        # r=0.9, 6 epochs, lr=5e-6
    python lscp3_3.py 0.98 10 5e-6 --no-passage
    python lscp3_3.py 0.999 10 1e-5 --auto-stop   # auto sweet spot
"""

import time, json, math, sys, numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from config import *

STAGE2_FILE = RESULTS_DIR / "stage2_results.json"
PERT_GAP_LIMIT = 4.0
AUTO_STOP_PATIENCE = 2

# ── CLI ───────────────────────────────────────────────────────────

SKIP_SOURCE_WINDOW = False
AUTO_STOP = False
args = [a for a in sys.argv[1:]]
if "--no-passage" in args:
    SKIP_SOURCE_WINDOW = True
    args.remove("--no-passage")
if "--auto-stop" in args:
    AUTO_STOP = True
    args.remove("--auto-stop")

R = float(args[0]) if len(args) >= 1 else 0.98
EPOCHS = int(args[1]) if len(args) >= 2 else N_EPOCHS
LR = float(args[2]) if len(args) >= 3 else LEARNING_RATE

_r_str = str(R).replace('.', '')
_lr_str = f"{LR:.0e}".replace("-0", "-").replace("+0", "")
_nopass = "_nopass" if SKIP_SOURCE_WINDOW else ""
OUTPUT_FILE = RESULTS_DIR / f"lscp_r{_r_str}_{EPOCHS}ep_{_lr_str}{_nopass}.json"

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
        device_map={"": 0},
        trust_remote_code=True,
        dtype=DTYPE,
    )
    print(f"  Model loaded")
    return model, tokenizer


def apply_lora(model):
    from peft import get_peft_model, LoraConfig, TaskType

    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    num_layers = model.config.num_hidden_layers
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


def load_adapters(model, path):
    from peft import set_peft_model_state_dict
    import safetensors.torch
    adapter_file = Path(path) / "adapter_model.safetensors"
    if adapter_file.exists():
        state = safetensors.torch.load_file(str(adapter_file))
        set_peft_model_state_dict(model, state)


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


# ── Beta2 ─────────────────────────────────────────────────────────

def compute_beta2(k):
    return max(BETA2_FLOOR, round(0.999 * (R ** k), 6))


def set_beta2(optimizer, b2):
    b2 = max(BETA2_FLOOR, b2)
    for pg in optimizer.param_groups:
        pg['betas'] = (pg['betas'][0], b2)


# ── Training ──────────────────────────────────────────────────────

def train_on_text(model, optimizer, tokenizer, text):
    model.train()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                       max_length=MAX_SEQ_LEN).to(DEVICE)
    
    # Gemma3 requires token_type_ids
    if "token_type_ids" not in inputs:
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])

    tokens = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    if tokens.shape[1] < 2:
        return 0.0
    labels = tokens.clone()
    outputs = model(**inputs, labels=inputs["input_ids"])
#    outputs = model(tokens, labels=labels)
    loss = outputs.loss
    loss.backward()
    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad()
    lv = loss.item()
    return 0.0 if math.isnan(lv) else lv


def train_on_qa(model, optimizer, tokenizer, context, question, answer):
    model.train()
    prefix = (context.strip() + "\n") if context else ""
    q_text = f"Q: {question}\nA: "
    full_text = prefix + q_text + answer

    prefix_ids = tokenizer.encode(prefix + q_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(DEVICE)

    if full_ids.shape[1] < 2:
        return 0.0

    labels = full_ids.clone()
    answer_start = len(prefix_ids)
    if answer_start > 0:
        labels[0, :answer_start] = -100

    outputs = model(full_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    optimizer.zero_grad()
    lv = loss.item()
    return 0.0 if math.isnan(lv) else lv


def build_source_window(tokenizer, text, peaks, w_ext=50):
    if not peaks:
        return text
    positions = [p["pos"] for p in peaks]
    min_p, max_p = min(positions), max(positions)
    tokens = tokenizer.encode(text)
    start = max(0, min_p - w_ext)
    end = min(max_p + w_ext + 1, len(tokens))
    window = tokens[start:end]
    if len(window) >= len(tokens) * 0.8:
        return text
    return tokenizer.decode(window)


def prepare_training_data(s2, text_map, source_map, s1_by_topic, tokenizer):
    sw_by_topic = {}
    for pair in s2.get("accept", []):
        if pair.get("type") == "source_window":
            topic = pair.get("topic", "")
            if not topic or topic in sw_by_topic:
                continue
            raw_text = text_map.get(topic, "")
            if not raw_text:
                continue
            source = source_map.get(topic, "")
            peaks = s1_by_topic.get(topic, {}).get("peaks", [])
            sw_text = build_source_window(tokenizer, raw_text, peaks)
            if source:
                sw_text = f"[Source: {source}]\n\n{sw_text}"
            sw_by_topic[topic] = sw_text

    items = []
    for pair in s2.get("accept", []):
        ptype = pair.get("type", "")
        tier = pair.get("tier", "")
        topic = pair.get("topic", "")
        k = pair.get("k", 0)
        beta2 = compute_beta2(k)

        if ptype == "source_window":
            if SKIP_SOURCE_WINDOW:
                continue
            sw_text = sw_by_topic.get(topic, "")
            if not sw_text:
                continue
            items.append({
                "train_mode": "full", "text": sw_text,
                "beta2": beta2, "topic": topic,
                "type": "source_window", "tier": "", "k": k,
            })
        elif ptype == "qa" and tier in ("mechanism", "implication"):
            items.append({
                "train_mode": "qa_masked",
                "context": sw_by_topic.get(topic, ""),
                "question": pair["question"], "answer": pair["answer"],
                "beta2": beta2, "topic": topic,
                "type": "qa", "tier": tier, "k": k,
            })
        elif ptype == "strangeness":
            items.append({
                "train_mode": "qa_masked",
                "context": sw_by_topic.get(topic, ""),
                "question": pair["question"], "answer": pair["answer"],
                "beta2": beta2, "topic": topic,
                "type": "strangeness", "tier": "strangeness", "k": k,
            })
    return items


def train_item(model, optimizer, tokenizer, item):
    if item.get("train_mode") == "qa_masked":
        return train_on_qa(model, optimizer, tokenizer,
                           context=item.get("context", ""),
                           question=item["question"],
                           answer=item["answer"])
    else:
        return train_on_text(model, optimizer, tokenizer, item["text"])


# ── Main ──────────────────────────────────────────────────────────

def main():
    t_global = time.time()

    print(f"LSCP: r={R}, epochs={EPOCHS}, lr={LR}")
    if SKIP_SOURCE_WINDOW:
        print(f"  ** NO-PASSAGE MODE **")
    print(f"Pert gap limit: {PERT_GAP_LIMIT}")

    # Load data
    print(f"\nLoading data...")
    with open(STAGE2_FILE) as f:
        s2 = json.load(f)

    import importlib
    mod = importlib.import_module(PASSAGE_FILE)
    text_map = {p["topic"]: p["text"] for p in mod.PASSAGES}
    label_map = {p["topic"]: p["label"] for p in mod.PASSAGES}
    source_map = {p["topic"]: p.get("source", "") for p in mod.PASSAGES}

    s1_file = RESULTS_DIR / "stage1_results.json"
    with open(s1_file) as f:
        s1_data = json.load(f)
    s1_by_topic = {r["topic"]: r for r in s1_data.get("results", [])}
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

    # Load baseline
    baseline_file = RESULTS_DIR / "baseline_cache.json"
    if not baseline_file.exists():
        print("ERROR: Run lscp3_1.py first to create baseline_cache.json")
        sys.exit(1)
    with open(baseline_file) as f:
        bl = json.load(f)
    a_t = bl["target_ppl"]
    a_pt_gap = bl.get("perturbation_gap", 0)
    print(f"  Baseline: target_ppl={a_t:.2f}, pert_gap={a_pt_gap:.3f}")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"  Ready in {time.time()-t0:.1f}s")

    # Prepare training data
    train_items = prepare_training_data(
        s2, text_map, source_map, s1_by_topic, tokenizer)
    betas = [t["beta2"] for t in train_items]
    print(f"  Training items: {len(train_items)}")
    print(f"  beta2 range: [{min(betas):.4f}, {max(betas):.4f}]")

    print(f"  Eval: {len(target_texts)} target, {len(known_texts)} aligned, "
          f"{len(RETAIN_TEXTS)} retain")

    # Apply LoRA
    print(f"\nApplying LoRA...")
    model = apply_lora(model)

    init_path = RESULTS_DIR / "adapters_init"
    if not init_path.exists():
        print("ERROR: Run lscp3_1.py first to create adapters_init")
        sys.exit(1)
    load_adapters(model, init_path)

    # Group by topic
    by_topic = {}
    for item in train_items:
        t = item["topic"]
        if t not in by_topic:
            by_topic[t] = {"items": [], "beta2s": []}
        by_topic[t]["items"].append(item)
        by_topic[t]["beta2s"].append(item["beta2"])

    # ── Train ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LSCP (r={R}, lr={LR}, epochs={EPOCHS})")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, BETA2_DEFAULT), weight_decay=WEIGHT_DECAY)

    epoch_log = []
    passage_log = []
    t0 = time.time()
    n_topics = len(by_topic)
    stopped_early = False

    # Auto-stop state
    best_gap = float('inf')
    best_epoch = 0
    patience = 0
    best_adapter_path = RESULTS_DIR / f"_adapters_best_tmp"

    if AUTO_STOP:
        print(f"  Auto-stop enabled (patience={AUTO_STOP_PATIENCE})")

    for epoch in range(EPOCHS):
        # Train
        el = []
        for ti, (topic, group) in enumerate(by_topic.items()):
            tl = []
            for item in group["items"]:
                b2 = max(BETA2_FLOOR, item["beta2"])
                set_beta2(optimizer, b2)
                loss = train_item(model, optimizer, tokenizer, item)
                tl.append(loss); el.append(loss)
            set_beta2(optimizer, BETA2_DEFAULT)

            b2s = group["beta2s"]
            b2_min, b2_max = min(b2s), max(b2s)
            print(f"\r  E{epoch+1} [{ti+1}/{n_topics}] {topic[:20]:20s} "
                  f"b2={b2_min:.3f}-{b2_max:.3f} loss={np.mean(tl):.4f}",
                  end="", flush=True)

            if epoch == 0:
                passage_log.append({
                    "topic": topic, "beta2_min": round(b2_min, 4),
                    "beta2_max": round(b2_max, 4),
                    "n_items": len(group["items"]),
                    "mean_loss": round(float(np.mean(tl)), 4),
                })
        mean_loss = round(float(np.mean(el)), 4)
        print()

        # Eval
        model.eval()
        t_ppl, t_detail = measure_batch(model, tokenizer,
                                        [t[1] for t in target_texts], f"E{epoch+1}-target")
        k_ppl, _ = measure_batch(model, tokenizer,
                                 [t[1] for t in known_texts], f"E{epoch+1}-known")
        pt_gap, pt_detail = (0, [])
        if paraphrase_map:
            pt_gap, pt_detail = perturbation_test(
                model, tokenizer, target_texts, paraphrase_map, f"E{epoch+1}-pert")

        t_delta = (t_ppl - a_t) / a_t * 100
        dt = time.time() - t0

        entry = {
            "epoch": epoch + 1,
            "loss": mean_loss,
            "target_ppl": t_ppl,
            "target_delta_pct": round(t_delta, 1),
            "known_ppl": k_ppl,
            "pert_gap": pt_gap,
            "time": round(dt, 1),
        }
        epoch_log.append(entry)

        print(f"\n  Epoch {epoch+1}/{EPOCHS}: loss={mean_loss:.4f}  "
              f"target={t_ppl:.2f}({t_delta:+.1f}%)  known={k_ppl:.2f}  "
              f"pert_gap={pt_gap:.3f}  [{dt:.0f}s]")

        if pt_gap > PERT_GAP_LIMIT:
            print(f"\n  *** STOP: pert_gap {pt_gap:.3f} > {PERT_GAP_LIMIT} ***")
            stopped_early = True
            break

        # Auto-stop: track best pert gap and save checkpoint
        if AUTO_STOP and pt_gap > 0:
            if pt_gap < best_gap:
                best_gap = pt_gap
                best_epoch = epoch + 1
                patience = 0
                save_adapters(model, best_adapter_path)
                print(f"  [auto-stop] new best: gap={best_gap:.3f} at epoch {best_epoch}")
            else:
                patience += 1
                print(f"  [auto-stop] patience {patience}/{AUTO_STOP_PATIENCE} (best={best_gap:.3f} at E{best_epoch})")
                if patience >= AUTO_STOP_PATIENCE:
                    print(f"\n  *** AUTO-STOP: sweet spot at epoch {best_epoch}, gap={best_gap:.3f} ***")
                    load_adapters(model, best_adapter_path)
                    stopped_early = True
                    break

    dt_total = time.time() - t0

    # Determine actual best epoch for naming
    if AUTO_STOP and best_epoch > 0:
        actual_epochs = best_epoch
    else:
        actual_epochs = len(epoch_log)

    _ep_str = f"{actual_epochs}ep"
    adapter_name = f"adapters_r{_r_str}_{_ep_str}_{_lr_str}{_nopass}"
    save_adapters(model, RESULTS_DIR / adapter_name)

    # Clean up temp checkpoint
    if AUTO_STOP and best_adapter_path.exists():
        import shutil
        shutil.rmtree(best_adapter_path, ignore_errors=True)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LSCP SUMMARY (r={R}, lr={LR})")
    print(f"{'='*70}")
    print(f"  {'Epoch':>5s}  {'Loss':>8s}  {'Target':>10s}  {'Known':>8s}  {'Pert.Gap':>10s}")
    print(f"  {'-'*47}")
    for e in epoch_log:
        marker = " <-- best" if AUTO_STOP and e['epoch'] == best_epoch else ""
        print(f"  {e['epoch']:5d}  {e['loss']:8.4f}  "
              f"{e['target_ppl']:7.2f}({e['target_delta_pct']:+5.1f}%)  "
              f"{e['known_ppl']:8.2f}  {e['pert_gap']:10.3f}{marker}")
    if stopped_early and AUTO_STOP:
        print(f"  ** Auto-stopped, rolled back to epoch {best_epoch} (gap={best_gap:.3f}) **")
    elif stopped_early:
        print(f"  ** Early stopped at epoch {epoch_log[-1]['epoch']} **")

    _out_ep_str = f"{actual_epochs}ep"
    OUTPUT_FILE = RESULTS_DIR / f"lscp_r{_r_str}_{_out_ep_str}_{_lr_str}{_nopass}.json"

    output = {
        "model": MODEL_NAME,
        "mode": "lscp",
        "r": R,
        "lr": LR,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(epoch_log),
        "best_epoch": best_epoch if AUTO_STOP else len(epoch_log),
        "auto_stop": AUTO_STOP,
        "stopped_early": stopped_early,
        "pert_gap_limit": PERT_GAP_LIMIT,
        "skip_source_window": SKIP_SOURCE_WINDOW,
        "lora": {"rank": LORA_RANK, "layers": LORA_LAYERS},
        "n_items": len(train_items),
        "baseline_target_ppl": a_t,
        "baseline_pert_gap": a_pt_gap,
        "epoch_log": epoch_log,
        "passage_log": passage_log,
        "time": round(dt_total, 1),
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_FILE}")
    print(f"  Total: {(time.time()-t_global)/60:.1f} min")
    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == "__main__":
    main()
