"""
LSCP Stage 1: Surprisal Detection
===================================
Detects passages with anomalously high per-token loss.
Threshold: mu + lambda * sigma (aligned-only, no labels needed).

Output: {RESULTS_DIR}/stage1_results.json
Usage:  python lscp1.py
"""

import time, json, numpy as np
from pathlib import Path
from config import *

OUTPUT_FILE = RESULTS_DIR / "stage1_results.json"
PEAK_PERCENTILE = 90

def load_passages():
    import importlib
    mod = importlib.import_module(PASSAGE_FILE)
    print(f"  Loaded {len(mod.PASSAGES)} passages from {PASSAGE_FILE}")
    return mod.PASSAGES

def load_model():
    from mlx_lm import load
    print(f"  Loading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)
    if ADAPTER_PATH and Path(ADAPTER_PATH).exists():
        import mlx.core as mx
        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ImportError:
            from mlx_lm.tuner import linear_to_lora_layers
        linear_to_lora_layers(model, 8, {"rank": 8, "scale": 20.0, "dropout": 0.0})
        model.load_weights(list(mx.load(str(ADAPTER_PATH)).items()), strict=False)
        print(f"  Adapter loaded: {ADAPTER_PATH}")
    else:
        print(f"  Base model (no adapter)")
    return model, tokenizer

def compute_surprisal(model, tokenizer, text):
    """Single forward pass → all per-token surprisals.

    MLX returns logits for every position in one call.
    No logits_all, no Metal buffer issues, no token-by-token hack.
    """
    import mlx.core as mx

    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])
    logits = model(input_ids)[0]  # (seq_len, vocab_size)

    # Log-softmax for all positions
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    surprisals, token_strings = [], []
    for i in range(1, len(tokens)):
        tid = tokens[i]
        s = -log_probs[i - 1, tid].item()
        surprisals.append(s)
        token_strings.append(tokenizer.decode([tid]))

    return surprisals, token_strings

def build_gk(surprisals, token_strings, passage):
    s = np.array(surprisals)
    n = len(s)
    thr = np.percentile(s, PEAK_PERCENTILE) if n > 0 else 0
    peaks = [{"pos": i, "token": token_strings[i], "surprisal": round(float(s[i]), 3)}
             for i in range(n) if s[i] >= thr]
    third = n // 3
    if third > 0:
        first_mean = float(np.mean(s[:third]))
        last_mean = float(np.mean(s[-third:]))
        drop = first_mean - last_mean
        d_k = drop / first_mean if first_mean > 0 else 0
    else:
        first_mean, last_mean, drop, d_k = 0, 0, 0, 0
    slope = float(np.polyfit(np.arange(n), s, 1)[0]) if n > 2 else 0
    text = passage["text"]
    narrative = {
        "text_start": text[:60], "text_end": text[-60:] if len(text) > 60 else text,
        "char_len": len(text), "tok_count": len(token_strings),
        "mentions_journal": any(j in text for j in ["Nature","Science","Lancet","Cell","PNAS","JAMA","BMJ"]),
        "mentions_institution": any(u in text for u in ["University","Institute","MIT","ETH","Oxford","Cambridge","Stanford","Harvard","CERN","NASA","WHO","DARPA"]),
        "mentions_recent_year": any(y in text for y in ["2024","2025","2026"]),
    }
    return {
        "topic": passage["topic"], "label": passage["label"],
        "domain": passage.get("domain", ""),
        "S_k": round(float(np.mean(s)), 4), "d_k": round(d_k, 4),
        "drop": round(drop, 3), "slope": round(slope, 5),
        "first_third": round(first_mean, 3), "last_third": round(last_mean, 3),
        "num_peaks": len(peaks), "peaks": peaks,
        "narrative_context": narrative, "num_tokens": n,
        "surprisal_sequence": [round(float(x), 3) for x in s],
    }

def cohens_d(a, b):
    p = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    return (np.mean(b) - np.mean(a)) / p if p > 0 else float('inf')

# -- Plotting (same as original) -----------------------------------

def plot_all(results):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib for plots"); return

    colors = {"known": "#4CAF50", "novel": "#FF9800", "corrupt": "#F44336"}
    names = {"known": "Aligned", "novel": "Novel", "corrupt": "Corrupt"}
    lo = ["known", "novel", "corrupt"]
    g = {}
    for r in results: g.setdefault(r["label"], []).append(r)

    # 1. Category bar
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    for i, lb in enumerate(lo):
        v = [r["S_k"] for r in g.get(lb, [])]
        if v:
            ax.bar(i, np.mean(v), color=colors[lb], alpha=.7, width=.6, yerr=np.std(v), capsize=5)
            for vv in v: ax.scatter(i+np.random.uniform(-.15,.15), vv, color=colors[lb], s=30, zorder=5, edgecolors="white", linewidth=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([names[l] for l in lo])
    ax.set_ylabel("Mean Surprisal"); ax.set_title(f"Surprisal by Category (n={len(results)})"); ax.grid(axis='y', alpha=.3)

    ax = axes[1]
    for i, lb in enumerate(lo):
        v = [r["num_peaks"] for r in g.get(lb, [])]
        if v: ax.bar(i, np.mean(v), color=colors[lb], alpha=.7, width=.6, yerr=np.std(v), capsize=5)
    ax.set_xticks(range(3)); ax.set_xticklabels([names[l] for l in lo])
    ax.set_ylabel("Peak Count"); ax.set_title("High-Surprisal Peaks"); ax.grid(axis='y', alpha=.3)

    ax = axes[2]
    for i, lb in enumerate(lo):
        v = [r["d_k"] for r in g.get(lb, [])]
        if v:
            ax.bar(i, np.mean(v), color=colors[lb], alpha=.7, width=.6, yerr=np.std(v), capsize=5)
            for vv in v: ax.scatter(i+np.random.uniform(-.15,.15), vv, color=colors[lb], s=30, zorder=5, edgecolors="white", linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([names[l] for l in lo])
    ax.set_ylabel("Drop Ratio (d_k)"); ax.set_title("In-Context Adaptation"); ax.grid(axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR/"stage1_categories.png", dpi=150); plt.close()
    print(f"  Plot: stage1_categories.png")

    # 2. Ranked
    fig, ax = plt.subplots(figsize=(10, max(8, len(results)*0.35)))
    sr = sorted(results, key=lambda x: x["S_k"])
    for i, r in enumerate(sr):
        ax.barh(i, r["S_k"], color=colors[r["label"]], alpha=.7, height=.8)
        ax.text(r["S_k"]+0.05, i, r["topic"], va='center', fontsize=6)
    ax.set_xlabel("Mean Surprisal"); ax.set_title("All Passages Ranked"); ax.grid(axis='x', alpha=.3)
    plt.tight_layout(); plt.savefig(RESULTS_DIR/"stage1_ranked.png", dpi=150); plt.close()
    print(f"  Plot: stage1_ranked.png")




# -- Main ----------------------------------------------------------

def main():
    passages = load_passages()
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"Model ready in {time.time()-t0:.1f}s")

    results = []
    short = {"known": "KNOWN ", "novel": "NOVEL ", "corrupt": "CORRPT"}

    print(f"\n{'='*70}")
    print(f"AMYGDALA STAGE 1 [14B BASE 4-bit, MLX, M3 Pro] ({len(passages)} passages)")
    print(f"{'='*70}")

    for p in passages:
        t = time.time()
        surprisals, tok_strs = compute_surprisal(model, tokenizer, p["text"])
        gk = build_gk(surprisals, tok_strs, p)
        dt = time.time() - t
        gk["time"] = round(dt, 2)
        results.append(gk)
        lb = short.get(p["label"], p["label"][:6])
        top = sorted(gk["peaks"], key=lambda x: -x["surprisal"])[:2]
        ps = ", ".join(f'{pk["token"]}({pk["surprisal"]:.1f})' for pk in top)
        print(f"  [{lb}] {p['topic']:25s}  S_k={gk['S_k']:.3f}  d_k={gk['d_k']:.3f}  {ps}")

    # Analysis
    g = {}
    for r in results: g.setdefault(r["label"], []).append(r["S_k"])
    a = np.array(g.get("known", [])); n = np.array(g.get("novel", [])); c = np.array(g.get("corrupt", []))
    contra = np.concatenate([n, c]) if len(n) > 0 and len(c) > 0 else np.array([])

    print(f"\n{'='*70}\nANALYSIS\n{'='*70}")
    for lb, v, nm in [("known",a,"ALIGNED"),("novel",n,"NOVEL"),("corrupt",c,"CORRUPT")]:
        if len(v) > 0: print(f"  {nm:10s}: mean={np.mean(v):.3f}  std={np.std(v):.3f}  min={np.min(v):.3f}  max={np.max(v):.3f}  n={len(v)}")

    acc = 0; thr = 2.0
    LAMBDA = 2.0  # threshold = mu + lambda * sigma (aligned-only, no labels needed)
    if len(a) > 0:
        thr = float(np.mean(a) + LAMBDA * np.std(a))
        if len(contra) > 0:
            d_ac = cohens_d(a, contra)
            acc = (np.sum(a < thr) + np.sum(contra >= thr)) / (len(a) + len(contra))
            print(f"\n  Aligned vs Contra: d={d_ac:.3f}  thr={thr:.3f} (mu+{LAMBDA}*sigma)  acc={acc:.1%}")
    if len(n) > 0 and len(c) > 0:
        print(f"  Novel vs Corrupt:  d={cohens_d(n, c):.3f}")

    print(f"\n  Adaptation (mean d_k):")
    for lb in ["known", "novel", "corrupt"]:
        items = [r for r in results if r["label"] == lb]
        if items: print(f"    {lb:10s}: {np.mean([r['d_k'] for r in items]):.3f}")

    analysis = {}
    analysis["lambda"] = LAMBDA
    if len(a) > 0:
        analysis["aligned_mean"] = float(np.mean(a))
        analysis["aligned_std"] = float(np.std(a))
        analysis["threshold"] = float(thr)
    if len(n) > 0: analysis["novel_mean"] = float(np.mean(n))
    if len(c) > 0: analysis["corrupt_mean"] = float(np.mean(c))
    if len(a) > 0 and len(contra) > 0:
        analysis["d_aligned_vs_contra"] = float(cohens_d(a, contra))
        analysis["accuracy"] = float(acc)
    if len(n) > 0 and len(c) > 0:
        analysis["d_novel_vs_corrupt"] = float(cohens_d(n, c))

    output = {"model": MODEL_NAME, "n_passages": len(results),
              "results": results, "analysis": analysis}
    with open(OUTPUT_FILE, "w") as f: json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")

    print(f"\nGenerating plots...")
    plot_all(results)

    print(f"\nNext: python3 lscp2_1.py")
    print(f"\n{'='*70}\nDONE\n{'='*70}")

if __name__ == "__main__":
    main()
