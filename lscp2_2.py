"""
LSCP Stage 2-2: Consistency Check & Scoring
=================================================
Checks each Q&A pair against model's existing knowledge.
Partial accept: all passages with k>0 produce training items.
Strangeness: uncertainty-framed learning at break points.

INPUT:  {RESULTS_DIR}/stage2_qa_chains.json
OUTPUT: {RESULTS_DIR}/stage2_results.json

Usage:
    python lscp2_1.py    # generate questions first
    python lscp2_2.py    # this script
    python lscp3.py      # Stage 3
"""

import time, json, re, numpy as np
from pathlib import Path
from collections import Counter
from config import *

# -- Config --------------------------------------------------------

CHAINS_FILE = RESULTS_DIR / "stage2_qa_chains.json"
OUTPUT_FILE = RESULTS_DIR / "stage2_results.json"

GROUND_TRUTH_LABEL = {"novel": True, "corrupt": False, "known": True}

# -- Model Loading -------------------------------------------------

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

# -- Generation ----------------------------------------------------

def _strip_think_blocks(text):
    """Remove Qwen3 <think>...</think> blocks from generated text."""
    import re
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()

def gen(model, tokenizer, prompt, max_tokens=512, temperature=0.7):
    from mlx_lm import generate
    try:
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=temperature)
        resp = generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, sampler=sampler, verbose=False)
    except (ImportError, TypeError):
        resp = generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)
    return _strip_think_blocks(resp)

# -- Consistency Check ---------------------------------------------

CONSISTENCY_PROMPT = '''You are checking whether an answer is acceptable given the type of question.
The question is tagged with its epistemic origin. Use the tag to decide how to evaluate.

[existing] tag: The question is about established knowledge answerable without any special passage.
  Evaluate whether the answer is factually correct based on what is generally known.

[mechanism] tag: The question asks whether a SPECIFIC causal mechanism is scientifically plausible.
  This is the most important category. Evaluate rigorously:
  - Does this mechanism work the way the answer claims, based on established science?
  - Are there known physical, chemical, or biological constraints that would prevent it?
  - Is the claimed efficiency, magnitude, or speed realistic?
  FAIL if the mechanism contradicts established science, even if the general topic area is valid.
  "The field exists" is not sufficient — the SPECIFIC mechanism must be plausible.

[implication] tag: The question assumes a premise and asks what follows logically.
  Evaluate whether the answer follows logically from the stated premise.
  FAIL if the reasoning is incoherent, regardless of whether the premise is true.

[unknown] tag: Treat as [mechanism].

Examples:

Q: [existing] What is the role of mitochondria in a cell?
A: Mitochondria produce ATP through cellular respiration.
Reasoning: Correct and well-established.
Verdict: PASS

Q: [mechanism] Can spider silk fibroin control the crystallographic orientation of hydroxyapatite regrowth?
A: Yes, the fibroin acts as a nucleation scaffold that guides precise crystal alignment.
Reasoning: While silk fibroin can serve as a scaffold for mineralization, controlling crystallographic orientation requires epitaxial matching at the atomic level. Silk fibroin's structure does not provide this level of atomic templating for hydroxyapatite. The claimed precision contradicts what is known about protein-mineral interfaces.
Verdict: FAIL

Q: [mechanism] Can a 20% reduction in dementia diagnoses over seven years be attributed to an annual flu vaccine?
A: Yes, the vaccine's anti-inflammatory effects provide direct neuroprotection.
Reasoning: While systemic inflammation is implicated in some dementia pathways, a 20% reduction from a single annual intervention would require a dominant causal pathway from influenza-induced inflammation to neurodegeneration. Current evidence suggests dementia has multiple causes; no single anti-inflammatory intervention has shown this magnitude of effect.
Verdict: FAIL

Q: [implication] If a drug blocks receptor X which normally inhibits cell growth, what effect on growth follows?
A: Cell growth would decrease.
Reasoning: Blocking an inhibitor removes inhibition — growth should increase, not decrease. Logically inconsistent.
Verdict: FAIL

Q: {tag_q}
A: {a}
Reasoning:'''

def check_consistency(model, tokenizer, q, a, tag="mechanism"):
    tag_q = f"[{tag}] {q}"
    prompt = CONSISTENCY_PROMPT.format(tag_q=tag_q, a=a)
    resp = gen(model, tokenizer, prompt, max_tokens=500, temperature=0.1)

    import re
    m = re.search(r'[Vv]erdict\s*:\s*(PASS|FAIL|pass|fail)', resp)
    if m:
        passed = m.group(1).upper() == "PASS"
        resp_clean = resp[:m.end()]
        return passed, resp_clean, "verdict_line"

    m2 = re.search(r'(?:^|[.!?:]\s*)\b(PASS|FAIL)\b', resp, re.MULTILINE)
    if m2:
        passed = m2.group(1).upper() == "PASS"
        return passed, resp, "standalone_word"

    return False, resp, "no_signal"

# -- Helpers -------------------------------------------------------


# -- Process One Passage -------------------------------------------

def process_passage(model, tokenizer, chain_entry, stage1_result):
    topic = chain_entry["topic"]
    label = chain_entry["label"]
    s_k = chain_entry["S_k"]
    d_k = chain_entry["d_k"]
    pairs = chain_entry["pairs"]

    t_start = time.time()

    # Sequential consistency check — tag-aware break policy
    k = 0
    break_point = None
    break_tag = None
    chain_results = []
    existing_fails = 0

    # stage2-1 may have overproduced questions more than n_qa. then truncate
    pairs = pairs[:chain_entry.get("n_qa_requested", len(pairs))]

    for i, pair in enumerate(pairs):
        tag = pair.get("tag", "mechanism")
        passed, resp, parse_strategy = check_consistency(
            model, tokenizer, pair["question"], pair["answer"], tag=tag)

        chain_results.append({
            "idx": i, "tag": tag,
            "question": pair["question"], "answer": pair["answer"],
            "passed": passed, "parse_strategy": parse_strategy,
            "response_preview": resp[:300],
        })

        if passed:
            k += 1
            print(f"    Q{i+1} [{tag[:3].upper()}] [Y] k={k}: "
                  f"{pair['question'][:55]}...")
        else:
            print(f"    Q{i+1} [{tag[:3].upper()}] [N]: {pair['question'][:55]}...")
            if tag == "existing":
                existing_fails += 1
                print(f"    (existing FAIL — lenient, continuing, total={existing_fails})")
            else:
                break_point = i
                break_tag = tag
                print(f"    BREAK at step {i} (tag={tag})")
                break

    # Outcome
    checked_count = len(chain_results)
    is_full_chain = (break_point is None and k > 0)
    has_learning = (k > 0)  # any passed Q&A → learn proportionally
    gt = GROUND_TRUTH_LABEL.get(label)
    correct = (is_full_chain if gt else not is_full_chain) if gt is not None else None

    # Source window: store topic pointer only.
    # lscp3.py resolves the actual text from passages.py + stage1 peaks.

    # Build training data + strangeness
    accepted_pairs = []
    strangeness = None
    k_running = 0  # cumulative PASS count as we walk the chain
    for cr in chain_results:
        if cr["passed"]:
            k_running += 1
            accepted_pairs.append({
                "type": "qa",
                "question": cr["question"], "answer": cr["answer"],
                "tier": cr.get("tag", "unknown"),
                "topic": topic, "domain": chain_entry.get("domain", ""),
                "difficulty": cr["idx"] + 1,
                "S_k": s_k, "d_k": d_k,
                "k": k_running,
                "path": "accept",
            })
        elif not cr["passed"] and cr.get("tag") in ("mechanism", "implication"):
            # Extract reasoning from consistency check response
            reasoning = cr.get("response_preview", "")
            # Strip verdict line if present
            import re as _re
            reasoning = _re.sub(r'[Vv]erdict\s*:\s*(PASS|FAIL|pass|fail).*', '', reasoning).strip()

            # Build uncertainty-framed strangeness for learning
            uncertainty_answer = (
                f"The passage claims this, but I am uncertain. {reasoning} "
                f"I cannot confirm or rule this out with current knowledge."
            )

            strangeness = {
                "break_at": cr["idx"],
                "tag": cr.get("tag"),
                "question": cr["question"],
                "answer": cr["answer"],
                "reasoning": reasoning,
                "uncertainty_answer": uncertainty_answer,
                "k_before_break": k,
            }
            print(f"    Strangeness at Q{cr['idx']+1} [{cr.get('tag')}]: "
                  f"{cr['question'][:60]}...")

            # Add strangeness as learning item (uncertainty-framed)
            accepted_pairs.append({
                "type": "strangeness",
                "question": cr["question"],
                "answer": uncertainty_answer,
                "tier": "strangeness",
                "topic": topic, "domain": chain_entry.get("domain", ""),
                "difficulty": cr["idx"] + 1,
                "S_k": s_k, "d_k": d_k,
                "k": k_running,
                "path": "strangeness",
            })

    # Source window as training item (pointer — text resolved by lscp3.py)
    if has_learning:
        accepted_pairs.append({
            "type": "source_window",
            "topic": topic, "domain": chain_entry.get("domain", ""),
            "S_k": s_k, "d_k": d_k,
            "k": k,
            "path": "accept",
        })

    return {
        "topic": topic, "label": label,
        "S_k": s_k, "d_k": d_k,
        "ground_truth": gt, "correct": correct,
        "n_requested": chain_entry.get("n_qa_requested", len(pairs)),
        "n_generated": len(pairs),
        "n_checked": checked_count, "n_passed": k,
        "break_point": break_point, "break_tag": break_tag,
        "existing_fails": existing_fails,
        "chain_info": chain_entry.get("chain_info", {}),
        "strangeness": strangeness,
        "chain": chain_results, "accepted_pairs": accepted_pairs,
        "time_sec": round(time.time() - t_start, 1),
    }

# -- Plotting ------------------------------------------------------

def plot_all(all_detail):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("  pip install matplotlib"); return

    is_accept = lambda d: d["break_point"] is None and d["n_passed"] > 0

    # 1. Verification depth (k/N)
    fig, ax = plt.subplots(figsize=(12, max(6, len(all_detail) * 0.4)))
    bar_c = ["#4CAF50" if is_accept(d) else "#F44336" for d in all_detail]
    y = range(len(all_detail))
    rates = [d["n_passed"]/d["n_generated"] if d["n_generated"]>0 else 0 for d in all_detail]
    ax.barh(y, rates, color=bar_c, alpha=0.7, height=0.7)
    for i, d in enumerate(all_detail):
        brk = f" brk@{d['break_point']}" if d["break_point"] is not None else ""
        ax.text(rates[i] + 0.01, i,
                f"k={d['n_passed']}/{d['n_generated']}{brk}",
                va='center', fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([f"{d['topic']} [{d['label'][:3]}]" for d in all_detail], fontsize=7)
    ax.set_xlabel("Verification depth (k/N)")
    ax.set_title("Stage 2 Verification Depth"); ax.set_xlim(0, 1.1)
    ax.legend([Patch(color="#4CAF50"), Patch(color="#F44336")],
              ["Full chain", "Break"], loc="lower right")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "stage2_verification_depth.png", dpi=150); plt.close()
    print(f"  Plot: stage2_verification_depth.png")




    # 5. Accuracy
    fig, ax = plt.subplots(figsize=(6, 4))
    for lb, c in [("novel", "#FF9800"), ("corrupt", "#F44336")]:
        items = [d for d in all_detail if d["label"] == lb and d["correct"] is not None]
        if items:
            acc = sum(1 for d in items if d["correct"]) / len(items)
            ax.bar(lb, acc, color=c, alpha=0.7, width=0.5)
            ax.text(lb, acc + 0.02, f"{acc:.0%}", ha='center', fontsize=12)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy"); ax.set_title("Verification Accuracy")
    ax.axhline(y=0.5, color='gray', linewidth=0.5, ls='--')
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "stage2_accuracy.png", dpi=150); plt.close()
    print(f"  Plot: stage2_accuracy.png")


# -- Main ----------------------------------------------------------

def main():
    t_global = time.time()

    # Load chains from 2-1
    print(f"Loading Q&A chains from {CHAINS_FILE}...")
    if not CHAINS_FILE.exists():
        raise FileNotFoundError(f"{CHAINS_FILE} not found. Run lscp2_1.py first.")
    with open(CHAINS_FILE) as f:
        chains_data = json.load(f)
    s1_analysis = chains_data["stage1_analysis"]
    all_chains = chains_data["chains"]
    print(f"  {len(all_chains)} passages with Q&A chains")

    # Model consistency check — all stages must use the same model
    chain_model = chains_data.get("model", "")
    if chain_model and chain_model != MODEL_NAME:
        print(f"\n  *** WARNING: Stage 2-1 used model '{chain_model}' but Stage 2-2 uses '{MODEL_NAME}' ***")
        print(f"  *** All stages must use the same model for valid results ***\n")

    # Load Stage 1 for peaks (source window)
    s1_file_raw = chains_data.get("stage1_file", str(RESULTS_DIR / "stage1_results.json"))
    s1_file = Path(s1_file_raw)
    if not s1_file.exists():
        s1_file = RESULTS_DIR / "stage1_results.json"
        print(f"  stage1_file '{s1_file_raw}' not found, using {s1_file}")
    with open(s1_file) as f:
        s1 = json.load(f)
    s1_by_topic = {r["topic"]: r for r in s1["results"]}

    s1_model = s1.get("model", "")
    if s1_model and s1_model != MODEL_NAME:
        print(f"\n  *** WARNING: Stage 1 used model '{s1_model}' but Stage 2-2 uses '{MODEL_NAME}' ***")
        print(f"  *** All stages must use the same model for valid results ***\n")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"Model ready in {time.time()-t0:.1f}s")

    # Run consistency checks
    all_detail = []
    accept_qa = []

    print(f"\n{'='*70}")
    print(f"STAGE 2-2: CONSISTENCY CHECK [{len(all_chains)} passages]")
    print(f"{'='*70}")

    for idx, chain_entry in enumerate(all_chains):
        topic = chain_entry["topic"]
        s1_result = s1_by_topic.get(topic, {})

        print(f"\n{'~'*60}")
        print(f"[{idx+1}/{len(all_chains)}] [{chain_entry['label']:7s}] {topic}  "
              f"S_k={chain_entry['S_k']:.3f}  {len(chain_entry['pairs'])} pairs")

        result = process_passage(model, tokenizer, chain_entry, s1_result)
        all_detail.append(result)

        for ap in result["accepted_pairs"]:
            accept_qa.append(ap)

        tag = "Y" if result["correct"] else ("N" if result["correct"] is False else "?")
        strang = f"  strange@Q{result['strangeness']['break_at']+1}" if result.get('strangeness') else ""
        brk_str = f"break@{result['break_point']}" if result['break_point'] is not None else "full"
        print(f"  => k={result['n_passed']}/{result['n_generated']}  {brk_str}  "
              f"[{tag}]  [{result['time_sec']:.1f}s]{strang}")

    # Analysis
    print(f"\n{'='*70}\nANALYSIS\n{'='*70}")

    is_accept_fn = lambda d: d["break_point"] is None and d["n_passed"] > 0

    vc = sum(1 for d in all_detail if d["correct"])
    vt = sum(1 for d in all_detail if d["correct"] is not None)
    if vt > 0: print(f"\n  Overall: {vc}/{vt} ({vc/vt:.0%})")

    for lb in ["novel", "corrupt"]:
        items = [d for d in all_detail if d["label"] == lb and d["correct"] is not None]
        if items:
            acc = sum(1 for d in items if d["correct"]) / len(items)
            print(f"    {lb:10s}: {sum(1 for d in items if d['correct'])}/{len(items)} ({acc:.0%})")

    print(f"\n  Outcome distribution:")
    for lb in ["novel", "corrupt"]:
        items = [d for d in all_detail if d["label"] == lb]
        n_full = sum(1 for d in items if d["break_point"] is None and d["n_passed"] > 0)
        n_break = sum(1 for d in items if d["break_point"] is not None)
        print(f"    {lb:10s}: full_chain={n_full}  break={n_break}")

    print(f"\n  Chain break depth:")
    for lb in ["novel", "corrupt"]:
        items = [d for d in all_detail if d["label"] == lb and d["break_point"] is not None]
        if items:
            depths = [d["break_point"] for d in items]
            tags = [d.get("break_tag", "?") for d in items]
            print(f"    {lb:10s}: breaks at {depths}  mean={np.mean(depths):.1f}  tags={tags}")
        else:
            print(f"    {lb:10s}: no breaks")

    n_qa_items = sum(1 for x in accept_qa if x.get("type") == "qa")
    n_sw_items = sum(1 for x in accept_qa if x.get("type") == "source_window")
    n_str_items = sum(1 for x in accept_qa if x.get("type") == "strangeness")
    n_strangeness = sum(1 for d in all_detail if d.get("strangeness"))
    print(f"\n  Training set F_k:")
    print(f"    Accept: {n_qa_items} Q&A + {n_sw_items} source windows + {n_str_items} strangeness = {len(accept_qa)} items")
    print(f"    Strangeness records: {n_strangeness}")

    # Parse strategy audit
    all_checks = [cr for d in all_detail for cr in d["chain"]]
    strat_counts = Counter(cr.get("parse_strategy", "unknown") for cr in all_checks)
    n_checks = len(all_checks)
    print(f"\n  Parse strategy (n={n_checks}):")
    for strat in ["verdict_line", "standalone_word", "no_signal"]:
        cnt = strat_counts.get(strat, 0)
        pct = cnt / n_checks * 100 if n_checks > 0 else 0
        print(f"    {strat:20s}: {cnt:4d} ({pct:5.1f}%)")

    print(f"\n  {'Topic':30s} {'Lbl':5s} {'S_k':>5s} {'d_k':>5s} "
          f"{'k':>3s} {'N':>3s} {'brk':>4s} {'brk_tag':>8s} {'ok':>3s}")
    print(f"  {'-'*70}")
    for d in all_detail:
        brk = str(d["break_point"]) if d["break_point"] is not None else "-"
        brk_tag = (d.get("break_tag") or "-")[:8]
        tag = "Y" if d["correct"] else ("N" if d["correct"] is False else "?")
        print(f"  {d['topic']:30s} {d['label'][:5]:5s} "
              f"{d['S_k']:5.2f} {d['d_k']:5.2f} "
              f"{d['n_passed']:3d} {d['n_generated']:3d} {brk:>4s} {brk_tag:>8s}  {tag}")

    # Retain Q&A
    retain_qa = [
        {"question": "What is the chemical formula for water?", "answer": "H2O"},
        {"question": "What is the speed of light in a vacuum?", "answer": "Approximately 299,792,458 meters per second."},
        {"question": "How many chromosomes do humans have?", "answer": "46 chromosomes, in 23 pairs."},
        {"question": "What is the capital of France?", "answer": "Paris."},
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare."},
        {"question": "What is the hardest natural material?", "answer": "Diamond."},
        {"question": "When did World War II end?", "answer": "1945."},
        {"question": "What planet is closest to the Sun?", "answer": "Mercury."},
        {"question": "What is the chemical symbol for gold?", "answer": "Au."},
        {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci."},
    ]

    output = {
        "model": MODEL_NAME,
        "backend": "MLX (M3 Pro 36GB)",
        "pipeline": "Q&A chain (mechanism-based) -> sequential check -> k",
        "params": {},
        "stage1_analysis": s1_analysis,
        "detail": all_detail,
        "accept": accept_qa, "retain": retain_qa,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_FILE}")

    print(f"\nGenerating plots...")
    plot_all(all_detail)

    print(f"\n  Total: {(time.time()-t_global)/60:.1f} min")
    print(f"\n  Next: python3 lscp3.py 0.98")
    print(f"\n{'='*70}\nDONE\n{'='*70}")

if __name__ == "__main__":
    main()
