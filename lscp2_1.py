"""
LSCP Stage 2-1: Generate Q&A Chains
========================================
Generates progressive Q&A chains for flagged passages.
Run lscp2_2.py to check consistency.

INPUT:  {RESULTS_DIR}/stage1_results.json
OUTPUT: {RESULTS_DIR}/stage2_qa_chains.json

Usage:
    python lscp1.py      # Stage 1 first
    python lscp2_1.py    # generate questions
    python lscp2_2.py    # check and score
"""

import time, json, re, numpy as np
from pathlib import Path
from config import *

STAGE1_FILE = RESULTS_DIR / "stage1_results.json"
OUTPUT_FILE = RESULTS_DIR / "stage2_qa_chains.json"

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

# -- Q&A Chain Generation -----------------------------------------

QA_GEN_PROMPT = '''You are verifying whether the following passage makes trustworthy claims.
Generate up to {n} question-answer pairs. Tag each with its epistemic origin.

IMPORTANT: Do NOT ask whether the passage's events have happened — you may not know
recent events. Instead, test whether the MECHANISMS the passage describes are plausible.

Step 1 — [existing]: Confirm background facts the passage relies on.
  "What is X?" "Is Y a known property of Z?" — basic foundations.

Step 2 — [mechanism]: Extract each specific causal claim the passage makes and test
  whether that mechanism is physically/biologically/logically possible given what you know.
  Be SPECIFIC. Do not ask "is this consistent with the field?" — instead ask about the
  exact causal step: "Can protein X actually bind to receptor Y?" "Does blocking pathway Z
  lead to effect W, or the opposite?" "Is a 99.97% efficiency physically achievable given
  the thermodynamic limits of this process?"
  A wrong passage will contain at least one mechanism that contradicts established science.

Step 3 — [implication]: If the passage's claims are true, what specific, testable
  consequences follow? Do those consequences conflict with anything known?
  "If X causes Y, then we should also see Z — do we?"

For [mechanism] questions: pick apart EACH causal step the passage chains together.
If the passage says "A causes B which leads to C which enables D", ask separately:
  - Does A actually cause B? (based on known science)
  - Does B lead to C? (or does B actually lead to the opposite of C?)
  - Is D achievable through C? (or are there known barriers?)

Format each pair exactly as:
Q: [tag] question text
A: answer text

Generate exactly {n} question-answer pairs. Do not generate more or fewer than {n}.

Passage: "{passage}"
'''

def _parse_tag(question_text):
    import re
    m = re.match(r'\[([a-z]+)\]\s*(.*)', question_text.strip(), re.IGNORECASE)
    if m:
        raw_tag = m.group(1).lower()
        text = m.group(2).strip()
        if raw_tag in ("existing", "background", "known"):
            return text, "existing"
        elif raw_tag in ("mechanism", "conflict", "consistency", "clash", "causal"):
            return text, "mechanism"
        elif raw_tag in ("implication", "implies", "follow"):
            return text, "implication"
        else:
            return text, "unknown"
    return question_text.strip(), "unknown"

def generate_qa_chain(model, tokenizer, text, n_qa):
    prompt = QA_GEN_PROMPT.format(passage=text, n=n_qa)
    response = gen(model, tokenizer, prompt, max_tokens=n_qa * 200, temperature=0.7)
    raw_pairs = parse_qa_pairs(response)

    pairs = []
    for pair in raw_pairs:
        clean_q, tag = _parse_tag(pair["question"])
        if not clean_q.strip() or not pair["answer"].strip():
            continue
        pairs.append({
            "question": clean_q,
            "answer": pair["answer"],
            "tag": tag,
        })

    tag_counts = {}
    for p in pairs:
        tag_counts[p["tag"]] = tag_counts.get(p["tag"], 0) + 1

    return pairs, {"tag_counts": tag_counts, "n": len(pairs)}

def parse_qa_pairs(resp):
    import re
    pairs = []; cur_q, cur_a = None, []
    q_re = re.compile(r'^\*{0,2}Q\d*[.:]\*{0,2}\s*(.*)', re.IGNORECASE)
    a_re = re.compile(r'^\*{0,2}A\d*[.:]\*{0,2}\s*(.*)', re.IGNORECASE)
    for line in resp.strip().split("\n"):
        line = line.strip()
        if not line: continue
        mq = q_re.match(line)
        if mq:
            if cur_q and cur_a:
                pairs.append({"question": cur_q, "answer": " ".join(cur_a)})
            cur_q = mq.group(1).strip(); cur_a = []; continue
        ma = a_re.match(line)
        if ma:
            cur_a = [ma.group(1).strip()]; continue
        if cur_q: cur_a.append(line)
    if cur_q and cur_a:
        pairs.append({"question": cur_q, "answer": " ".join(cur_a)})
    return pairs

def compute_n_qa(s_k):
    return max(MIN_QA, min(MAX_QA, round(s_k * 4)))

def build_source_window(tokenizer, text, peaks, w_ext=50):
    """Extract high-surprisal region for Q&A generation. Returns (window_text, is_full)."""
    if not peaks:
        return text, True
    peak_positions = [p["pos"] for p in peaks]
    min_peak, max_peak = min(peak_positions), max(peak_positions)
    tokens = tokenizer.encode(text)
    window_start = max(0, min_peak - w_ext)
    window_end = min(max_peak + w_ext + 1, len(tokens))
    window_tokens = tokens[window_start:window_end]
    if len(window_tokens) >= len(tokens) * 0.8:
        return text, True
    return tokenizer.decode(window_tokens), False

# -- Main ----------------------------------------------------------

def main():
    t_global = time.time()

    # Load Stage 1
    print(f"Loading Stage 1 results from {STAGE1_FILE}...")
    if not STAGE1_FILE.exists():
        raise FileNotFoundError(f"{STAGE1_FILE} not found. Run lscp1.py first.")
    with open(STAGE1_FILE) as f:
        s1 = json.load(f)
    s1_analysis = s1["analysis"]
    thr = s1_analysis.get("threshold", 2.0)
    print(f"  {s1['n_passages']} passages, model: {s1['model']}")
    print(f"  threshold={thr:.3f}")

    # Load passages
    import importlib
    mod = importlib.import_module(PASSAGE_FILE)
    text_map = {p["topic"]: p["text"] for p in mod.PASSAGES}

    # Filter flagged
    flagged = [r for r in s1["results"] if r["S_k"] > thr]
    for r in flagged: r["_text"] = text_map.get(r["topic"], "")
    n_n = sum(1 for r in flagged if r["label"] == "novel")
    n_c = sum(1 for r in flagged if r["label"] == "corrupt")
    n_a = sum(1 for r in flagged if r["label"] == "known")
    print(f"  Flagged: {len(flagged)} (novel={n_n} corrupt={n_c} aligned={n_a})")

    # Load model
    print(f"\nLoading model...")
    t0 = time.time()
    model, tokenizer = load_model()
    print(f"Model ready in {time.time()-t0:.1f}s")

    # Generate Q&A chains
    all_chains = []

    print(f"\n{'='*70}")
    print(f"STAGE 2-1: GENERATE Q&A CHAINS [{len(flagged)} passages]")
    print(f"{'='*70}")

    for idx, r in enumerate(flagged):
        text = r["_text"]
        if not text: continue

        topic = r["topic"]
        s_k = r["S_k"]
        n_qa = compute_n_qa(s_k)

        # Use source window (high-surprisal region) for Q&A generation
        peaks = r.get("peaks", [])
        window_text, is_full = build_source_window(tokenizer, text, peaks)

        print(f"\n[{idx+1}/{len(flagged)}] [{r['label']:7s}] {topic}  S_k={s_k:.3f}  n_qa={n_qa}"
              f"  window={'full' if is_full else f'{len(window_text.split())}w'}")

        t = time.time()
        pairs, chain_info = generate_qa_chain(model, tokenizer, window_text, n_qa)
        dt = time.time() - t
        tc = chain_info["tag_counts"]

        if len(pairs) < 3:
            extra_pairs, _ = generate_qa_chain(model, tokenizer, window_text, n_qa)
            for ep in extra_pairs:
                ep["tag"] = "supplementary"
            pairs.extend(extra_pairs)
            print(f"  +{len(extra_pairs)} supplemented = {len(pairs)} total")

        print(f"  {len(pairs)} pairs  "
              f"[exi={tc.get('existing',0)} mec={tc.get('mechanism',0)} "
              f"imp={tc.get('implication',0)} unk={tc.get('unknown',0)}]  "
              f"[{dt:.1f}s]")

        for i, p in enumerate(pairs):
            print(f"    Q{i+1} [{p['tag'][:3].upper()}] {p['question'][:65]}")

        all_chains.append({
            "topic": topic,
            "label": r["label"],
            "domain": r.get("domain", ""),
            "S_k": s_k,
            "d_k": r["d_k"],
            "pairs": pairs,
            "chain_info": chain_info,
            "n_qa_requested": n_qa,
            "gen_time": round(dt, 1),
        })

    # Save
    output = {
        "model": MODEL_NAME,
        "stage1_file": str(STAGE1_FILE),
        "stage1_analysis": s1_analysis,
        "n_flagged": len(flagged),
        "chains": all_chains,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Total: {(time.time()-t_global)/60:.1f} min")
    print(f"\nNext: python3 lscp2_2.py")

if __name__ == "__main__":
    main()
