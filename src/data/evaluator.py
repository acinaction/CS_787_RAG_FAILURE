"""P2: Merge generated answers with benchmark and produce labeled files.

Usage:
  python -m src.data.evaluator \
      --gen   artifacts/runs/llama2-7b/gen_std.jsonl \
      --bench artifacts/data/benchmark_v2.jsonl \
      --out   artifacts/runs/llama2-7b/labeled_std.jsonl
"""
import argparse, json, re, string
from collections import Counter

# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------
REFUSAL_PATTERNS = [
    r"i don'?t know", r"cannot answer", r"not mentioned", r"no information",
    r"not provided", r"cannot determine", r"unclear", r"no passage",
    r"not stated", r"insufficient", r"don'?t have", r"not available",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

def is_refusal(text: str) -> bool:
    if not text or not text.strip():
        return True
    return bool(REFUSAL_RE.search(text))

# ---------------------------------------------------------------------------
# Answer normalization and F1
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = " ".join(text.split())
    return text

def token_f1(pred: str, gold: str) -> float:
    pred_toks  = Counter(normalize(pred).split())
    gold_toks  = Counter(normalize(gold).split())
    common     = sum((pred_toks & gold_toks).values())
    if common == 0:
        return 0.0
    prec = common / sum(pred_toks.values())
    rec  = common / sum(gold_toks.values())
    return 2 * prec * rec / (prec + rec)

def best_f1(pred: str, golds: list) -> float:
    return max((token_f1(pred, g) for g in golds), default=0.0)

def contains_tokens(pred: str, gold: str) -> bool:
    """Normalized gold appears in the normalized answer as a whole-token sequence
    (gold "91" does NOT match "1991", "art" does not match "artistic", "city1" does not match "city10")."""
    g = normalize(gold)
    return bool(g) and f" {g} " in f" {normalize(pred)} "

def is_correct(pred: str, golds: list, f1_threshold: float = 0.5) -> bool:
    """Plan rule (EXECUTION_PLAN 3.4): normalized gold contained in the answer OR max token-F1 >= 0.5.
    Containment is checked on token boundaries; the F1 part is unchanged."""
    return any(contains_tokens(pred, g) for g in golds) or best_f1(pred, golds) >= f1_threshold

def is_correct_legacy(pred: str, golds: list, f1_threshold: float = 0.5) -> bool:
    """Day-1 rule (raw substring containment). Kept ONLY to report how many labels the boundary fix changes."""
    n_pred = normalize(pred)
    return any(normalize(g) and normalize(g) in n_pred for g in golds) or best_f1(pred, golds) >= f1_threshold

def followed_counterfactual(pred: str, swapped_entity: str) -> bool:
    if not swapped_entity:
        return False
    return contains_tokens(pred, swapped_entity)

# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------
def label_row(gen: dict, bench: dict) -> dict:
    ans  = gen.get("answer", "")
    cond = bench["condition"]
    golds = bench["gold_answers"]

    refusal  = is_refusal(ans)
    answered = not refusal
    f1       = best_f1(ans, golds) if answered else 0.0
    correct  = int(is_correct(ans, golds)) if answered else 0

    if cond == "C4_missing":
        failure = int(answered)   # failure = model answered when it should abstain
    else:
        failure = int(not correct)

    cf_entity = bench.get("swapped_entity")
    fc = False
    if cond == "C5_counterfactual" and answered and cf_entity:
        fc = followed_counterfactual(ans, cf_entity)

    correct_legacy = int(is_correct_legacy(ans, golds)) if answered else 0
    failure_legacy = int(answered) if cond == "C4_missing" else int(not correct_legacy)

    out = dict(gen)
    out.update({"qid": bench["qid"], "condition": cond, "refusal": refusal, "answered": answered, "f1": round(f1, 4),
                "correct": correct, "failure": failure,
                "followed_counterfactual": fc,
                # audit only (not used by any detector): label under the Day-1 substring rule
                "correct_legacy_substring": correct_legacy, "failure_legacy_substring": failure_legacy})
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Label generated answers")
    ap.add_argument("--gen",   required=True, help="gen_std.jsonl or gen_abstain.jsonl")
    ap.add_argument("--bench", required=True, help="benchmark_v2.jsonl")
    ap.add_argument("--out",   required=True, help="labeled output .jsonl")
    a = ap.parse_args()

    bench = {}
    for l in open(a.bench, encoding="utf-8"):
        r = json.loads(l); bench[r["uid"]] = r
    gen   = [json.loads(l) for l in open(a.gen, encoding="utf-8")]

    with open(a.out, "w", encoding="utf-8") as f:
        for gr in gen:
            br = bench.get(gr["uid"])
            if br is None:
                print(f"[evaluator] WARNING: uid {gr['uid']} not in benchmark — skipping")
                continue
            rec = label_row(gr, br)
            f.write(json.dumps(rec) + "\n")

    # Quick summary
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    for cond in ["C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        n = len(sub)
        fail_r = sum(r["failure"] for r in sub) / n
        ref_r  = sum(r["refusal"] for r in sub) / n
        print(f"  {cond:25s}  n={n:4d}  failure={fail_r:.2f}  refusal={ref_r:.2f}")

    # How many labels does the token-boundary fix change? (reported, never hidden)
    ch = {}
    for cond in ["C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]:
        sub = [r for r in rows if r["condition"] == cond]
        ch[cond] = {"n": len(sub), "correct_changed": sum(r["correct"] != r["correct_legacy_substring"] for r in sub),
                    "failure_changed": sum(r["failure"] != r["failure_legacy_substring"] for r in sub),
                    "mean_f1": round(sum(r["f1"] for r in sub) / max(len(sub), 1), 4)}
    ch["examples"] = [{"uid": r["uid"], "answer": r["answer"][:120]} for r in rows
                      if r["correct"] != r["correct_legacy_substring"]][:20]
    json.dump(ch, open(a.out.replace(".jsonl", "_label_changes.json"), "w"), indent=1)
    print("  label changes vs substring rule:", {k: v["failure_changed"] for k, v in ch.items() if k != "examples"})
    print(f"[evaluator] Done -> {a.out}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
