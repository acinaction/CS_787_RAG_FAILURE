"""E8: the 20-prompt side-by-side (plan §8). Original = s_redeep_c1, Ours = s_redeep, both out-of-fold,
each with its own fold's F1-optimal threshold tau (flag/abstain when score >= tau).

Outputs (into --out):
  comparison_20.csv          one row per (prompt, condition): plan §8 columns + why_draft
  comparison_20_summary.csv  outcome counts per side per condition
  comparison_20_disagree.md  table `Prompt | Condition | Original says | Ours says | Truth | Why` for every disagreement;
                             `Why` is an automatic DRAFT from the token-level ECS/PKS dump — P2 edits it by hand.
Usage:
  python -m src.analysis.make_comparison --scores ... --labeled_std ... --bench artifacts/data/benchmark_v2.jsonl \
      --cmp_set artifacts/data/comparison_set_v2.json --tokens .../redeep_tokens_cmp.json --out .../comparison
"""
import argparse, json, os
import numpy as np
import pandas as pd
from src.analysis.make_tables import outcomes, flags_for
from src.analysis.metrics import COND_ORDER
from src.detect.common import load_labels, read_jsonl

SAY = {True: "abstain (likely wrong)", False: "answer"}


def why_draft(row, tok, fit_log):
    """Short machine draft explaining the disagreement from the token dump (to be edited by a human)."""
    parts = []
    if tok is not None and len(tok.get("ecs_tok", [])):
        e = np.asarray(tok["ecs_tok"], float); p = np.asarray(tok["pks_tok"], float)
        parts.append(f"{len(e)} tok; mean ECS={e.mean():.3f}, mean PKS={p.mean():.1f}, late-PKS={p[:, p.shape[1] // 2:].mean():.1f}")
    if row["condition"] in ("C4_missing", "C5_counterfactual"):
        parts.append("context misleading/missing: high copying does not imply correct")
    who = "ours" if row["ours_flag"] == bool(row["failure"]) else "orig"
    parts.append(f"{who} side right")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    for k in ["scores", "labeled_std", "bench", "cmp_set", "out"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--orig", default="s_redeep_c1"); ap.add_argument("--ours", default="s_redeep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rdir = os.path.dirname(os.path.abspath(a.scores))
    thr = json.load(open(os.path.join(rdir, "thresholds.json")))
    scores = pd.read_parquet(a.scores)
    lab = load_labels(a.labeled_std).set_index("uid")
    bench = {r["uid"]: r for r in read_jsonl(a.bench)}
    cmp_q = json.load(open(a.cmp_set))["qids"]
    toks = json.load(open(a.tokens))["rows"] if a.tokens and os.path.exists(a.tokens) else {}
    fit_log = []

    s = scores[scores.qid.isin(cmp_q)].copy()
    s["orig_flag"] = flags_for(s, a.orig, thr); s["ours_flag"] = flags_for(s, a.ours, thr)
    s["orig_outcome"] = outcomes(s.failure.values, s.orig_flag.values)
    s["ours_outcome"] = outcomes(s.failure.values, s.ours_flag.values)
    rows = []
    for _, r in s.iterrows():
        b, l = bench[r.uid], lab.loc[r.uid]
        rows.append({"qid": r.qid, "question": b["question"], "condition": r.condition, "answer": l["answer"],
                     "gold": " | ".join(b["gold_answers"]), "correct": int(l["correct"]), "failure": int(r.failure),
                     "orig_score": round(float(r[a.orig]), 4), "orig_flag": bool(r.orig_flag), "orig_outcome": r.orig_outcome,
                     "ours_score": round(float(r[a.ours]), 4), "ours_flag": bool(r.ours_flag), "ours_outcome": r.ours_outcome,
                     "probe_score": round(float(r["s_probe_best"]), 4) if "s_probe_best" in r else np.nan})
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df.condition, COND_ORDER)
    df = df.sort_values(["qid", "condition"]).reset_index(drop=True)
    df["why_draft"] = [why_draft(r, toks.get(f"{r.qid}__{r.condition}"), fit_log) if r.orig_flag != r.ours_flag else ""
                       for _, r in df.iterrows()]
    assert len(df) == 5 * len(cmp_q), f"comparison grid incomplete: {len(df)} rows"
    df.to_csv(f"{a.out}/comparison_20.csv", index=False)

    summ = []
    for side in ["orig", "ours"]:
        for c in COND_ORDER + ["all"]:
            g = df if c == "all" else df[df.condition == c]
            vc = g[f"{side}_outcome"].value_counts()
            summ.append({"side": side, "condition": c, **{k: int(vc.get(k, 0)) for k in
                         ["kept-correct", "caught-failure", "missed-failure", "false-abstain"]}})
    pd.DataFrame(summ).to_csv(f"{a.out}/comparison_20_summary.csv", index=False)

    dis = df[df.orig_flag != df.ours_flag]
    lines = ["| Prompt | Condition | Original says | Ours says | Truth | Why (draft — edit by hand) |", "|---|---|---|---|---|---|"]
    for _, r in dis.iterrows():
        q = r.question.replace("|", "/")[:70]
        truth = "failure" if r.failure else "correct/abstained"
        lines.append(f"| {q} | {r.condition} | {SAY[r.orig_flag]} | {SAY[r.ours_flag]} | {truth} (ans: {str(r.answer)[:30]}) | {r.why_draft} |")
    open(f"{a.out}/comparison_20_disagree.md", "w", encoding="utf-8").write(
        f"# 20-prompt comparison: {len(dis)} disagreements of {len(df)} cells\n\n" + "\n".join(lines) + "\n")
    print(f"[make_comparison] {len(df)} cells, {len(dis)} disagreements -> {a.out}")
    if len(dis) < 8:
        print("[make_comparison] NOTE: fewer than 8 disagreements; plan asks to annotate at least 8 (use all that exist).")


if __name__ == "__main__":
    main()
