"""Report appendices (no model, no GPU):
  A. all 48 disagreements of the 20-prompt comparison with a one-line reason each (human-picked for 9, rule-based for the rest,
     clearly labelled) + outcome counts per condition for both methods
  B. complete metrics table: AUROC, AUPRC, PCC with 95% CIs (bootstrap over questions) for every detector and condition
Outputs: artifacts/results/llama2-7b/comparison/comparison_20_disagree_all48.{csv,md}, tables/metrics_full_with_ci.{csv,md}
"""
import json
import numpy as np
import pandas as pd
from src.detect.common import auc

R = "artifacts/results/llama2-7b/"
sc = pd.read_parquet(R + "scores.parquet")
lab = pd.DataFrame([json.loads(l) for l in open("artifacts/runs/llama2-7b/labeled_std.jsonl", encoding="utf-8")])
gen = {json.loads(l)["uid"]: json.loads(l) for l in open("artifacts/runs/llama2-7b/gen_std.jsonl", encoding="utf-8")}
thr = json.load(open(R + "thresholds.json"))
cmp_ = pd.read_csv(R + "comparison/comparison_20.csv").fillna("")
ORIG, OURS = "s_redeep_c1", "s_redeep"

# ------------------------------------------------------------------ A. disagreements
fold = dict(zip(sc.uid, sc.fold))
dis = cmp_[cmp_.orig_flag != cmp_.ours_flag].copy()
rows = []
for r in dis.itertuples():
    uid = f"{r.qid}__{r.condition}"
    f = str(int(fold[uid]))
    mo, mu = float(r.orig_score) - thr[ORIG][f], float(r.ours_score) - thr[OURS][f]
    ans = str(r.answer)
    ntok = gen[uid]["n_answer_tokens"]
    pre = ans.lower().startswith(("according to", "based on", "from passage", "passage"))
    right = "original" if bool(r.orig_flag) == bool(r.failure) else "ours"
    if min(abs(mo), abs(mu)) <= 0.02:
        why = f"Score within 0.02 of the cut-off for one version (margins {mo:+.3f} orig, {mu:+.3f} ours), so a small difference flips the call"
    elif r.condition == "C4_missing":
        why = "Evidence is missing, so the answer is wrong whatever the signals say; the two versions use different cut-offs and head sets"
    elif ntok <= 6:
        why = f"Very short answer ({ntok} tokens), so the token average is noisy"
    elif pre:
        why = "Long answer that starts with a citation preamble, which dilutes the averaged scores"
    else:
        why = "Different head/layer selections and cut-offs give different scores on this answer"
    rows.append(dict(n=len(rows) + 1, prompt=r.question, condition=r.condition, answer=ans[:80].replace("\n", " "), truth="wrong" if r.failure else "correct",
                     original="abstain" if r.orig_flag else "answer", ours="abstain" if r.ours_flag else "answer",
                     correct_call=right, margin_orig=round(mo, 3), margin_ours=round(mu, 3), n_tokens=ntok, why=why, why_source="rule-based"))
d = pd.DataFrame(rows)
human = {1: "Short answer: only a few tokens, so the average is noisy", 2: "Long, hedged answer with a citation preamble, which shifts the averaged scores",
         3: "Long, hedged answer with a citation preamble, which shifts the averaged scores", 4: "Passages are misleading or missing evidence, so internal signals are not informative",
         5: "Passages are misleading or missing evidence, so internal signals are not informative",
         6: "The two detectors use different cut-offs: the original was tuned on clean passages, ours on all five conditions",
         7: "Long, hedged answer with a citation preamble, which shifts the averaged scores", 8: "Long, hedged answer with a citation preamble, which shifts the averaged scores",
         9: "Passages are misleading or missing evidence, so internal signals are not informative"}
for k, v in human.items():
    d.loc[d.n == k, "why"] = v
    d.loc[d.n == k, "why_source"] = "human-picked"
d.to_csv(R + "comparison/comparison_20_disagree_all48.csv", index=False)

summ = pd.read_csv(R + "comparison/comparison_20_summary.csv")
L = ["# 20-prompt comparison: all disagreements with a one-line reason\n",
     f"{len(d)} disagreements out of 100 cells. `why_source` = human-picked (cards 1-9, chosen by the project owner from a fixed list) or rule-based "
     "(computed from the score margins to each version's fold cut-off, the answer length and whether the answer starts with a citation preamble). "
     "All reasons are hypotheses read from the token-level dump and scores, not tested causes. `margin` = score minus the cut-off of the fold the question belongs to. "
     "Caveat: scores sit within 0.02 of the cut-off for 31% of all 750 rows (original) and 50% (ours), so most disagreements are near-threshold flips rather than genuine differences in judgement.\n",
     "| # | Prompt | Cond. | LLM answer | Truth | Orig. | Ours | Right call | Reason | Source |", "|---|---|---|---|---|---|---|---|---|---|"]
for r in d.itertuples():
    L.append(f"| {r.n} | {r.prompt[:55]} | {r.condition[:2]} | {r.answer[:40].replace('|', '/')} | {r.truth} | {r.original} | {r.ours} | {r.correct_call} | {r.why} | {r.why_source} |")
L.append("\n## Outcome counts per condition (20 prompts)\n")
L.append("Outcomes: kept-correct (answered, right), caught-failure (abstained, would have been wrong), missed-failure (answered, wrong), false-abstain (abstained, would have been right).\n")
L.append(summ.to_markdown(index=False))
open(R + "comparison/comparison_20_disagree_all48.md", "w", encoding="utf-8").write("\n".join(L))
print("A:", len(d), d.why_source.value_counts().to_dict(), d.why.str[:40].value_counts().to_dict())

# ------------------------------------------------------------------ B. full metrics with CIs
m = pd.read_csv(R + "tables/e2_metrics_long.csv")
y = sc.set_index("uid").failure
rng = np.random.default_rng(0)
qids = np.array(sorted(sc.qid.unique()))
idx_by_q = {q: np.where(sc.qid.values == q)[0] for q in qids}
boots = [rng.integers(0, len(qids), len(qids)) for _ in range(1000)]
names = m.drop_duplicates("detector").set_index("detector").name.to_dict()


def pcc(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


out = []
for det in m.detector.unique():
    if det not in sc.columns:
        continue
    for cond in ["all", "C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]:
        sub = np.arange(len(sc)) if cond == "all" else np.where(sc.condition.values == cond)[0]
        s_all, y_all = sc[det].values.astype(float), sc.failure.values.astype(float)
        base = pcc(s_all[sub], y_all[sub])
        qsub = {q: np.intersect1d(idx_by_q[q], sub) for q in qids}
        vals = []
        for b in boots:
            ii = np.concatenate([qsub[qids[k]] for k in b])
            v = pcc(s_all[ii], y_all[ii])
            if not np.isnan(v):
                vals.append(v)
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) > 50 else (np.nan, np.nan))
        row = m[(m.detector == det) & (m.condition == cond)].iloc[0]
        out.append(dict(detector=det, name=names[det], condition=cond, n=int(row.n), n_fail=int(row.n_fail),
                        auroc=row.auroc, auroc_lo=row.auroc_lo, auroc_hi=row.auroc_hi,
                        auprc=row.auprc, auprc_lo=row.auprc_lo, auprc_hi=row.auprc_hi,
                        pcc=base, pcc_lo=lo, pcc_hi=hi, aurc=row.aurc))
o = pd.DataFrame(out)
o.to_csv(R + "tables/metrics_full_with_ci.csv", index=False)


def f(v, lo, hi):
    return "n/a" if pd.isna(v) else (f"{v:.3f} [{lo:.3f}, {hi:.3f}]" if not pd.isna(lo) else f"{v:.3f}")


M = ["# Appendix: complete metrics (pooled out-of-fold predictions, 95% bootstrap CIs over questions, 1000 resamples)\n",
     "AUROC and AUPRC are n/a where a condition has fewer than 10 failures or 10 successes (C4). Base rate of failure per condition: C1 0.107, C2 0.160, C3 0.213, C4 0.980, C5 0.327, all 0.357 (AUPRC is only meaningful against these).\n"]
for cond in ["all", "C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]:
    g = o[o.condition == cond]
    M.append(f"\n## {cond}  (n={int(g.n.iloc[0])}, failures={int(g.n_fail.iloc[0])})\n")
    M.append("| detector | AUROC [95% CI] | AUPRC [95% CI] | PCC [95% CI] | AURC |")
    M.append("|---|---|---|---|---|")
    for r in g.itertuples():
        M.append(f"| {r.name} | {f(r.auroc, r.auroc_lo, r.auroc_hi)} | {f(r.auprc, r.auprc_lo, r.auprc_hi)} | {f(r.pcc, r.pcc_lo, r.pcc_hi)} | {'n/a' if pd.isna(r.aurc) else f'{r.aurc:.3f}'} |")
open(R + "tables/metrics_full_with_ci.md", "w", encoding="utf-8").write("\n".join(M))
print("B:", len(o), "rows; PCC CI missing:", int(o.pcc_lo.isna().sum()))
