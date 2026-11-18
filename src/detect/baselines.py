"""P3: Unfitted baselines + the condition-prior reference.

  s_perplexity        exp(-mean answer-token logprob)
  s_lnentropy         length-normalised predictive entropy = mean over answer tokens of H(p_t)
                      (falls back to -mean logprob if the gen file has no answer_token_entropies)
  s_firsttok_entropy  entropy of the first generated token's distribution
  s_bm25_margin       -(BM25 top1 - top2) over the shown passages (small margin = ambiguous retrieval)
  s_cond_prior        REFERENCE, NOT DEPLOYABLE: failure rate of the row's condition in the training folds.
                      Shows how much pooled AUROC comes from merely recognising the noise type.
Thresholds (for the comparison / abstention operating point): F1-optimal on outer-training rows.
Empty answers have no tokens: perplexity/entropy are set to the max observed value (most uncertain).

Usage:
  python -m src.detect.baselines --labels .../labeled_std.jsonl --bench artifacts/data/benchmark_v2.jsonl \
      --folds artifacts/data/folds_v2.json --scores .../scores.parquet
"""
import argparse, json, os
import numpy as np
import pandas as pd
from src.detect.common import base_table, f1_threshold, load_labels, merge_scores, read_jsonl, save_thresholds, N_OUTER


def perplexity(lps):
    return float(np.exp(-np.mean(lps))) if len(lps) else np.nan


def ln_entropy(r):
    ents = r.get("answer_token_entropies")
    if ents is not None:
        return float(np.mean(ents)) if len(ents) else np.nan
    lps = r.get("answer_token_logprobs") or []
    return float(-np.mean(lps)) if len(lps) else np.nan


def compute(labels, bench, folds):
    t = base_table(labels, folds)
    lab = labels.set_index("uid")
    t["s_perplexity"] = [perplexity(lab.at[u, "answer_token_logprobs"] or []) for u in t.uid]
    t["s_lnentropy"] = [ln_entropy(lab.loc[u].to_dict()) for u in t.uid]
    t["s_firsttok_entropy"] = [float(lab.at[u, "first_token_entropy"]) for u in t.uid]
    t["s_bm25_margin"] = [-float(bench[u]["bm25_margin"]) for u in t.uid]
    for c in ["s_perplexity", "s_lnentropy", "s_firsttok_entropy"]:
        t[c] = t[c].fillna(t[c].max())
    prior = np.full(len(t), np.nan)
    for k in range(N_OUTER):
        tr = t.fold != k
        rate = t[tr].groupby("condition")["failure"].mean()
        prior[(t.fold == k).values] = t.loc[t.fold == k, "condition"].map(rate).values
    t["s_cond_prior"] = prior
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--folds", required=True); ap.add_argument("--scores", required=True)
    a = ap.parse_args()
    labels = load_labels(a.labels)
    bench = {r["uid"]: r for r in read_jsonl(a.bench)}
    folds = json.load(open(a.folds))
    t = compute(labels, bench, folds)
    thr = os.path.join(os.path.dirname(os.path.abspath(a.scores)), "thresholds.json")
    cols = ["s_perplexity", "s_lnentropy", "s_firsttok_entropy", "s_bm25_margin", "s_cond_prior"]
    for c in cols:
        save_thresholds(thr, c, {k: f1_threshold(t.loc[t.fold != k, c], t.loc[t.fold != k, "failure"]) for k in range(N_OUTER)})
    merge_scores(a.scores, t)      # also writes the base columns uid/qid/condition/failure/answered/fold
    print(f"[baselines] wrote {cols} for {len(t)} rows -> {a.scores}")


if __name__ == "__main__":
    main()
