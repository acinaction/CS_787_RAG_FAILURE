"""Shared helpers for all detectors: loading, fast AUROC, F1 threshold, grouped folds, score-table merge.
Sign convention everywhere: HIGHER score = MORE LIKELY failure."""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

N_OUTER = 5
N_INNER = 3


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_labels(path):
    df = pd.DataFrame(read_jsonl(path))
    if "qid" not in df:
        df["qid"] = df["uid"].str.rsplit("__", n=1).str[0]
    if "condition" not in df:
        df["condition"] = df["uid"].str.rsplit("__", n=1).str[1]
    df["failure"] = df["failure"].astype(int)
    df["answered"] = df["answered"].astype(int)
    return df


def base_table(labels, folds):
    t = labels[["uid", "qid", "condition", "failure", "answered"]].copy()
    t["fold"] = t["qid"].map(folds).astype(int)
    return t


def auc_rows(S, y):
    """Vectorised AUROC of every row of S [m, n] against binary y [n]. Ties get average rank."""
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[0], np.nan)
    R = rankdata(S, axis=1)
    return (R[:, y].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc(s, y):
    return float(auc_rows(np.asarray(s, float)[None, :], y)[0])


def f1_threshold(scores, labels):
    """F1-optimal threshold: flag (abstain) when score >= tau. Searched over all unique scores."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    if len(s) == 0 or y.sum() == 0:
        return float("inf")
    order = np.argsort(-s); s_sorted, y_sorted = s[order], y[order]
    tp = np.cumsum(y_sorted); fp = np.cumsum(1 - y_sorted)
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]       # only cut between distinct values
    prec = tp / (tp + fp); rec = tp / y.sum()
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    f1[~last] = -1
    return float(s_sorted[int(np.argmax(f1))])


def inner_splits(qids, n_splits=N_INNER, seed=0):
    """Grouped K-fold over qids (deterministic, shuffled)."""
    uq = np.array(sorted(set(qids)))
    rng = np.random.default_rng(seed); rng.shuffle(uq)
    fold_of = {q: i % n_splits for i, q in enumerate(uq)}
    f = np.array([fold_of[q] for q in qids])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def merge_scores(path, new_df):
    """Add/replace columns of new_df (keyed by uid) in scores.parquet."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        base = pd.read_parquet(path)
        base = base.drop(columns=[c for c in new_df.columns if c != "uid" and c in base.columns])
        new_df = base.merge(new_df, on="uid", how="left")
    new_df.to_parquet(path, index=False)
    return new_df


def save_thresholds(path, name, per_fold):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[name] = {str(k): (None if not np.isfinite(v) else float(v)) for k, v in per_fold.items()}
    json.dump(d, open(path, "w"), indent=1)
