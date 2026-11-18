"""P3: metrics on pooled out-of-fold scores. HIGHER score = MORE LIKELY failure.

AUROC / AUPRC / PCC with 95% bootstrap CIs resampling QUESTIONS (qid) with replacement (1000x);
AURC (area under risk-coverage, lower = better) for selective answering.
Rigor rule (plan §7b): AUROC/AUPRC are 'n/a' when the subset has < MIN_CLASS failures or successes.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

MIN_CLASS = 10
N_BOOT = 1000
COND_ORDER = ["C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]

DETECTORS = [
    "s_redeep_c1", "s_redeep", "s_redeep_c1_aucrank", "s_redeep_aucrank", "s_redeep_o1", "s_ecs", "s_pks",
    "s_lr_ecspks", "s_lr_ecspks_bm25", "s_mlp_ecspks",
    "s_probe_L1", "s_probe_L2", "s_probe_L3", "s_probe_best",
    "s_perplexity", "s_lnentropy", "s_firsttok_entropy", "s_bm25_margin",
    "s_cond_prior",
]
NICE = {"s_redeep_c1": "ReDeEP original (C1-fit)", "s_redeep": "ReDeEP noise-aware (pooled)",
        "s_redeep_c1_aucrank": "ReDeEP original, AUROC ranking (sensitivity)",
        "s_redeep_aucrank": "ReDeEP noise-aware, AUROC ranking (sensitivity)",
        "s_redeep_o1": "ReDeEP published hparams (O1)", "s_ecs": "ECS only", "s_pks": "PKS only",
        "s_lr_ecspks": "LR(ECS+PKS)", "s_lr_ecspks_bm25": "LR(ECS+PKS+BM25)", "s_mlp_ecspks": "MLP(ECS+PKS)",
        "s_probe_L1": "Probe L1", "s_probe_L2": "Probe L2", "s_probe_L3": "Probe L3", "s_probe_best": "Probe (best)",
        "s_perplexity": "Perplexity", "s_lnentropy": "LN-entropy", "s_firsttok_entropy": "First-token entropy",
        "s_bm25_margin": "BM25 margin", "s_cond_prior": "Condition prior (not deployable)"}


def _auc(y, s):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _ap(y, s):
    return average_precision_score(y, s) if 0 < y.sum() < len(y) else np.nan


def _pcc(y, s):
    if np.std(s) < 1e-12 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(y, s)[0, 1])


def risk_coverage(y, s):
    """Answer the lowest-score rows first. Returns coverage, risk arrays (risk = failure rate among kept)."""
    order = np.argsort(s, kind="stable")
    yk = np.asarray(y)[order]
    n = len(yk)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(yk) / np.arange(1, n + 1)
    return cov, risk


def aurc(y, s):
    return float(np.mean(risk_coverage(y, s)[1])) if len(y) else np.nan


def operating_point(y, s, tau):
    """Abstain when s >= tau. Returns coverage and risk among kept rows."""
    keep = s < tau
    return float(keep.mean()), (float(np.asarray(y)[keep].mean()) if keep.any() else np.nan)


def boot_indices(qids, n_boot=N_BOOT, seed=0):
    """List of row-index arrays, each a resample of questions (all rows of a sampled qid are included)."""
    qids = np.asarray(qids)
    uq, inv = np.unique(qids, return_inverse=True)
    rows_of = [np.where(inv == i)[0] for i in range(len(uq))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows_of[i] for i in rng.integers(0, len(uq), len(uq))]) for _ in range(n_boot)]


def ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def metrics_for(y, s, qids, n_boot=N_BOOT, boots=None):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n_fail, n_ok = int(y.sum()), int(len(y) - y.sum())
    out = {"n": len(y), "n_fail": n_fail, "n_ok": n_ok, "aurc": aurc(y, s), "pcc": _pcc(y, s)}
    if n_fail < MIN_CLASS or n_ok < MIN_CLASS:
        out.update(auroc=np.nan, auroc_lo=np.nan, auroc_hi=np.nan, auprc=np.nan, auprc_lo=np.nan, auprc_hi=np.nan,
                   note=f"n/a: {n_fail} failures / {n_ok} successes (< {MIN_CLASS})")
        return out
    boots = boots if boots is not None else boot_indices(qids, n_boot)
    out["auroc"] = _auc(y, s); out["auprc"] = _ap(y, s)
    out["auroc_lo"], out["auroc_hi"] = ci([_auc(y[b], s[b]) for b in boots])
    out["auprc_lo"], out["auprc_hi"] = ci([_ap(y[b], s[b]) for b in boots])
    out["note"] = ""
    return out


def paired_diff(y, s_a, s_b, qids, n_boot=N_BOOT):
    """AUROC(b) - AUROC(a) with a paired question-level bootstrap CI."""
    y = np.asarray(y).astype(int)
    if min(y.sum(), len(y) - y.sum()) < MIN_CLASS:
        return np.nan, np.nan, np.nan
    d = _auc(y, s_b) - _auc(y, s_a)
    lo, hi = ci([_auc(y[b], s_b[b]) - _auc(y[b], s_a[b]) for b in boot_indices(qids, n_boot, seed=1)])
    return float(d), lo, hi


def compute_all_metrics(scores, detectors=None, n_boot=N_BOOT):
    """Per detector x {all, C1..C5}. Returns a long DataFrame."""
    detectors = [d for d in (detectors or DETECTORS) if d in scores.columns]
    res = []
    for cond in ["all"] + [c for c in COND_ORDER if c in set(scores.condition)]:
        sub = scores if cond == "all" else scores[scores.condition == cond]
        boots = boot_indices(sub.qid.values, n_boot)
        for d in detectors:
            m = metrics_for(sub.failure.values, sub[d].values, sub.qid.values, boots=boots)
            res.append({"detector": d, "name": NICE.get(d, d), "condition": cond, **m})
    return pd.DataFrame(res)
