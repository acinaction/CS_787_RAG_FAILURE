import argparse, json, os
import numpy as np
import pandas as pd
from src.detect.common import (auc_rows, f1_threshold, inner_splits, load_labels, merge_scores,
                               save_thresholds, N_OUTER)

BETAS = np.round(np.arange(0.1, 2.01, 0.1), 2)
O1_PATH = "third_party/ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/token_hyperparameter.json"


# ---------------------------------------------------------------------------
# token tables
# ---------------------------------------------------------------------------
class TokenData:
    """Flattened token-level features for a set of rows (row order preserved)."""
    def __init__(self, df):
        ecs, pks, owner = [], [], []
        for i, (e, p) in enumerate(zip(df["ecs_tok"], df["pks_tok"])):
            e = np.asarray([np.asarray(x, float) for x in e]) if len(e) else np.zeros((0, 0))
            p = np.asarray([np.asarray(x, float) for x in p]) if len(p) else np.zeros((0, 0))
            if len(e) == 0:
                continue
            ecs.append(e); pks.append(p); owner.append(np.full(len(e), i))
        self.n_rows = len(df)
        self.ecs = np.concatenate(ecs); self.pks = np.concatenate(pks); self.owner = np.concatenate(owner)
        self.has = np.zeros(self.n_rows, bool); self.has[np.unique(self.owner)] = True
        cnt = np.bincount(self.owner, minlength=self.n_rows).astype(float)
        self.inv_cnt = np.where(cnt > 0, 1 / np.maximum(cnt, 1), np.nan)

    def subset_rows(self, rows):
        """token mask + row->new index mapping for a subset of row indices."""
        m = np.isin(self.owner, rows)
        remap = -np.ones(self.n_rows, int); remap[rows] = np.arange(len(rows))
        return m, remap

    def row_mean(self, X, mask, remap, n):
        """mean over tokens per row for token matrix X[mask] -> [n, X.shape[1]] (NaN for rows w/o tokens)."""
        own = remap[self.owner[mask]]
        out = np.zeros((n, X.shape[1])); np.add.at(out, own, X)
        cnt = np.bincount(own, minlength=n).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return out / cnt[:, None]


def _pearson_cols(X, y):
    """Pearson r of every column of X with y (0 for constant columns)."""
    Xc = X - X.mean(0); yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(0) / den
    return np.nan_to_num(r, nan=0.0)


RANK_METHOD = "pearson"     # set from --rank in main(); per-run override via Fitted(rank=...)


def rank_features(ecs_tr, pks_tr, y_tok, method=None):
    """Order copy heads and layers for the top-K search on TRAINING tokens only, and give each its orientation sign.
    Returns (h_rank, l_rank, h_sign, l_sign); signs are aligned with the rank order.

    method='pearson'  PRIMARY, FINAL decision 2026-11-07 (FINAL_PLAN_V3 sec. 6):
        signed Pearson r(feature, failure) is computed for every head's ECS and every layer's PKS;
        candidates are RANKED BY |r| (descending); the ORIGINAL SIGN of r is RETAINED and used in scoring:
        head sign = sign r(ECS, failure), layer sign = sign r(PKS, failure).
        A feature with positive sign is 'failure-increasing', with negative sign 'failure-decreasing'.
    method='auroc'    SENSITIVITY ONLY (authors' token_level_reg.py): heads by AUROC(ECS, 1-failure), layers by
        AUROC(PKS, failure); signs fixed to the paper's assumption (ECS failure-decreasing, PKS failure-increasing).
    Zero / undefined r falls back to the paper's direction (head -1, layer +1)."""
    method = method or RANK_METHOD
    y = np.asarray(y_tok, float)
    if method == "auroc":
        a_e = np.nan_to_num(auc_rows(ecs_tr.T, 1 - y), nan=0); a_p = np.nan_to_num(auc_rows(pks_tr.T, y), nan=0)
        h_rank, l_rank = np.argsort(-a_e, kind="stable"), np.argsort(-a_p, kind="stable")
        return h_rank, l_rank, -np.ones(len(h_rank)), np.ones(len(l_rank))
    if method == "pearson":
        r_e = _pearson_cols(ecs_tr, y); r_p = _pearson_cols(pks_tr, y)
        h_rank, l_rank = np.argsort(-np.abs(r_e), kind="stable"), np.argsort(-np.abs(r_p), kind="stable")
        h_sign = np.where(r_e[h_rank] == 0, -1.0, np.sign(r_e[h_rank]))
        l_sign = np.where(r_p[l_rank] == 0, 1.0, np.sign(r_p[l_rank]))
        return h_rank, l_rank, h_sign, l_sign
    raise ValueError(method)


class Fitted:
    """Ranking, orientation signs and min-max constants learned on training tokens.
    Oriented features: ECS_h * h_sign, PKS_l * l_sign (higher oriented value = more failure-like).
    With the paper's assumed signs (ECS -1, PKS +1) the score  Pn + beta * En  is the paper's  PKS_n - beta * ECS_n
    up to an additive constant, i.e. it gives identical rankings; other signs are the signed generalisation."""
    def __init__(self, ecs_tr, pks_tr, y_tok, rank=None):
        self.h_rank, self.l_rank, self.h_sign, self.l_sign = rank_features(ecs_tr, pks_tr, y_tok, rank)
        E, P = self._cum(ecs_tr, pks_tr)
        self.e_min, self.e_max = E.min(0), E.max(0)
        self.p_min, self.p_max = P.min(0), P.max(0)

    def _cum(self, ecs, pks):
        return (np.cumsum(ecs[:, self.h_rank] * self.h_sign[None, :], 1),
                np.cumsum(pks[:, self.l_rank] * self.l_sign[None, :], 1))

    def norm_cum(self, ecs, pks):
        E, P = self._cum(ecs, pks)
        En = (E - self.e_min) / np.where(self.e_max - self.e_min > 1e-12, self.e_max - self.e_min, 1)
        Pn = (P - self.p_min) / np.where(self.p_max - self.p_min > 1e-12, self.p_max - self.p_min, 1)
        return En, Pn          # column k = top-(k+1) oriented features


def grid_scores(Re, Rp, mode):
    """All configs -> row scores. Re [n,H], Rp [n,L] are row means of normalised cumsums.
    mode 'full': [H*L*B, n] with config list; 'ecs': [H, n]; 'pks': [L, n]."""
    H, L = Re.shape[1], Rp.shape[1]
    if mode == "ecs":
        return Re.T, [(h + 1, None, None) for h in range(H)]       # Re is already oriented: higher = failure
    if mode == "pks":
        return Rp.T, [(None, l + 1, None) for l in range(L)]
    S = Rp.T[None, :, None, :] + BETAS[None, None, :, None] * Re.T[:, None, None, :]   # [H, L, B, n]
    cfg = [(h + 1, l + 1, float(b)) for h in range(H) for l in range(L) for b in BETAS]
    return S.reshape(-1, Re.shape[0]), cfg


def score_with(cfg, Re, Rp, mode):
    h, l, b = cfg
    if mode == "ecs":
        return Re[:, h - 1]
    if mode == "pks":
        return Rp[:, l - 1]
    return Rp[:, l - 1] + b * Re[:, h - 1]


def fit_fold(td, y_rows, qids, train_rows, test_rows, mode, rank=None):
    """Inner-CV config choice on train_rows, refit on all train_rows, score test_rows. Returns (scores, cfg, tau, inner_auc)."""
    tr_has = train_rows[td.has[train_rows]]
    splits = inner_splits(qids[tr_has])
    tot, cnt = None, None
    inner_parts = []
    for itr, iva in splits:
        rtr, rva = tr_has[itr], tr_has[iva]
        m_tr, _ = td.subset_rows(rtr)
        y_tok = y_rows[td.owner[m_tr]]
        if y_tok.min() == y_tok.max():
            continue
        fit = Fitted(td.ecs[m_tr], td.pks[m_tr], y_tok, rank)
        m_va, remap = td.subset_rows(rva)
        En, Pn = fit.norm_cum(td.ecs[m_va], td.pks[m_va])
        Re, Rp = td.row_mean(En, m_va, remap, len(rva)), td.row_mean(Pn, m_va, remap, len(rva))
        S, cfgs = grid_scores(Re, Rp, mode)
        a = auc_rows(S, y_rows[rva])
        if np.all(np.isnan(a)):
            continue
        tot = a if tot is None else tot + np.nan_to_num(a)
        cnt = (~np.isnan(a)).astype(int) if cnt is None else cnt + (~np.isnan(a))
        inner_parts.append((rva, S))
    if tot is None:
        raise RuntimeError("no usable inner fold (single-class training data)")
    mean_auc = tot / np.maximum(cnt, 1)
    best = int(np.nanargmax(mean_auc)); cfg = cfgs[best]
    inner_s = np.concatenate([S[best] for _, S in inner_parts])
    inner_y = np.concatenate([y_rows[r] for r, _ in inner_parts])
    tau = f1_threshold(inner_s, inner_y)

    m_tr, _ = td.subset_rows(tr_has)
    fit = Fitted(td.ecs[m_tr], td.pks[m_tr], y_rows[td.owner[m_tr]], rank)
    m_te, remap = td.subset_rows(test_rows)
    En, Pn = fit.norm_cum(td.ecs[m_te], td.pks[m_te])
    Re, Rp = td.row_mean(En, m_te, remap, len(test_rows)), td.row_mean(Pn, m_te, remap, len(test_rows))
    s_te = score_with(cfg, Re, Rp, mode)
    # rows without answer tokens (empty answers): fill with the training-score median
    m_tr2, remap2 = td.subset_rows(tr_has)
    En2, Pn2 = fit.norm_cum(td.ecs[m_tr2], td.pks[m_tr2])
    s_tr = score_with(cfg, td.row_mean(En2, m_tr2, remap2, len(tr_has)), td.row_mean(Pn2, m_tr2, remap2, len(tr_has)), mode)
    s_te = np.where(np.isfinite(s_te), s_te, np.nanmedian(s_tr))
    return s_te, cfg, tau, float(mean_auc[best]), fit


def run(df, mode, train_conditions=None, name="s_redeep", log=None, rank=None):
    td = TokenData(df)
    y = df["failure"].values.astype(int); q = df["qid"].values; fold = df["fold"].values
    out = np.full(len(df), np.nan); taus = {}
    for k in range(N_OUTER):
        te = np.where(fold == k)[0]
        trm = fold != k
        if train_conditions:
            trm &= df["condition"].isin(train_conditions).values
        tr = np.where(trm)[0]
        s, cfg, tau, ia, fit = fit_fold(td, y, q, tr, te, mode, rank)
        out[te] = s; taus[k] = tau
        heads = fit.h_rank[: cfg[0]].tolist() if cfg[0] else []
        layers = fit.l_rank[: cfg[1]].tolist() if cfg[1] else []
        hs = fit.h_sign[: cfg[0]].astype(int).tolist() if cfg[0] else []
        ls = fit.l_sign[: cfg[1]].astype(int).tolist() if cfg[1] else []
        msg = f"[{name}] fold={k} n_train={len(tr)} K_h={cfg[0]} K_l={cfg[1]} beta={cfg[2]} inner_auc={ia:.3f} tau={tau:.3f}"
        print(msg)
        if log is not None:
            log.append({"detector": name, "rank": rank or RANK_METHOD, "fold": k, "n_train": int(len(tr)), "K_heads": cfg[0], "K_layers": cfg[1],
                        "beta": cfg[2], "inner_auc": ia, "tau": tau, "head_idx": heads, "layer_idx": layers,
                        "head_sign": hs, "layer_sign": ls,
                        "n_head_sign_unexpected": int(sum(x > 0 for x in hs)),     # ECS rising with failure (paper: falls)
                        "n_layer_sign_unexpected": int(sum(x < 0 for x in ls))})  # PKS falling with failure (paper: rises)
    return out, taus


def run_o1(df, meta):
    """Published hyperparameters (Appendix J / token_hyperparameter.json), no refit."""
    if not os.path.exists(O1_PATH):
        return None
    hp = json.load(open(O1_PATH))
    head_list = [tuple(h) for h in meta.get("head_list", [])]
    try:
        hcols = [head_list.index(tuple(h)) for h in hp["select_heads"]]
    except ValueError:
        return None
    lcols = list(hp["select_layers"])
    (emax, emin), (pmax, pmin) = hp["head_max_min"], hp["layers_max_min"]
    out = []
    for e, p in zip(df["ecs_tok"], df["pks_tok"]):
        if len(e) == 0:
            out.append(np.nan); continue
        e = np.asarray([np.asarray(x, float) for x in e]); p = np.asarray([np.asarray(x, float) for x in p])
        en = (e[:, hcols].sum(1) - emin) / (emax - emin); pn = (p[:, lcols].sum(1) - pmin) / (pmax - pmin)
        out.append(float(np.mean(pn - hp["weight"] * en)))
    out = np.array(out)
    return np.where(np.isfinite(out), out, np.nanmedian(out)), hp


def main():
    ap = argparse.ArgumentParser(description="Fit ReDeEP detectors")
    ap.add_argument("--feats", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rank", default="pearson", choices=["pearson", "auroc"],
                    help="PRIMARY head/layer ranking: pearson = rank by |r|, sign retained (FROZEN decision 2026-11-07). "
                         "auroc = sensitivity only")
    ap.add_argument("--no_sensitivity_auroc", dest="sensitivity_auroc", action="store_false",
                    help="skip the AUROC-ranking sensitivity columns")
    ap.add_argument("--cross", nargs="*", default=["C2_random"], help="E7: training conditions for cross-noise fits")
    a = ap.parse_args()

    global RANK_METHOD
    RANK_METHOD = a.rank
    print(f"[fit_redeep] primary head/layer ranking: {a.rank}")
    feats = pd.read_parquet(a.feats)
    labels = load_labels(a.labels)
    folds = json.load(open(a.folds))
    df = labels[["uid", "qid", "condition", "failure"]].merge(feats, on="uid", how="inner")
    df["fold"] = df["qid"].map(folds).astype(int)
    assert len(df) == len(labels), f"features missing for {len(labels) - len(df)} rows"
    meta = json.load(open(a.meta)) if a.meta and os.path.exists(a.meta) else {}
    out = pd.DataFrame({"uid": df["uid"]})
    thr_path = os.path.join(os.path.dirname(os.path.abspath(a.scores)), "thresholds.json")
    log = []
    for name, mode, conds in [("s_redeep_c1", "full", ["C1_gold"]), ("s_redeep", "full", None),
                              ("s_ecs", "ecs", None), ("s_pks", "pks", None)]:
        s, taus = run(df, mode, conds, name, log)
        out[name] = s
        save_thresholds(thr_path, name, taus)
    # Sensitivity (not the primary method): authors' AUROC-based ranking, same everything else.
    if a.sensitivity_auroc:
        for name, conds in [("s_redeep_c1_aucrank", ["C1_gold"]), ("s_redeep_aucrank", None)]:
            s, taus = run(df, "full", conds, name, log, rank="auroc")
            out[name] = s
            save_thresholds(thr_path, name, taus)
    # E7 cross-noise: fit on ONE noise condition (training folds only), score every held-out row.
    # Evaluate these columns only on the other conditions (see make_tables E7).
    for tr_c in a.cross:
        s, taus = run(df, "full", [tr_c], f"x_redeep_tr{tr_c.split('_')[0]}", log)
        out[f"x_redeep_tr{tr_c.split('_')[0]}"] = s
    o1 = run_o1(df, meta)
    if o1 is not None:
        out["s_redeep_o1"] = o1[0]
        save_thresholds(thr_path, "s_redeep_o1", {k: 0.6 for k in range(N_OUTER)})   # published token threshold
        print("[s_redeep_o1] published hyperparameters applied (no refit)")
    else:
        print("[s_redeep_o1] skipped: published heads not in this run's head list (or repo missing)")
    json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(a.scores)), "redeep_fit_log.json"), "w"), indent=1)
    merge_scores(a.scores, out)
    print(f"[fit_redeep] wrote {list(out.columns[1:])} -> {a.scores}")


if __name__ == "__main__":
    main()
