"""P3: all result tables (E1-E7, E9 export) from the out-of-fold scores.

Usage:
  python -m src.analysis.make_tables --scores .../scores.parquet --labeled_std .../labeled_std.jsonl \
      --labeled_abstain .../labeled_abstain.jsonl --feats .../redeep_feats.parquet --out .../tables
Writes CSVs (and a combined tables.md) into --out.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from src.analysis.metrics import (COND_ORDER, DETECTORS, NICE, aurc, compute_all_metrics, metrics_for,
                                  operating_point, paired_diff)
from src.detect.common import load_labels


def fmt(m, lo, hi):
    return "n/a" if not np.isfinite(m) else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def e1(lab_std, lab_abs):
    rows = []
    for tmpl, lab in [("std", lab_std), ("abstain", lab_abs)]:
        if lab is None:
            continue
        for c in COND_ORDER:
            g = lab[lab.condition == c]
            if len(g) == 0:
                continue
            rows.append({"template": tmpl, "condition": c, "n": len(g), "accuracy": round(g.correct.mean(), 3),
                         "refusal_rate": round(g.refusal.mean(), 3), "failure_rate": round(g.failure.mean(), 3),
                         "followed_counterfactual": round(g.followed_counterfactual.mean(), 3) if c == "C5_counterfactual" else np.nan})
    return pd.DataFrame(rows)


def e2(scores):
    m = compute_all_metrics(scores)
    m["auroc_ci"] = [fmt(a, l, h) for a, l, h in zip(m.auroc, m.auroc_lo, m.auroc_hi)]
    wide = m.pivot(index="detector", columns="condition", values="auroc_ci")
    wide = wide[[c for c in COND_ORDER if c in wide.columns] + ["all"]]
    wide.insert(0, "name", [NICE.get(d, d) for d in wide.index])
    order = [d for d in DETECTORS if d in wide.index]
    return m, wide.loc[order].reset_index()


def selected_sets(fit_log, n_heads, n_layers, det="s_redeep_c1"):
    """Heads/layers picked by the ORIGINAL fit in >= 3 of 5 folds (fallback: fold 0's picks)."""
    recs = [r for r in fit_log if r["detector"] == det]
    if not recs:
        return list(range(n_heads)), list(range(n_layers))
    from collections import Counter
    ch = Counter(h for r in recs for h in r["head_idx"]); cl = Counter(l for r in recs for l in r["layer_idx"])
    hs = sorted(h for h, c in ch.items() if c >= 3) or recs[0]["head_idx"]
    ls = sorted(l for l, c in cl.items() if c >= 3) or recs[0]["layer_idx"]
    return hs, ls


def e3(scores, feats, fit_log):
    """Sign analysis, no fitting: per condition, corr(mean ECS over selected heads, failure) and same for PKS."""
    df = scores[["uid", "condition", "failure"]].merge(feats[["uid", "ecs_mean", "pks_mean"]], on="uid")
    E = np.stack([np.asarray(x, float) for x in df.ecs_mean]); P = np.stack([np.asarray(x, float) for x in df.pks_mean])
    hs, ls = selected_sets(fit_log, E.shape[1], P.shape[1])
    late = list(range(P.shape[1] // 2, P.shape[1]))
    df["ecs_sel"] = np.nanmean(E[:, hs], 1); df["pks_sel"] = np.nanmean(P[:, ls], 1); df["pks_late"] = np.nanmean(P[:, late], 1)
    rows = []
    for c in COND_ORDER + ["all"]:
        g = df if c == "all" else df[df.condition == c]
        g = g.dropna(subset=["ecs_sel", "pks_sel"])
        y = g.failure.values
        def r(col):
            return float(np.corrcoef(y, g[col])[0, 1]) if 0 < y.sum() < len(y) else np.nan
        rows.append({"condition": c, "n": len(g), "n_fail": int(y.sum()),
                     "corr_ECS_selected_heads": r("ecs_sel"), "corr_PKS_selected_layers": r("pks_sel"),
                     "corr_PKS_late_layers": r("pks_late"),
                     "mean_ECS_fail": g.loc[g.failure == 1, "ecs_sel"].mean(), "mean_ECS_ok": g.loc[g.failure == 0, "ecs_sel"].mean(),
                     "mean_PKS_fail": g.loc[g.failure == 1, "pks_sel"].mean(), "mean_PKS_ok": g.loc[g.failure == 0, "pks_sel"].mean()})
    out = pd.DataFrame(rows)
    out.attrs["heads"], out.attrs["layers"] = hs, ls
    return out


def e4(m):
    keep = ["s_ecs", "s_pks", "s_redeep", "s_redeep_c1", "s_probe_L1", "s_probe_L2", "s_probe_L3", "s_probe_best"]
    t = m[m.detector.isin(keep)][["name", "condition", "auroc", "auroc_lo", "auroc_hi", "pcc", "note"]]
    return t


def e4b_ranking(fit_log, scores):
    """Sensitivity: Pearson (primary) vs AUROC (authors) head/layer ranking — selections and performance."""
    sel = []
    by = {(r["detector"], r["fold"]): r for r in fit_log}
    for prim, sens in [("s_redeep_c1", "s_redeep_c1_aucrank"), ("s_redeep", "s_redeep_aucrank")]:
        for k in range(5):
            a, b = by.get((prim, k)), by.get((sens, k))
            if not a or not b:
                continue
            ha, hb, la, lb = set(a["head_idx"]), set(b["head_idx"]), set(a["layer_idx"]), set(b["layer_idx"])
            sel.append({"fit": prim, "fold": k,
                        "pearson_K_h/K_l/beta": f"{a['K_heads']}/{a['K_layers']}/{a['beta']}",
                        "auroc_K_h/K_l/beta": f"{b['K_heads']}/{b['K_layers']}/{b['beta']}",
                        "pearson_heads_ECS_rises_with_failure": a.get("n_head_sign_unexpected"),
                        "pearson_layers_PKS_falls_with_failure": a.get("n_layer_sign_unexpected"),
                        "head_jaccard": round(len(ha & hb) / max(len(ha | hb), 1), 2),
                        "layer_jaccard": round(len(la & lb) / max(len(la | lb), 1), 2),
                        "pearson_heads": a["head_idx"], "auroc_heads": b["head_idx"],
                        "pearson_layers": a["layer_idx"], "auroc_layers": b["layer_idx"]})
    perf = []
    for prim, sens in [("s_redeep_c1", "s_redeep_c1_aucrank"), ("s_redeep", "s_redeep_aucrank")]:
        if prim not in scores or sens not in scores:
            continue
        for cond in COND_ORDER + ["all"]:
            g = scores if cond == "all" else scores[scores.condition == cond]
            mp = metrics_for(g.failure.values, g[prim].values, g.qid.values)
            ms = metrics_for(g.failure.values, g[sens].values, g.qid.values)
            d, lo, hi = paired_diff(g.failure.values, g[prim].values, g[sens].values, g.qid.values)
            perf.append({"fit": prim, "condition": cond, "pearson_rank_auroc": fmt(mp["auroc"], mp["auroc_lo"], mp["auroc_hi"]),
                         "auroc_rank_auroc": fmt(ms["auroc"], ms["auroc_lo"], ms["auroc_hi"]),
                         "delta_auroc(aucrank-pearson)": "n/a" if not np.isfinite(d) else f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"})
    return pd.DataFrame(sel), pd.DataFrame(perf)


def e5(scores, lab_abs, thresholds):
    rows = []
    y = scores.failure.values
    for d in [x for x in DETECTORS if x in scores.columns]:
        s = scores[d].values
        cov, risk = [], []
        for k in range(5):
            m = scores.fold.values == k
            tau = (thresholds.get(d) or {}).get(str(k))
            if tau is None:
                continue
            c_, r_ = operating_point(y[m], s[m], tau); cov.append(c_ * m.sum()); risk.append(r_ * c_ * m.sum() if np.isfinite(r_) else 0)
        kept = sum(cov)
        rows.append({"detector": d, "name": NICE.get(d, d), "AURC": round(aurc(y, s), 4),
                     "coverage_at_tau": round(kept / len(y), 3) if cov else np.nan,
                     "risk_at_tau": round(sum(risk) / kept, 3) if cov and kept else np.nan})
    rows.append({"detector": "oracle", "name": "Oracle (perfect ranking)", "AURC": round(aurc(y, y + 1e-9 * np.arange(len(y))), 4)})
    rows.append({"detector": "none", "name": "No abstention (answer all)", "AURC": round(y.mean(), 4), "coverage_at_tau": 1.0, "risk_at_tau": round(y.mean(), 3)})
    if lab_abs is not None:
        ans = lab_abs.answered.values.astype(bool)
        rows.append({"detector": "prompt_abstain", "name": "Prompt-based abstention (single point)", "AURC": np.nan,
                     "coverage_at_tau": round(ans.mean(), 3), "risk_at_tau": round(lab_abs.failure.values[ans].mean(), 3)})
    return pd.DataFrame(rows)


def outcomes(y, flag):
    """kept-correct / caught-failure / missed-failure / false-abstain."""
    return np.where(flag, np.where(y == 1, "caught-failure", "false-abstain"), np.where(y == 1, "missed-failure", "kept-correct"))


def flags_for(scores, d, thresholds):
    tau = np.array([(thresholds.get(d) or {}).get(str(k)) or np.inf for k in scores.fold.values], float)
    return scores[d].values >= tau


def e6(scores, thresholds, a="s_redeep_c1", b="s_redeep", c="s_probe_best"):
    rows = []
    for cond in COND_ORDER + ["all"]:
        g = scores if cond == "all" else scores[scores.condition == cond]
        y, q = g.failure.values, g.qid.values
        row = {"condition": cond, "n": len(g), "failure_rate": round(y.mean(), 3)}
        for d in [a, b, c]:
            if d not in g:
                continue
            m = metrics_for(y, g[d].values, q)
            row[f"{d}_auroc"] = fmt(m["auroc"], m["auroc_lo"], m["auroc_hi"])
            o = outcomes(y, flags_for(g, d, thresholds))
            for k in ["caught-failure", "missed-failure", "false-abstain", "kept-correct"]:
                row[f"{d}_{k}"] = int((o == k).sum())
        if a in g and b in g:
            dlt, lo, hi = paired_diff(y, g[a].values, g[b].values, q)
            row["delta_auroc_ours_minus_orig"] = "n/a" if not np.isfinite(dlt) else f"{dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def e7(scores):
    rows = []
    for col in [c for c in scores.columns if c.startswith("x_")]:
        tr = col.split("_tr")[1]
        for cond in COND_ORDER:
            if cond.startswith(tr):
                continue
            g = scores[scores.condition == cond]
            m = metrics_for(g.failure.values, g[col].values, g.qid.values)
            ref = "s_redeep" if "redeep" in col else "s_probe_best"
            mr = metrics_for(g.failure.values, g[ref].values, g.qid.values)
            rows.append({"detector": col, "train_on": tr, "test_on": cond, "n_fail": m["n_fail"],
                         "auroc": fmt(m["auroc"], m["auroc_lo"], m["auroc_hi"]),
                         f"pooled_ref({ref})": fmt(mr["auroc"], mr["auroc_lo"], mr["auroc_hi"])})
    return pd.DataFrame(rows)


def e9(lab_std, scores, n=15, n_c4=5, seed=0):
    """Qualitative export for hand annotation: 15 random cases + 5 C4 rows where the model answered anyway."""
    df = lab_std.merge(scores[[c for c in ["uid", "s_redeep_c1", "s_redeep", "s_probe_best"] if c in scores]], on="uid")
    rnd = df.sample(n=min(n, len(df)), random_state=seed)
    c4 = df[(df.condition == "C4_missing") & (df.answered == 1)]
    c4 = c4.sample(n=min(n_c4, len(c4)), random_state=seed)
    out = pd.concat([rnd.assign(set="random15"), c4.assign(set="c4_answered")])
    cols = ["set", "uid", "condition", "answer", "correct", "failure", "s_redeep_c1", "s_redeep", "s_probe_best"]
    out = out[[c for c in cols if c in out]].copy()
    out["notes"] = ""
    return out


def to_md(df, title):
    return f"\n### {title}\n\n" + df.to_markdown(index=False, floatfmt=".3f") + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--labeled_std", required=True)
    ap.add_argument("--labeled_abstain", default=None); ap.add_argument("--feats", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rdir = os.path.dirname(os.path.abspath(a.scores))
    scores = pd.read_parquet(a.scores)
    lab_std = load_labels(a.labeled_std)
    lab_abs = load_labels(a.labeled_abstain) if a.labeled_abstain and os.path.exists(a.labeled_abstain) else None
    thresholds = json.load(open(os.path.join(rdir, "thresholds.json")))
    fit_log = json.load(open(os.path.join(rdir, "redeep_fit_log.json"))) if os.path.exists(os.path.join(rdir, "redeep_fit_log.json")) else []
    md = ["# Result tables (auto-generated by src.analysis.make_tables)\n",
          "All numbers are pooled out-of-fold predictions (5-fold CV grouped by qid); 95% CIs bootstrap questions (1000x). "
          "AUROC is n/a when a subset has <10 failures or <10 successes.\n"]

    t1 = e1(lab_std, lab_abs); t1.to_csv(f"{a.out}/e1_noise_impact.csv", index=False); md.append(to_md(t1, "E1 — noise impact"))
    m, t2 = e2(scores); m.to_csv(f"{a.out}/e2_metrics_long.csv", index=False); t2.to_csv(f"{a.out}/e2_auroc_by_condition.csv", index=False)
    md.append(to_md(t2, "E2 — AUROC [95% CI] per condition (main result = within-condition columns)"))
    if a.feats and os.path.exists(a.feats):
        t3 = e3(scores, pd.read_parquet(a.feats), fit_log); t3.to_csv(f"{a.out}/e3_sign_analysis.csv", index=False)
        md.append(to_md(t3, f"E3 — sign analysis (heads idx {t3.attrs['heads']}, layers {t3.attrs['layers']}; no fitting)"))
    t4 = e4(m); t4.to_csv(f"{a.out}/e4_ablation.csv", index=False)
    t4s, t4p = e4b_ranking(fit_log, scores)
    if len(t4s):
        t4s.to_csv(f"{a.out}/e4b_ranking_selections.csv", index=False); t4p.to_csv(f"{a.out}/e4b_ranking_performance.csv", index=False)
        md.append(to_md(t4s.drop(columns=["pearson_heads", "auroc_heads", "pearson_layers", "auroc_layers"]),
                        "E4b — head/layer ranking sensitivity: selections (Pearson = primary, AUROC = authors)"))
        md.append(to_md(t4p, "E4b — ranking sensitivity: performance"))
    t5 = e5(scores, lab_abs, thresholds); t5.to_csv(f"{a.out}/e5_selective.csv", index=False); md.append(to_md(t5, "E5 — selective answering (AURC lower = better)"))
    t6 = e6(scores, thresholds); t6.to_csv(f"{a.out}/e6_original_vs_ours.csv", index=False); md.append(to_md(t6, "E6 — original (C1-fit) vs ours (pooled) vs probe"))
    t7 = e7(scores)
    if len(t7):
        t7.to_csv(f"{a.out}/e7_cross_noise.csv", index=False); md.append(to_md(t7, "E7 — cross-noise transfer"))
    e9(lab_std, scores).to_csv(f"{a.out}/e9_qualitative_to_annotate.csv", index=False)
    open(f"{a.out}/tables.md", "w", encoding="utf-8").write("\n".join(md))
    print(f"[make_tables] wrote tables to {a.out}")
    print(t2[["name"] + [c for c in t2.columns if c.startswith("C") or c == "all"]].to_string(index=False))


if __name__ == "__main__":
    main()
