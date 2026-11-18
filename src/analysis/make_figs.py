"""P3: figures.
  e5_risk_coverage.png   E5: risk-coverage curves (pooled) + prompt-abstain point; per-condition panels
  e2_auroc_by_condition.png  E2/E6: within-condition AUROC with CIs for the key detectors
  e3_sign_analysis.png   E3: corr(ECS / PKS, failure) per condition
  comparison_20.png      E8: two heatmaps (Original | Ours), rows = 20 prompts, cols = C1..C5, colour = outcome
Usage:
  python -m src.analysis.make_figs --scores ... --tables .../tables --labeled_abstain ... \
      --comparison .../comparison/comparison_20.csv --out .../figs
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from src.analysis.metrics import COND_ORDER, NICE, aurc, risk_coverage
from src.detect.common import load_labels

KEY = ["s_redeep_c1", "s_redeep", "s_probe_best", "s_lr_ecspks", "s_perplexity", "s_firsttok_entropy", "s_bm25_margin"]
COLORS = {"s_redeep_c1": "#7f7f7f", "s_redeep": "#1f77b4", "s_probe_best": "#2ca02c", "s_lr_ecspks": "#9467bd",
          "s_perplexity": "#ff7f0e", "s_firsttok_entropy": "#8c564b", "s_bm25_margin": "#e377c2"}
SHORT = {"C1_gold": "C1 gold", "C2_random": "C2 random", "C3_hardneg": "C3 hard-neg",
         "C4_missing": "C4 missing", "C5_counterfactual": "C5 counterf."}
OUT_COL = {"kept-correct": "#4daf4a", "caught-failure": "#377eb8", "missed-failure": "#e41a1c", "false-abstain": "#ff7f00"}


def fig_rc(scores, lab_abs, out):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.2), sharey=True)
    for ax, cond in zip(axes, ["all"] + COND_ORDER):
        g = scores if cond == "all" else scores[scores.condition == cond]
        y = g.failure.values
        for d in [k for k in KEY if k in g]:
            cov, risk = risk_coverage(y, g[d].values)
            ax.plot(cov, risk, color=COLORS[d], lw=2 if d in ("s_redeep", "s_redeep_c1") else 1.2,
                    ls="--" if d == "s_redeep_c1" else "-", label=f"{NICE[d]} ({aurc(y, g[d].values):.3f})")
        ax.axhline(y.mean(), color="k", lw=.6, ls=":")
        if lab_abs is not None:
            la = lab_abs if cond == "all" else lab_abs[lab_abs.condition == cond]
            ans = la.answered.values.astype(bool)
            if ans.any():
                ax.scatter([ans.mean()], [la.failure.values[ans].mean()], marker="*", s=180, color="k", zorder=5,
                           label="prompt abstain")
        ax.set_title(SHORT.get(cond, "all conditions (pooled)")); ax.set_xlabel("coverage"); ax.grid(alpha=.3)
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("risk (failure rate among answered)")
    fig.suptitle("E5 — risk–coverage (legend: AURC, lower is better)")
    fig.tight_layout(); fig.savefig(f"{out}/e5_risk_coverage.png", dpi=140); plt.close(fig)


def fig_auroc(m, out):
    dets = [d for d in ["s_redeep_c1", "s_redeep", "s_ecs", "s_pks", "s_lr_ecspks", "s_probe_best", "s_perplexity", "s_firsttok_entropy"] if d in set(m.detector)]
    conds = COND_ORDER + ["all"]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    w = .8 / len(dets)
    for i, d in enumerate(dets):
        g = m[m.detector == d].set_index("condition").reindex(conds)
        x = np.arange(len(conds)) + i * w - .4 + w / 2
        err = np.vstack([g.auroc - g.auroc_lo, g.auroc_hi - g.auroc]).clip(min=0)
        ax.bar(x, g.auroc.fillna(0), w, yerr=np.nan_to_num(err), capsize=2, label=NICE[d],
               color=COLORS.get(d, plt.cm.tab20(i)), alpha=.9)
        for xi, v in zip(x, g.auroc):
            if not np.isfinite(v):
                ax.text(xi, .02, "n/a", rotation=90, fontsize=6, ha="center")
    ax.axhline(.5, color="k", lw=.7, ls=":")
    ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels([SHORT.get(c, "pooled") for c in conds])
    ax.set_ylim(0, 1); ax.set_ylabel("AUROC (95% CI)"); ax.legend(fontsize=7, ncol=4); ax.grid(axis="y", alpha=.3)
    ax.set_title("E2/E6 — within-condition AUROC")
    fig.tight_layout(); fig.savefig(f"{out}/e2_auroc_by_condition.png", dpi=140); plt.close(fig)


def fig_sign(t3, out):
    t = t3.set_index("condition").reindex(COND_ORDER)
    fig, ax = plt.subplots(figsize=(7, 3.8)); x = np.arange(len(t))
    ax.bar(x - .2, t.corr_ECS_selected_heads, .4, label="ECS (selected copy heads)", color="#1f77b4")
    ax.bar(x + .2, t.corr_PKS_selected_layers, .4, label="PKS (selected layers)", color="#d62728")
    ax.axhline(0, color="k", lw=.7); ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in t.index], fontsize=8)
    ax.set_ylabel("corr with failure"); ax.legend(fontsize=8); ax.set_title("E3 — does the signal keep its sign under noise?")
    fig.tight_layout(); fig.savefig(f"{out}/e3_sign_analysis.png", dpi=140); plt.close(fig)


def fig_cmp(cmp, out):
    keys = list(OUT_COL); cmap = ListedColormap([OUT_COL[k] for k in keys])
    qs = list(dict.fromkeys(cmp.qid))
    import textwrap
    def _wrap(t):
        w = textwrap.wrap(t, 40)
        return chr(10).join(w[:2]) + ('...' if len(w) > 2 else '')
    labels = [_wrap(cmp[cmp.qid == q].question.iloc[0]) for q in qs]
    fig, axes = plt.subplots(1, 2, figsize=(11, .46 * len(qs) + 1.6), sharey=True)
    for ax, side, title in zip(axes, ["orig", "ours"], ["Original ReDeEP (fit on C1 only)", "Ours: noise-aware ReDeEP (pooled fit)"]):
        piv = cmp.pivot(index="qid", columns="condition", values=f"{side}_outcome").reindex(index=qs, columns=COND_ORDER)
        mat = piv.apply(lambda c: c.map({k: i for i, k in enumerate(keys)})).values.astype(float)
        ax.imshow(mat, cmap=cmap, vmin=-.5, vmax=3.5, aspect="auto")
        ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[c] for c in COND_ORDER], fontsize=8, rotation=20)
        ax.set_yticks(range(len(qs))); ax.set_yticklabels(labels, fontsize=7.5); ax.set_title(title, fontsize=10)
        dis = cmp.pivot(index="qid", columns="condition", values="orig_flag").reindex(index=qs, columns=COND_ORDER).values != \
              cmp.pivot(index="qid", columns="condition", values="ours_flag").reindex(index=qs, columns=COND_ORDER).values
        for i, j in zip(*np.where(dis)):
            ax.text(j, i, "≠", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    fig.legend(handles=[Patch(color=OUT_COL[k], label=k) for k in keys], loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("E8 — 20 fixed random prompts × 5 conditions (≠ marks disagreement)")
    fig.tight_layout(rect=(0, .05, 1, .97)); fig.subplots_adjust(left=.29); fig.savefig(f"{out}/comparison_20.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--labeled_abstain", default=None); ap.add_argument("--comparison", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scores = pd.read_parquet(a.scores)
    lab_abs = load_labels(a.labeled_abstain) if a.labeled_abstain and os.path.exists(a.labeled_abstain) else None
    fig_rc(scores, lab_abs, a.out)
    fig_auroc(pd.read_csv(f"{a.tables}/e2_metrics_long.csv"), a.out)
    if os.path.exists(f"{a.tables}/e3_sign_analysis.csv"):
        fig_sign(pd.read_csv(f"{a.tables}/e3_sign_analysis.csv"), a.out)
    if a.comparison and os.path.exists(a.comparison):
        fig_cmp(pd.read_csv(a.comparison), a.out)
    print(f"[make_figs] figures -> {a.out}")


if __name__ == "__main__":
    main()
