import argparse, json, os
import numpy as np
import pandas as pd
from src.detect.common import load_labels, merge_scores, read_jsonl, save_thresholds
from src.detect.cv_models import nested_cv


def feature_matrix(df):
    E = np.stack([np.asarray(x, float) for x in df["ecs_mean"]])
    P = np.stack([np.asarray(x, float) for x in df["pks_mean"]])
    X = np.hstack([E, np.log1p(np.maximum(P, 0))])          # PKS is heavy-tailed (JSD*1e6)
    col_mean = np.nanmean(X, 0)
    return np.where(np.isfinite(X), X, col_mean)              # empty answers -> column means


def main():
    ap = argparse.ArgumentParser()
    for k in ["feats", "labels", "bench", "folds", "scores"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--no_mlp", action="store_true")
    a = ap.parse_args()
    labels = load_labels(a.labels)
    df = labels[["uid", "qid", "condition", "failure"]].merge(pd.read_parquet(a.feats), on="uid")
    folds = json.load(open(a.folds)); f = df["qid"].map(folds).values
    bench = {r["uid"]: r for r in read_jsonl(a.bench)}
    X = feature_matrix(df)
    B = np.array([[bench[u]["bm25_margin"], bench[u]["bm25_top1"]] for u in df.uid], float)
    y, q = df["failure"].values, df["qid"].values
    out = pd.DataFrame({"uid": df.uid})
    thr = os.path.join(os.path.dirname(os.path.abspath(a.scores)), "thresholds.json")
    log = []
    jobs = [("s_lr_ecspks", {"ecs+pks": X}, "lr"), ("s_lr_ecspks_bm25", {"ecs+pks+bm25": np.hstack([X, B])}, "lr")]
    if not a.no_mlp:
        jobs.append(("s_mlp_ecspks", {"ecs+pks": X}, "mlp"))
    for name, Xs, kind in jobs:
        s, taus, _ = nested_cv(Xs, y, q, f, kind, name=name, log=log)
        out[name] = s; save_thresholds(thr, name, taus)
    json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(a.scores)), "lr_fit_log.json"), "w"), indent=1)
    merge_scores(a.scores, out)
    print(f"[fit_lr] wrote {list(out.columns[1:])}")


if __name__ == "__main__":
    main()
