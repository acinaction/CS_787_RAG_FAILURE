"""P3: pre-generation probes — standardised logistic regression on the last-prompt-token hidden state.
  s_probe_L1/L2/L3  one probe per layer (floor(L/3), floor(L/2), floor(3L/4)); C chosen by inner CV
  s_probe_best      per outer fold, the layer AND C are chosen by inner CV (never by test AUROC)
Usage:
  python -m src.detect.fit_probes --probe_feats ... --probe_index ... --labels ... --folds ... --scores ...
"""
import argparse, json, os
import numpy as np
import pandas as pd
from src.detect.common import load_labels, merge_scores, save_thresholds
from src.detect.cv_models import nested_cv


def main():
    ap = argparse.ArgumentParser()
    for k in ["probe_feats", "probe_index", "labels", "folds", "scores"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--cross", nargs="*", default=["C2_random"])
    a = ap.parse_args()
    arr = np.load(a.probe_feats).astype(np.float32)
    idx = json.load(open(a.probe_index))
    labels = load_labels(a.labels).set_index("uid").loc[idx["uids"]].reset_index()
    folds = json.load(open(a.folds))
    y, q = labels["failure"].values, labels["qid"].values
    f = labels["qid"].map(folds).values
    out = pd.DataFrame({"uid": labels["uid"]})
    thr = os.path.join(os.path.dirname(os.path.abspath(a.scores)), "thresholds.json")
    log = []
    for j, layer in enumerate(idx["layers"]):
        name = f"s_probe_L{j + 1}"
        s, taus, _ = nested_cv({layer: arr[:, j]}, y, q, f, "lr", name=f"{name}(layer {layer})", log=log)
        out[name] = s; save_thresholds(thr, name, taus)
    s, taus, chosen = nested_cv({layer: arr[:, j] for j, layer in enumerate(idx["layers"])}, y, q, f, "lr",
                                name="s_probe_best", log=log)
    out["s_probe_best"] = s; save_thresholds(thr, "s_probe_best", taus)
    allX = {layer: arr[:, j] for j, layer in enumerate(idx["layers"])}
    for tr_c in a.cross:                      # E7 cross-noise: train on one condition only
        name = f"x_probe_tr{tr_c.split('_')[0]}"
        s, _, _ = nested_cv(allX, y, q, f, "lr", train_mask=(labels["condition"] == tr_c).values, name=name, log=log)
        out[name] = s
    json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(a.scores)), "probe_fit_log.json"), "w"), indent=1)
    merge_scores(a.scores, out)
    print(f"[fit_probes] wrote {list(out.columns[1:])}; best layer per fold: {[c[0] for c in chosen.values()]}")


if __name__ == "__main__":
    main()
