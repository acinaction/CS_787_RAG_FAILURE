"""P2: 40-row hand check of the automatic labels (plan gate: >= 90% agreement).

Step 1 (export): stratified sample, 8 rows per condition, to a CSV with an empty `human_failure` column.
  python -m src.data.handcheck export --labeled .../labeled_std.jsonl --bench artifacts/data/benchmark_v2.jsonl \
      --out artifacts/runs/llama2-7b/handcheck_40.csv
Step 2: a person fills `human_failure` (1 = failure, 0 = not; C4: answering at all is a failure) — without
        looking at `auto_failure` first (it is placed last for that reason).
Step 3 (score):
  python -m src.data.handcheck score --csv artifacts/runs/llama2-7b/handcheck_40.csv
"""
import argparse, json, os
import pandas as pd

CONDS = ["C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]


def export(a):
    lab = pd.DataFrame([json.loads(l) for l in open(a.labeled, encoding="utf-8")])
    lab["condition"] = lab["uid"].str.rsplit("__", n=1).str[1]
    bench = {json.loads(l)["uid"]: json.loads(l) for l in open(a.bench, encoding="utf-8")}
    s = pd.concat([lab[lab.condition == c].sample(n=min(a.per_cond, (lab.condition == c).sum()), random_state=a.seed) for c in CONDS])
    out = pd.DataFrame({"uid": s.uid, "condition": s.condition,
                        "question": [bench[u]["question"] for u in s.uid],
                        "gold_answers": [" | ".join(bench[u]["gold_answers"]) for u in s.uid],
                        "swapped_entity": [bench[u]["swapped_entity"] or "" for u in s.uid],
                        "model_answer": s.answer, "human_failure": "", "human_notes": "",
                        "auto_failure": s.failure})
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows -> {a.out}")


def score(a):
    d = pd.read_csv(a.csv, encoding="utf-8-sig")
    if "auto_failure" not in d.columns:                      # blind sheet: take the automatic labels from the full export
        full = pd.read_csv(os.path.join(os.path.dirname(a.csv), "handcheck_40.csv"))[["uid", "auto_failure"]]
        d = d.merge(full, on="uid")
    d = d[d.human_failure.astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    d["human_failure"] = d.human_failure.astype(float).astype(int)
    agree = (d.human_failure == d.auto_failure).mean()
    print(f"rows checked: {len(d)}  agreement: {agree:.1%}  ({'PASS' if agree >= .9 else 'FAIL'} vs 90% gate)")
    print(d.groupby("condition").apply(lambda g: (g.human_failure == g.auto_failure).mean()).round(2).to_string())
    bad = d[d.human_failure != d.auto_failure]
    if len(bad):
        print("\ndisagreements (fix evaluator rules if a pattern shows up):")
        print(bad[["uid", "model_answer", "gold_answers", "auto_failure", "human_failure"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--labeled", required=True); e.add_argument("--bench", required=True)
    e.add_argument("--out", required=True); e.add_argument("--per_cond", type=int, default=8); e.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("score"); s.add_argument("--csv", required=True)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else score(a)
