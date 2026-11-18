"""Everything after the GPU: labels -> detectors -> tables -> figures -> 20-prompt comparison. CPU only.

  python -m src.run_cpu_pipeline --run artifacts/runs/llama2-7b --results artifacts/results/llama2-7b
  python -m src.run_cpu_pipeline --mock       # whole pipeline on mock data (artifacts/data/mock)

Expects in --run: gen_std.jsonl, gen_abstain.jsonl (labels are made here), redeep_feats.parquet,
redeep_meta.json, probe_feats.npy, probe_index.json, redeep_tokens_cmp.json (optional).
With --mock, labeled_*.jsonl are taken as-is from artifacts/data/mock.
"""
import argparse, os, subprocess, sys, time

DATA = "artifacts/data"


def sh(*args):
    t = time.time()
    print("\n$ python -m " + " ".join(args), flush=True)
    r = subprocess.run([sys.executable, "-m", *args])
    if r.returncode:
        sys.exit(f"step failed: {args[0]}")
    print(f"  ({time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="artifacts/runs/llama2-7b")
    ap.add_argument("--results", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no_mlp", action="store_true")
    ap.add_argument("--skip_label", action="store_true", help="labeled_*.jsonl already exist in --run")
    a = ap.parse_args()
    run = f"{DATA}/mock" if a.mock else a.run
    res = a.results or ("artifacts/results/mock" if a.mock else run.replace("runs", "results"))
    os.makedirs(res, exist_ok=True)
    bench, folds, cmp_set = f"{DATA}/benchmark_v2.jsonl", f"{DATA}/folds_v2.json", f"{DATA}/comparison_set_v2.json"
    lab_std, lab_abs = f"{run}/labeled_std.jsonl", f"{run}/labeled_abstain.jsonl"
    if not (a.mock or a.skip_label):
        for t in ("std", "abstain"):
            if os.path.exists(f"{run}/gen_{t}.jsonl"):
                sh("src.data.evaluator", "--gen", f"{run}/gen_{t}.jsonl", "--bench", bench, "--out", f"{run}/labeled_{t}.jsonl")
    for f in ["scores.parquet", "thresholds.json"]:
        if os.path.exists(f"{res}/{f}"):
            os.remove(f"{res}/{f}")          # rebuild from scratch: never mix runs
    S = f"{res}/scores.parquet"
    sh("src.detect.baselines", "--labels", lab_std, "--bench", bench, "--folds", folds, "--scores", S)
    sh("src.detect.fit_redeep", "--feats", f"{run}/redeep_feats.parquet", "--meta", f"{run}/redeep_meta.json",
       "--labels", lab_std, "--folds", folds, "--scores", S)
    sh("src.detect.fit_lr", "--feats", f"{run}/redeep_feats.parquet", "--labels", lab_std, "--bench", bench,
       "--folds", folds, "--scores", S, *(["--no_mlp"] if a.no_mlp else []))
    if os.path.exists(f"{run}/probe_feats.npy"):
        sh("src.detect.fit_probes", "--probe_feats", f"{run}/probe_feats.npy", "--probe_index", f"{run}/probe_index.json",
           "--labels", lab_std, "--folds", folds, "--scores", S)
    sh("src.analysis.make_tables", "--scores", S, "--labeled_std", lab_std, "--labeled_abstain", lab_abs,
       "--feats", f"{run}/redeep_feats.parquet", "--out", f"{res}/tables")
    sh("src.analysis.make_comparison", "--scores", S, "--labeled_std", lab_std, "--bench", bench, "--cmp_set", cmp_set,
       "--tokens", f"{run}/redeep_tokens_cmp.json", "--out", f"{res}/comparison")
    sh("src.analysis.make_figs", "--scores", S, "--tables", f"{res}/tables", "--labeled_abstain", lab_abs,
       "--comparison", f"{res}/comparison/comparison_20.csv", "--out", f"{res}/figs")
    print(f"\nDONE -> {res}  (tables/tables.md, figs/, comparison/)")


if __name__ == "__main__":
    main()
