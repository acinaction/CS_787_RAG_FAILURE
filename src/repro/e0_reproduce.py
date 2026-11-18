"""E0 — CPU reproduction of ReDeEP(chunk) on the authors' shipped Llama-2-7B RAGTruth features.

Runs the authors' own `ReDeEP/chunk_level_reg.py` (unchanged except for ONE documented fix) on
`ReDeEP/log/test_llama2_7B/llama2_7B_response_chunk.json`, inside a temporary copy so the
third_party checkout stays untouched.

Two shims are needed, both recorded in the output JSON:
  1. The script reads `../dataset/source_info.jsonl` only to attach a `task_type` column that never
     affects the AUC/PCC. The dataset is on Google Drive, so we write a stub with task_type="unknown".
  2. Bug in the shipped script: `calculate_auc_pcc(df.iloc[:, :int(df.shape[1] * 0.5)], ...)` slices
     COLUMNS, which drops `hallucination_label` and crashes with KeyError. The evident intent is the
     first half of ROWS (features ranked on half the data); we use `df.iloc[:int(df.shape[0] * 0.5)]`.
     We also report the variant that ranks on all rows.

Usage:  python -m src.repro.e0_reproduce [--repo third_party/ReDEeP-ICLR] [--out artifacts/results/e0_reproduction.json]
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

PAPER = {"ReDeEP(token)": {"auc": 0.7325, "pcc": 0.3979}, "ReDeEP(chunk)": {"auc": 0.7458, "pcc": 0.4203}}
BUGGY = "df.iloc[:, :int(df.shape[1] * 0.5)]"


def run_variant(work, script_src, slice_expr, tag):
    src = script_src.replace(BUGGY, slice_expr).replace("ReDeEP(chunk).json", f"ReDeEP(chunk)_{tag}.json")
    path = os.path.join(work, "ReDeEP", f"chunk_level_reg_{tag}.py")
    open(path, "w", encoding="utf-8").write(src)
    env = dict(os.environ, PYTHONUTF8="1")
    out = subprocess.run([sys.executable, os.path.basename(path), "--model_name", "llama2-7b", "--dataset", "ragtruth"],
                         cwd=os.path.join(work, "ReDeEP"), capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    res = json.load(open(os.path.join(work, "ReDeEP", "log", "test_llama2_7B", f"ReDeEP(chunk)_{tag}.json")))
    return {"auc": float(res["auc"]), "pcc": float(res["pcc"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="third_party/ReDEeP-ICLR")
    ap.add_argument("--out", default="artifacts/results/e0_reproduction.json")
    a = ap.parse_args()

    feats = os.path.join(a.repo, "ReDeEP", "log", "test_llama2_7B", "llama2_7B_response_chunk.json")
    script = os.path.join(a.repo, "ReDeEP", "chunk_level_reg.py")
    assert os.path.exists(feats) and os.path.exists(script), "clone github.com/Jeryi-Sun/ReDEeP-ICLR into third_party/ first"
    script_src = open(script, encoding="utf-8").read()
    assert BUGGY in script_src, "authors' script changed upstream; re-check the documented fix"
    shipped = json.load(open(os.path.join(a.repo, "ReDeEP", "log", "test_llama2_7B", "ReDeEP(chunk).json")))
    try:
        sha = subprocess.check_output(["git", "-C", a.repo, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"

    work = tempfile.mkdtemp(prefix="e0_")
    try:
        os.makedirs(os.path.join(work, "ReDeEP", "log", "test_llama2_7B"))
        os.makedirs(os.path.join(work, "dataset"))
        shutil.copy(feats, os.path.join(work, "ReDeEP", "log", "test_llama2_7B"))
        data = json.load(open(feats, encoding="utf-8"))
        with open(os.path.join(work, "dataset", "source_info.jsonl"), "w") as f:
            for s in sorted({d["source_id"] for d in data}):
                f.write(json.dumps({"source_id": s, "task_type": "unknown"}) + "\n")
        n_test = sum(d["split"] == "test" for d in data)
        n_chunks = sum(len(d["scores"]) for d in data if d["split"] == "test")
        half = run_variant(work, script_src, "df.iloc[:int(df.shape[0] * 0.5)]", "halfrows")
        allr = run_variant(work, script_src, "df", "allrows")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    out = {
        "experiment": "E0 reproduction, ReDeEP(chunk), Llama-2-7B-chat, RAGTruth test split, CPU",
        "repo_commit": sha,
        "hyperparameters_used_by_script": {"top_heads": 3, "top_layers": 4, "beta": 0.6, "m": 1},
        "n_responses": n_test, "n_chunks": n_chunks,
        "paper": PAPER,
        "authors_shipped_result": shipped,
        "ours_rank_on_first_half_rows": half,
        "ours_rank_on_all_rows": allr,
        "shims": ["stub source_info.jsonl (task_type unused by metrics)",
                  f"fixed column-slice bug: {BUGGY} -> df.iloc[:int(df.shape[0] * 0.5)]"],
        "note": "Token-level features are not shipped, so ReDeEP(token) cannot be reproduced on CPU; paper value reported for reference.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"paper chunk AUC={PAPER['ReDeEP(chunk)']['auc']} PCC={PAPER['ReDeEP(chunk)']['pcc']}")
    print(f"ours  chunk AUC={half['auc']:.4f} PCC={half['pcc']:.4f}  (all-rows ranking: AUC={allr['auc']:.4f})")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
