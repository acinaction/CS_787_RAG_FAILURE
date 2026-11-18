"""Demo: show, for one benchmark question, what the model answered in each retrieval condition and what each detector decided.

This REPLAYS the saved outputs of the full run (no GPU needed, nothing is generated live):
  prompt + passages   artifacts/data/benchmark_v2.jsonl
  model answers       artifacts/runs/llama2-7b/labeled_std.jsonl (standard prompt), labeled_abstain.jsonl (abstain prompt)
  detector scores     artifacts/results/llama2-7b/scores.parquet, thresholds.json (cut-offs from cross-validation)

Usage:
  python demo.py                 # the Amazon-drought example, all 5 conditions
  python demo.py --list          # the 20 comparison questions
  python demo.py --pick 3        # question number 3 from that list
  python demo.py --random        # a random question
  python demo.py --pick 3 --cond C4_missing      # only one condition
  python demo.py --qid 572a0bebaf94a219006aa772
"""
import argparse, json, random, sys, textwrap
import pandas as pd

R = "artifacts/results/llama2-7b/"
CONDS = ["C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]
DEFAULT_QID = "572a0bebaf94a219006aa772"
NAMES = {"C1_gold": "C1 gold only", "C2_random": "C2 gold + random passages", "C3_hardneg": "C3 gold + look-alike passages",
         "C4_missing": "C4 look-alike passages, gold REMOVED", "C5_counterfactual": "C5 gold + copy with a FAKE answer"}
DETECTORS = [("s_redeep_c1", "ReDeEP original"), ("s_redeep", "ReDeEP noise-aware"), ("s_probe_best", "Hidden-state probe"), ("s_lnentropy", "LN-entropy")]


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def wrap(t, indent="      ", width=96):
    return textwrap.fill(" ".join(str(t).split()), width=width, initial_indent=indent, subsequent_indent=indent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid"); ap.add_argument("--pick", type=int); ap.add_argument("--list", action="store_true")
    ap.add_argument("--random", action="store_true"); ap.add_argument("--cond", choices=CONDS)
    a = ap.parse_args()

    bench = {r["uid"]: r for r in read("artifacts/data/benchmark_v2.jsonl")}
    cmp_q = json.load(open("artifacts/data/comparison_set_v2.json"))["qids"]
    if a.list:
        for i, q in enumerate(cmp_q, 1):
            print(f"{i:2d}. {bench[q + '__C1_gold']['question']}")
        return
    qid = a.qid or (cmp_q[a.pick - 1] if a.pick else (random.choice(cmp_q) if a.random else DEFAULT_QID))

    std = {r["uid"]: r for r in read("artifacts/runs/llama2-7b/labeled_std.jsonl")}
    ab = {r["uid"]: r for r in read("artifacts/runs/llama2-7b/labeled_abstain.jsonl")}
    sc = pd.read_parquet(R + "scores.parquet").set_index("uid")
    thr = json.load(open(R + "thresholds.json"))

    g = bench[qid + "__C1_gold"]
    print("=" * 100)
    print("QUESTION :", g["question"])
    print("GOLD     :", " | ".join(g["gold_answers"][:3]))
    print("(replay of saved results from the full run; Llama-2-7B-chat, 750 prompts)")
    for c in ([a.cond] if a.cond else CONDS):
        u = f"{qid}__{c}"
        row = bench[u]
        print("\n" + "-" * 100)
        print(f"[{NAMES[c]}]   passages shown to the model: {len(row['passages'])}")
        for i, p in enumerate(row["passages"], 1):
            tag = {"gold": "GOLD", "counterfactual": "FAKE-COPY"}.get(p["role"], "noise")
            print(wrap(f"P{i} ({tag}): {p['text'][:170]}...", indent="   "))
        s_, b_ = std[u], ab[u]
        verdict = "WRONG" if s_["failure"] else "CORRECT"
        if c == "C4_missing":
            verdict = "WRONG (the answer is not in the passages: it should have abstained)" if s_["failure"] else "CORRECT (abstained)"
        print("\n   Model answer (standard prompt):")
        print(wrap(s_["answer"], "      "))
        print(f"      -> {verdict}")
        print("   Model answer (prompt that allows 'I don't know'):")
        print(wrap(b_["answer"], "      "))
        print("      -> " + ("REFUSED (said it cannot answer)" if b_.get("refusal") else "answered"))
        print("   Detectors (score >= cut-off  =>  ABSTAIN):")
        fold = str(int(sc.loc[u, "fold"]))
        for d, nm in DETECTORS:
            tau = (thr.get(d) or {}).get(fold)
            val = float(sc.loc[u, d])
            flag = tau is not None and val >= tau
            ok = (flag == bool(s_["failure"]))
            print(f"      {nm:20s} score {val:8.3f}  cut-off {tau:8.3f}  ->  {'ABSTAIN' if flag else 'ANSWER '}   ({'right call' if ok else 'wrong call'})")
    print("=" * 100)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
