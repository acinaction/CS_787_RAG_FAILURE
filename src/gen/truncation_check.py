"""Choose max_new_tokens from data, not by guess.

Step 1 (CPU) sample:  python -m src.gen.truncation_check sample --bench artifacts/data/benchmark_v2.jsonl --out artifacts/data/trunc_sample_uids.txt
   -> 60 uids: 12 per condition, stratified, seed 2 (questions NOT restricted to the comparison set).
Step 2 (GPU): run gen.py on those uids, both templates, with a generous --max_new_tokens 160.
Step 3 (CPU) report:  python -m src.gen.truncation_check report --gen <gen_std_160.jsonl> <gen_abstain_160.jsonl> \
                         --bench artifacts/data/benchmark_v2.jsonl --out artifacts/runs/llama2-7b/truncation_report.json
Greedy decoding is prefix-deterministic, so the first L tokens of the 160-token run are exactly what a run with
max_new_tokens = L would produce. For each L in {32, 48, 64, 96, 128} we report the share of answers that would
be cut off (no EOS within L tokens) and the share whose LABEL would change if cut at L.
Decision rule (proposed): smallest L in {64, 96, 128} with truncation <= 2% and label changes <= 1% in every
condition and template; 64 if it qualifies (it is what the Day-1 runs used).
"""
import argparse, json, random, collections
from src.data.evaluator import label_row

LS = [32, 48, 64, 96, 128]


def sample(a):
    rows = [json.loads(l) for l in open(a.bench, encoding="utf-8")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r["uid"])
    rng = random.Random(2)
    uids = [u for c in sorted(by) for u in rng.sample(sorted(by[c]), a.per_cond)]
    open(a.out, "w").write("\n".join(uids) + "\n")
    print(f"{len(uids)} uids -> {a.out}")


def report(a):
    from transformers import AutoTokenizer
    bench = {}
    for l in open(a.bench, encoding="utf-8"):
        r = json.loads(l); bench[r["uid"]] = r
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    out = {"max_new_tokens_of_sample_run": a.gen_limit, "by_L": {}}
    for path in a.gen:
        g = [json.loads(l) for l in open(path, encoding="utf-8")]
        t = g[0]["template"]
        full_trunc = sum(not r.get("stopped_by_eos", r["n_answer_tokens"] < a.gen_limit) for r in g)
        out[f"{t}_still_truncated_at_{a.gen_limit}"] = full_trunc
        for L in LS:
            stats = collections.defaultdict(lambda: {"n": 0, "truncated": 0, "label_changed": 0})
            for r in g:
                b = bench[r["uid"]]; c = b["condition"]
                ended = r.get("stopped_by_eos", True) and r["n_answer_tokens"] < L
                cut = dict(r, answer=tok.decode(r["answer_token_ids"][:L], skip_special_tokens=True).strip())
                s = stats[c]; s["n"] += 1
                s["truncated"] += int(not ended)
                s["label_changed"] += int(label_row(cut, b)["failure"] != label_row(r, b)["failure"])
            out["by_L"].setdefault(str(L), {})[t] = dict(stats)
    lens = [json.loads(l)["n_answer_tokens"] for p in a.gen for l in open(p, encoding="utf-8")]
    lens.sort()
    out["answer_length_tokens"] = {"median": lens[len(lens) // 2], "p90": lens[int(.9 * len(lens))], "p99": lens[int(.99 * len(lens)) - 1], "max": lens[-1]}
    choice = None
    for L in (64, 96, 128):
        worst_t = max(s["truncated"] / s["n"] for t in out["by_L"][str(L)].values() for s in t.values())
        worst_l = max(s["label_changed"] / s["n"] for t in out["by_L"][str(L)].values() for s in t.values())
        out["by_L"][str(L)]["worst_truncation_rate"] = worst_t; out["by_L"][str(L)]["worst_label_change_rate"] = worst_l
        if choice is None and worst_t <= .02 and worst_l <= .01:
            choice = L
    out["proposed_max_new_tokens"] = choice or 160
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_L"}, indent=1))
    for L in LS:
        row = out["by_L"][str(L)]
        print(f"L={L:4d} " + "  ".join(f"{t}: trunc={sum(s['truncated'] for s in v.values())}/{sum(s['n'] for s in v.values())}"
                                    f" label_chg={sum(s['label_changed'] for s in v.values())}"
                                    for t, v in row.items() if isinstance(v, dict)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample"); s.add_argument("--bench", required=True); s.add_argument("--out", required=True)
    s.add_argument("--per_cond", type=int, default=12)
    r = sp.add_parser("report"); r.add_argument("--gen", nargs="+", required=True); r.add_argument("--bench", required=True)
    r.add_argument("--out", required=True); r.add_argument("--gen_limit", type=int, default=160)
    r.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    r.add_argument("--revision", default="f5db02db724555f92da89c216ac04704f23d4590")
    a = ap.parse_args()
    sample(a) if a.cmd == "sample" else report(a)
