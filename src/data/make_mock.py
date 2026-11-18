"""P2: fake data in the REAL schemas so P1/P3 can test the whole pipeline before real runs exist.
All numbers are synthetic. Never report anything computed from mock files.

python -m src.data.make_mock --bench artifacts/data/benchmark_v2.jsonl --out artifacts/data/mock

Writes: labeled_std.jsonl, labeled_abstain.jsonl, redeep_feats.parquet, redeep_meta.json,
        probe_feats.npy, probe_index.json, redeep_tokens_cmp.json
Planted structure (so analysis code has something to find):
  - failure rate rises with noise (C1 < C2 < C3 < C5 < C4)
  - copy-head ECS is LOWER for failures, except in C5 where copying the swapped passage causes the
    failure (ECS HIGHER for failures)  -> E3 sign flip
  - late-layer PKS (layers 16-31) HIGHER for failures (paper Fig. 4d direction)
"""
import argparse, json, os
import numpy as np
import pandas as pd

FAIL = {"C1_gold": .15, "C2_random": .25, "C3_hardneg": .35, "C4_missing": .8, "C5_counterfactual": .5}
REFUSE_ABSTAIN = {"C1_gold": .03, "C2_random": .05, "C3_hardneg": .1, "C4_missing": .55, "C5_counterfactual": .1}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=32); ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--cmp_set", default=None)
    a = ap.parse_args(); rng = np.random.default_rng(0)
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.bench, encoding="utf-8")]
    conds = [r["condition"] for r in rows]
    fail = np.array([rng.random() < FAIL[c] for c in conds]).astype(int)

    for template in ("std", "abstain"):
        with open(f"{a.out}/labeled_{template}.jsonl", "w", encoding="utf-8") as f:
            for r, fl in zip(rows, fail):
                c = r["condition"]
                refusal = bool(template == "abstain" and rng.random() < REFUSE_ABSTAIN[c])
                fl_t = int(not refusal) if c == "C4_missing" else int(fl or refusal)
                n = int(rng.integers(2, 9))
                f.write(json.dumps({
                    "uid": r["uid"], "qid": r["qid"], "condition": c, "model": "mock", "template": template,
                    "prompt_version": "prompt_v1", "answer": "I don't know" if refusal else ("wrong" if fl_t else r["gold_answers"][0]),
                    "n_answer_tokens": n, "answer_token_ids": list(range(n)),
                    "answer_token_logprobs": list(rng.normal(-0.4 - 0.6 * fl_t, .3, n)),
                    "answer_token_entropies": list(np.abs(rng.normal(0.8 + 0.8 * fl_t, .4, n))),
                    "first_token_entropy": float(abs(rng.normal(1 + fl_t, .5))), "first_token_maxprob": float(rng.random()),
                    "prompt_len": 400, "gen_seconds": 1.0,
                    "refusal": refusal, "answered": not refusal, "f1": 0.0 if fl_t else 1.0,
                    "correct": int(not fl_t and c != "C4_missing"), "failure": fl_t,
                    "followed_counterfactual": bool(c == "C5_counterfactual" and fl_t and rng.random() < .7)}) + "\n")

    recs, tok_dump = [], {}
    sign = np.array([1.0 if c == "C5_counterfactual" else -1.0 for c in conds])
    for i, r in enumerate(rows):
        n = int(rng.integers(2, 9))
        ecs = rng.normal(0.55, .06, (n, a.K)) + rng.normal(0, .03) + 0.025 * sign[i] * fail[i] * (np.arange(a.K) < 6)
        pks = np.abs(rng.normal(3.0, .8, (n, a.L)) + rng.normal(0, .6))     # row-level offset = nuisance
        pks[:, 16:] += 0.25 * fail[i]
        recs.append({"uid": r["uid"], "n_tokens": n, "tf_match": 1.0, "ecs_tok": ecs.tolist(), "pks_tok": pks.tolist(),
                     "ecs_mean": ecs.mean(0).tolist(), "ecs_first": ecs[0].tolist(),
                     "pks_mean": pks.mean(0).tolist(), "pks_first": pks[0].tolist()})
    pd.DataFrame(recs).to_parquet(f"{a.out}/redeep_feats.parquet", index=False)
    json.dump({"model": "mock", "head_list": [[l, h] for l, h in sorted((i % 32, (5 * i) % 32) for i in range(a.K))],
               "n_layers": a.L, "pks_layers": list(range(a.L)), "probe_layers": [10, 16, 24],
               "implementation": "MOCK"}, open(f"{a.out}/redeep_meta.json", "w"), indent=2)
    np.save(f"{a.out}/probe_feats.npy",
            (rng.normal(0, 1, (len(rows), 3, a.dim)) + np.array([.1, .25, .15])[None, :, None] * fail[:, None, None]).astype("float16"))
    json.dump({"uids": [r["uid"] for r in rows], "layers": [10, 16, 24]}, open(f"{a.out}/probe_index.json", "w"))
    if a.cmp_set:
        q = set(json.load(open(a.cmp_set))["qids"])
        for rec in recs:
            if rec["uid"].rsplit("__", 1)[0] in q:
                tok_dump[rec["uid"]] = {"tokens": [f"tok{j}" for j in range(rec["n_tokens"])],
                                        "ecs_tok": rec["ecs_tok"], "pks_tok": rec["pks_tok"]}
        json.dump({"head_list": [], "rows": tok_dump}, open(f"{a.out}/redeep_tokens_cmp.json", "w"))
    print("mock written:", len(rows), "rows, failure rate", fail.mean().round(2))
