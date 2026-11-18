import argparse, json, os, subprocess, sys
from collections import Counter

CFG = json.load(open("artifacts/data/experiment_config_v2.json")) if os.path.exists("artifacts/data/experiment_config_v2.json") else {}
FIRST = CFG.get("generation", {}).get("max_new_tokens", 128)
REGEN = CFG.get("generation", {}).get("regenerate_truncated_with_max_new_tokens", 256)
REGEN2 = CFG.get("generation", {}).get("second_regeneration_max_new_tokens", 512)   # tier 3 (added 2026-11-08, DEVIATIONS #4b)
ALLOWED_TRUNC = CFG.get("generation", {}).get("max_truncated_rows_allowed_after_regeneration", 0)

REQUIRED = ["uid", "model", "template", "prompt_version", "answer", "n_answer_tokens", "answer_token_ids",
            "answer_token_logprobs", "first_token_entropy", "first_token_maxprob", "prompt_len", "gen_seconds",
            "answer_token_entropies", "stopped_by_eos"]


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def truncated(rows):
    return [r for r in rows if r.get("stopped_by_eos") is False]     # a missing field is reported by validate()


def merge(base, regen):
    """Replace truncated rows of `base` by their regenerated version. Returns (rows, stats)."""
    by_new = {r["uid"]: r for r in regen}
    out, stats = [], Counter()
    for r in base:
        n = by_new.get(r["uid"])
        if r["stopped_by_eos"] or n is None:
            out.append(dict(r, regenerated=r.get("regenerated", False))); continue
        k = len(r["answer_token_ids"])
        stats["prefix_match" if n["answer_token_ids"][:k] == r["answer_token_ids"] else "prefix_MISMATCH"] += 1
        stats["now_complete" if n["stopped_by_eos"] else "still_truncated_at_regen_limit"] += 1
        out.append(dict(n, regenerated=True, first_pass_tokens=r.get("first_pass_tokens", k)))
    return out, dict(stats)


def validate(rows, bench_uids, template, allowed_trunc=ALLOWED_TRUNC):
    """Returns (errors, report)."""
    errs = []
    uids = [r["uid"] for r in rows]
    if len(uids) != len(set(uids)):
        errs.append(f"duplicate uids: {len(uids) - len(set(uids))}")
    if set(uids) != set(bench_uids):
        errs.append(f"uid set != benchmark: missing {len(set(bench_uids) - set(uids))}, extra {len(set(uids) - set(bench_uids))}")
    miss = Counter(f for r in rows for f in REQUIRED if f not in r)
    if miss:
        errs.append(f"missing fields: {dict(miss)}")
    bad_t = [r["uid"] for r in rows if r.get("template") != template]
    if bad_t:
        errs.append(f"{len(bad_t)} rows with wrong template")
    inconsistent = [r["uid"] for r in rows if all(f in r for f in REQUIRED) and (
                    r["n_answer_tokens"] != len(r["answer_token_ids"])
                    or len(r["answer_token_logprobs"]) != len(r["answer_token_ids"])
                    or len(r["answer_token_entropies"]) != len(r["answer_token_ids"]))]
    if inconsistent:
        errs.append(f"{len(inconsistent)} rows with inconsistent token/logprob lengths")
    still = truncated(rows)
    if len(still) > allowed_trunc:
        errs.append(f"{len(still)} rows still truncated after regeneration (allowed {allowed_trunc})")
    empty = sum(not r.get("answer", "x").strip() for r in rows)
    lens = sorted(r.get("n_answer_tokens", 0) for r in rows)
    rep = {"template": template, "rows": len(rows), "still_truncated": len(still), "still_truncated_uids": [r["uid"] for r in still][:20],
           "empty_answers": empty, "empty_rate": round(empty / max(len(rows), 1), 4),
           "regenerated_rows": sum(bool(r.get("regenerated")) for r in rows),
           "answer_tokens": {"median": lens[len(lens) // 2], "p99": lens[int(.99 * len(lens)) - 1], "max": lens[-1]} if lens else {},
           "errors": errs}
    return errs, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CFG.get("model", "meta-llama/Llama-2-7b-chat-hf"))
    ap.add_argument("--bench", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--template", choices=["std", "abstain"], required=True)
    ap.add_argument("--first_limit", type=int, default=FIRST); ap.add_argument("--regen_limit", type=int, default=REGEN)
    ap.add_argument("--regen2_limit", type=int, default=REGEN2, help="third tier: rows still without EOS at --regen_limit")
    ap.add_argument("--no_regen", action="store_true", help="only merge/validate existing files (no GPU)")
    a = ap.parse_args()
    t, R = a.template, a.run
    f128, fregen, ffinal = f"{R}/gen_{t}_{a.first_limit}.jsonl", f"{R}/gen_{t}_regen{a.regen_limit}.jsonl", f"{R}/gen_{t}.jsonl"
    base = read(f128)
    bench_uids = [json.loads(l)["uid"] for l in open(a.bench, encoding="utf-8")]
    todo = [r["uid"] for r in truncated(base)]
    print(f"[finalize:{t}] {len(base)} rows at limit {a.first_limit}; {len(todo)} truncated -> regenerate at {a.regen_limit}")
    reuse = os.path.exists(fregen) and {r["uid"] for r in read(fregen)} == set(todo)     # resume: exactly these rows already regenerated
    if reuse:
        print(f"[finalize:{t}] reusing existing {fregen} (exactly the {len(todo)} truncated rows)")
    if todo and not a.no_regen and not reuse:
        uf = f"{R}/gen_{t}_truncated_uids.txt"
        open(uf, "w").write("\n".join(todo) + "\n")
        if os.path.exists(fregen):
            os.remove(fregen)       # the file must contain exactly this run's rows
        cmd = [sys.executable, "-m", "src.gen.gen", "--model", a.model, "--bench", a.bench, "--uids", uf, "--template", t,
               "--max_new_tokens", str(a.regen_limit), "--out", fregen, "--checkpoint_every", "10"]
        print("$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode:
            sys.exit("regeneration failed")
    regen = read(fregen) if os.path.exists(fregen) else []
    if todo and {r["uid"] for r in regen} != set(todo):
        sys.exit(f"regeneration incomplete: {len(regen)}/{len(todo)} rows")
    merged, stats = merge(base, regen)
    # tier 3: rows still without EOS at regen_limit are regenerated (alone) at regen2_limit
    todo2 = [r["uid"] for r in truncated(merged)]
    fregen2 = f"{R}/gen_{t}_regen{a.regen2_limit}.jsonl"
    regen2 = []
    if todo2:
        print(f"[finalize:{t}] {len(todo2)} rows still truncated at {a.regen_limit} -> regenerate at {a.regen2_limit}")
        if not a.no_regen:
            uf2 = f"{R}/gen_{t}_truncated2_uids.txt"
            open(uf2, "w").write("\n".join(todo2) + "\n")
            if os.path.exists(fregen2) and {r["uid"] for r in read(fregen2)} != set(todo2):
                os.remove(fregen2)
            cmd = [sys.executable, "-m", "src.gen.gen", "--model", a.model, "--bench", a.bench, "--uids", uf2, "--template", t,
                   "--max_new_tokens", str(a.regen2_limit), "--out", fregen2, "--checkpoint_every", "10"]
            print("$", " ".join(cmd), flush=True)
            if subprocess.run(cmd).returncode:
                sys.exit("tier-3 regeneration failed")
        regen2 = read(fregen2) if os.path.exists(fregen2) else []
        if {r["uid"] for r in regen2} != set(todo2):
            sys.exit(f"tier-3 regeneration incomplete: {len(regen2)}/{len(todo2)} rows")
        merged, stats2 = merge(merged, regen2)
        stats["tier3"] = stats2
    # keep benchmark order
    order = {u: i for i, u in enumerate(bench_uids)}
    merged.sort(key=lambda r: order.get(r["uid"], 1e9))
    errs, rep = validate(merged, bench_uids, t)
    rep.update({"first_limit": a.first_limit, "regen_limit": a.regen_limit, "truncated_after_first_pass": len(todo), "truncated_after_tier2": len(todo2), "regen2_limit": a.regen2_limit, "merge_stats": stats})
    if stats.get("prefix_MISMATCH"):
        print(f"[finalize:{t}] NOTE: {stats['prefix_MISMATCH']} regenerated rows differ from the first pass inside the first {a.first_limit} tokens "
              f"(GPU numerics); the {a.regen_limit}-token version is kept")
    with open(ffinal, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    json.dump(rep, open(f"{R}/gen_{t}_final_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "still_truncated_uids"}, indent=1))
    if errs:
        print("VALIDATION FAILED:", errs); sys.exit(1)
    print(f"[finalize:{t}] OK -> {ffinal}")


if __name__ == "__main__":
    main()
