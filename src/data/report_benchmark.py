"""P2: benchmark audit report — dropped questions + C5 counterfactual validation.

  python -m src.data.report_benchmark --version v2 [--compare v1]

Writes artifacts/data/benchmark_report_<v>.md and c5_validation_<v>.csv (one row per C5 item, for hand review).
C5 checks (automatic; flagged rows must be read by a person):
  type_match     coarse answer type of replacement == of gold (date / num / cap / low)
  wh_match       replacement comes from a SQuAD question with the same wh-word as this question
  swap_correct   no gold string left, replacement present, #replacements == #gold occurrences,
                 and the passage is unchanged outside the swapped spans
  grammar flags  'a'/'an' before the replacement disagrees with its first letter; doubled article
                 ('the the', 'a the'); possessive lost/added ("'s"); plural/singular mismatch (trailing s);
                 lowercase replacement at sentence start
"""
import argparse, collections, json, re
import pandas as pd
from src.data.noise_builder import answer_type, wh_key, contains_any, apply_swap

VOWEL = re.compile(r"^[aeiou]", re.I)


def squad_wh_index():
    from datasets import load_dataset
    idx = collections.defaultdict(set)
    for r in load_dataset("rajpurkar/squad_v2")["validation"]:
        for a in r["answers"]["text"]:
            idx[a].add(wh_key(r["question"]))
    return idx


def swap_spans(gold_text, golds):
    pat = re.compile("|".join(r"(?<!\w)" + re.escape(g) + r"(?!\w)" for g in sorted(golds, key=len, reverse=True) if g.strip()), re.I)
    return list(pat.finditer(gold_text))


def check_row(c5, gold_p, cf_p, wh_idx):
    golds, fake, q = c5["gold_answers"], c5["swapped_entity"], c5["question"]
    main = min(golds, key=len)
    spans = swap_spans(gold_p, golds)
    rebuilt = apply_swap(gold_p, golds, fake)
    n_fake = len(re.findall(re.escape(fake), cf_p, flags=re.I))
    flags = []
    for m in re.finditer(re.escape(fake), cf_p, flags=re.I):
        before = cf_p[max(0, m.start() - 12): m.start()]
        art = re.search(r"\b(a|an)\s+$", before, re.I)
        if art and (art.group(1).lower() == "an") != bool(VOWEL.match(fake)):
            flags.append(f"article '{art.group(1)} {fake[:15]}'")
        if re.search(r"\b(the|a|an)\s+$", before, re.I) and re.match(r"(the|a|an)\s", fake, re.I):
            flags.append("doubled article")
        if (m.start() == 0 or re.search(r"[.!?]\s+$", cf_p[:m.start()])) and cf_p[m.start()].islower():
            flags.append("lowercase at sentence start")
    if main.endswith("'s") != fake.endswith("'s"):
        flags.append("possessive mismatch")
    if answer_type(main) == "low" and (main.split()[-1].endswith("s") != fake.split()[-1].endswith("s")):
        flags.append("plural/singular mismatch?")
    return {
        "uid": c5["uid"], "question": q, "wh": wh_key(q), "gold": " | ".join(golds), "replacement": fake,
        "gold_type": answer_type(main), "repl_type": answer_type(fake),
        "type_match": answer_type(main) == answer_type(fake),
        "wh_match": wh_key(q) in wh_idx.get(fake, set()) or any(wh_key(q) in v for k, v in wh_idx.items() if k.lower().endswith(fake.lower()) and len(k) - len(fake) <= 4),
        "tier": c5.get("swap_tier"),
        "n_gold_occurrences": len(spans), "n_replacements": n_fake,
        "swap_correct": (not contains_any(cf_p, golds)) and n_fake >= 1 and len(spans) >= 1 and rebuilt == cf_p,
        "grammar_flags": "; ".join(dict.fromkeys(flags)),
        "context": (lambda i: cf_p[max(0, i - 70): i + len(fake) + 70])(cf_p.lower().find(fake.lower())),
        "human_ok": "", "human_note": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2"); ap.add_argument("--compare", default="v1")
    ap.add_argument("--data", default="artifacts/data")
    a = ap.parse_args()
    D, v = a.data, a.version
    rows = [json.loads(l) for l in open(f"{D}/benchmark_{v}.jsonl", encoding="utf-8")]
    rep = json.load(open(f"{D}/validation_report_{v}.json"))
    folds = json.load(open(f"{D}/folds_{v}.json")); cmp_set = json.load(open(f"{D}/comparison_set_{v}.json"))
    wh_idx = squad_wh_index()

    by = collections.defaultdict(dict)
    for r in rows:
        by[r["qid"]][r["condition"]] = r
    checks = []
    for q, conds in by.items():
        c5, c1 = conds["C5_counterfactual"], conds["C1_gold"]
        gold_p = c1["passages"][0]["text"]
        cf_p = [p for p in c5["passages"] if p["role"] == "counterfactual"][0]["text"]
        checks.append(check_row(c5, gold_p, cf_p, wh_idx))
    ck = pd.DataFrame(checks)
    ck.to_csv(f"{D}/c5_validation_{v}.csv", index=False)

    # integrity
    fold_sizes = collections.Counter(folds.values())
    integ = {
        "rows": len(rows), "qids": len(by), "all_5_conditions_per_qid": all(len(c) == 5 for c in by.values()),
        "fold_sizes": dict(sorted(fold_sizes.items())), "every_qid_has_fold": set(folds) == set(by),
        "comparison_set_size": len(cmp_set["qids"]), "comparison_subset_of_benchmark": set(cmp_set["qids"]) <= set(by),
        "validation_errors": len(rep["errors"]),
    }
    md = [f"# Benchmark {v} audit report\n", "## Integrity\n", "```json", json.dumps(integ, indent=1), "```\n",
          f"## Dropped questions ({len(rep['dropped'])})\n",
          "Candidates are drawn in a fixed shuffled order (seed 0, 195 spares); a candidate is dropped when its "
          "conditions cannot be built cleanly, and the next candidate takes its place. All 5 conditions of a dropped "
          "question are removed together.\n",
          "| qid | question | gold | replacement | reason |", "|---|---|---|---|---|"]
    for d in rep["dropped"]:
        info = d[2] if len(d) > 2 else {}
        md.append(f"| {d[0]} | {info.get('question', '')[:70]} | {' / '.join(info.get('gold_answers', []))[:40]} | "
                  f"{info.get('replacement', '')} | {d[1]}: {info.get('why', '')}<br>`…{info.get('context_after_swap', '')}…` |")
    if a.compare:
        try:
            old = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{D}/benchmark_{a.compare}.jsonl", encoding="utf-8")}
            oq, nq = set(old), set(by)
            oc = set(json.load(open(f"{D}/comparison_set_{a.compare}.json"))["qids"])
            of = json.load(open(f"{D}/folds_{a.compare}.json"))
            same_fold = sum(of[q] == folds[q] for q in oq & nq)
            md += [f"\n## Difference from {a.compare}\n",
                   f"- questions in both: {len(oq & nq)}; only in {a.compare}: {sorted(oq - nq)}; only in {v}: {sorted(nq - oq)}",
                   f"- C1–C4 rows of shared questions identical: "
                   f"{sum(all(by[q][c] == next(json.loads(l) for l in [json.dumps(old[q])]) or True for c in []) for q in [])}"
                   if False else f"- shared questions keep identical C1–C4 passages (only C5 logic changed): "
                   f"{_same_c1_c4(D, a.compare, v, oq & nq)}",
                   f"- same outer fold as in {a.compare}: {same_fold}/{len(oq & nq)} (folds are re-drawn by the plan's rule, seed 0, over the new qid set)",
                   f"- comparison-set overlap with {a.compare}: {len(oc & set(cmp_set['qids']))}/20 (re-drawn by the plan's rule, seed 1)"]
        except FileNotFoundError:
            pass
    md += [f"\n## C5 counterfactual validation ({len(ck)} rows) — details in `c5_validation_{v}.csv`\n",
           f"- coarse type match: **{ck.type_match.sum()}/{len(ck)}**",
           f"- same question wh-word: **{ck.wh_match.sum()}/{len(ck)}**",
           f"- selection tier (1 = same article + wh + form ... 6 = type only): {dict(sorted(collections.Counter(ck.tier.dropna().astype(int)).items()))}",
           f"- swap correct (gold removed everywhere, replacement inserted, passage otherwise unchanged): **{ck.swap_correct.sum()}/{len(ck)}**",
           f"- rows with grammar flags: **{(ck.grammar_flags != '').sum()}** "
           f"({dict(collections.Counter(f for x in ck.grammar_flags for f in x.split('; ') if f))})",
           "\nFlagged rows (need a human look):\n", "| uid | question | gold → replacement | flags | context |", "|---|---|---|---|---|"]
    bad = ck[(~ck.type_match) | (~ck.swap_correct) | (ck.grammar_flags != "")]
    for _, r in bad.iterrows():
        f = "; ".join(x for x in [("" if r.type_match else "TYPE"), ("" if r.swap_correct else "SWAP"), r.grammar_flags] if x)
        md.append(f"| {r.uid.split('__')[0]} | {r.question[:60]} | {r.gold[:30]} → {r.replacement[:30]} | {f} | `…{r.context.replace('|', '/')[:120]}…` |")
    open(f"{D}/benchmark_report_{v}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:8])); print(f"... -> {D}/benchmark_report_{v}.md, {D}/c5_validation_{v}.csv")
    print(ck[["type_match", "wh_match", "swap_correct"]].sum().to_dict(), "flagged:", len(bad))


def _same_c1_c4(D, va, vb, qids):
    A = {}
    for l in open(f"{D}/benchmark_{va}.jsonl", encoding="utf-8"):
        r = json.loads(l); A[r["uid"]] = r
    n = 0
    for l in open(f"{D}/benchmark_{vb}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["qid"] in qids and r["condition"] != "C5_counterfactual":
            n += A[r["uid"]] == r
    return f"{n}/{4 * len(qids)}"


if __name__ == "__main__":
    main()
