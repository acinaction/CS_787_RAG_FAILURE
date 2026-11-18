"""LEGACY (benchmark v1, Day-1 Kaggle build). Verbatim copy of src/data/noise_builder.py from rag-noise-probe-fixed.zip
(2026-11-06), kept only to reproduce benchmark_v1.jsonl. Do not edit. Use src/data/noise_builder.py (v2) for new builds."""
"""P2: builds the 5 retrieval conditions (SQuAD 2.0 branch).
Run:  python -m src.data.noise_builder --n 150 --out artifacts/data
Mini: python -m src.data.noise_builder --n 20 --mini --out artifacts/data
"""
import argparse, hashlib, json, random, re
import numpy as np

CONDS = ["C1_gold", "C2_random", "C3_hardneg", "C4_missing", "C5_counterfactual"]

def contains_any(text, answers):
    t = text.lower()
    return any(a.strip() and a.lower() in t for a in answers)

def uid_rng(seed, uid):
    return random.Random(int(hashlib.md5(f"{seed}-{uid}".encode()).hexdigest()[:8], 16))

class Retriever:
    """BM25 (rank_bm25) with a TF-IDF fallback. scores(q) -> np.array over the pool."""
    def __init__(self, texts):
        self.tok = lambda s: re.findall(r"\w+", s.lower())
        try:
            from rank_bm25 import BM25Okapi
            self.mode, self.bm25 = "bm25", BM25Okapi([self.tok(t) for t in texts])
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.mode = "tfidf"
            self.vec = TfidfVectorizer(lowercase=True).fit(texts)
            self.mat = self.vec.transform(texts)
    def scores(self, q):
        if self.mode == "bm25":
            return np.asarray(self.bm25.get_scores(self.tok(q)))
        return (self.mat @ self.vec.transform([q]).T).toarray().ravel()

def build_rows(examples, pool, k=3, seed=0):
    """examples: [{qid, question, gold_answers, gold_pid}], pool: [{pid, title, text}]."""
    rng = random.Random(seed)
    idx = {p["pid"]: i for i, p in enumerate(pool)}
    ret = Retriever([p["text"] for p in pool])
    all_answers = [a for e in examples for a in e["gold_answers"]]
    rows, dropped = [], []
    for e in examples:
        ans = e["gold_answers"]
        gi = idx[e["gold_pid"]]
        gold = pool[gi]
        sc = ret.scores(e["question"])
        order = np.argsort(-sc)
        hard = [int(i) for i in order if i != gi and not contains_any(pool[i]["text"], ans)][:k]
        rand, tries = [], 0
        while len(rand) < k and tries < 500:
            tries += 1
            i = rng.randrange(len(pool))
            if i != gi and i not in rand and not contains_any(pool[i]["text"], ans):
                rand.append(i)
        fakes = [a for a in all_answers if not contains_any(a, ans) and not contains_any(" ".join(ans), [a])]
        if len(hard) < k or len(rand) < k or not fakes:
            dropped.append((e["qid"], "not enough clean distractors")); continue
        fake = rng.choice(fakes)
        cf = gold["text"]
        for a in sorted(ans, key=len, reverse=True):
            cf = re.sub(re.escape(a), fake, cf, flags=re.I)
        if contains_any(cf, ans) or fake.lower() not in cf.lower():
            dropped.append((e["qid"], "counterfactual swap failed")); continue
        P = lambda i, role: {"pid": pool[i]["pid"], "text": pool[i]["text"], "role": role}
        G = P(gi, "gold")
        CF = {"pid": gold["pid"] + "_cf", "text": cf, "role": "counterfactual"}
        conds = {
            "C1_gold": [G],
            "C2_random": [G] + [P(i, "random") for i in rand],
            "C3_hardneg": [G] + [P(i, "hardneg") for i in hard],
            "C4_missing": [P(i, "hardneg") for i in hard],
            "C5_counterfactual": [G, CF],
        }
        for c, ps in conds.items():
            uid = f"{e['qid']}__{c}"
            ps = list(ps); uid_rng(seed, uid).shuffle(ps)
            roles = [p["role"] for p in ps]
            if "gold" in roles:
                gp = roles.index("gold")
                pos = "first" if gp == 0 else ("last" if gp == len(ps) - 1 else "middle")
            else:
                pos = "none"
            # retriever features: pool-level score of each shown passage (counterfactual copy uses the gold's score)
            s = sorted([sc[idx[p["pid"].replace("_cf", "")]] for p in ps], reverse=True)
            rows.append({
                "uid": uid, "qid": e["qid"], "condition": c, "question": e["question"],
                "gold_answers": ans, "answerable": c != "C4_missing", "passages": ps,
                "n_distractors": sum(r in ("random", "hardneg") for r in roles),
                "gold_position": pos,
                "bm25_top1": float(s[0]), "bm25_margin": float(s[0] - (s[1] if len(s) > 1 else 0.0)),
                "swapped_entity": fake if c == "C5_counterfactual" else None})
    return rows, dropped

def validate(rows):
    errs, uids, byq = [], set(), {}
    for r in rows:
        if r["uid"] in uids: errs.append(f"duplicate uid {r['uid']}")
        uids.add(r["uid"]); byq.setdefault(r["qid"], set()).add(r["condition"])
        if not r["gold_answers"]: errs.append(f"{r['uid']}: no gold answers")
        c, ans = r["condition"], r["gold_answers"]
        non_gold = [p for p in r["passages"] if p["role"] in ("random", "hardneg")]
        if any(contains_any(p["text"], ans) for p in non_gold): errs.append(f"{r['uid']}: answer leaked into distractor")
        if c == "C4_missing" and any(p["role"] == "gold" for p in r["passages"]): errs.append(f"{r['uid']}: gold present in C4")
        if c == "C5_counterfactual":
            cf = [p for p in r["passages"] if p["role"] == "counterfactual"]
            if not cf or contains_any(cf[0]["text"], ans) or r["swapped_entity"].lower() not in cf[0]["text"].lower():
                errs.append(f"{r['uid']}: bad counterfactual")
        if r["answerable"] != (c != "C4_missing"): errs.append(f"{r['uid']}: answerable flag wrong")
    for q, cs in byq.items():
        if cs != set(CONDS): errs.append(f"{q}: missing conditions {set(CONDS) - cs}")
    return errs

def make_folds(qids, n_folds=5, seed=0):
    q = sorted(qids); random.Random(seed).shuffle(q)
    return {x: i % n_folds for i, x in enumerate(q)}

def make_comparison_set(qids, n=20, seed=1):
    return {"seed": seed, "qids": sorted(random.Random(seed).sample(sorted(qids), n))}

def load_squad_v2(n, seed=0, max_words=220, max_ans_words=5):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2")
    pool, seen = [], {}
    for split in ("train", "validation"):
        for r in ds[split]:
            c = r["context"]
            if c not in seen and len(c.split()) <= max_words:
                seen[c] = hashlib.md5(c.encode()).hexdigest()[:12]
                pool.append({"pid": seen[c], "title": r["title"], "text": c})
    rng = random.Random(seed)
    cands = [r for r in ds["validation"] if r["answers"]["text"] and r["context"] in seen
             and len(r["answers"]["text"][0].split()) <= max_ans_words]
    rng.shuffle(cands)
    ex, used = [], set()
    for r in cands:
        if r["context"] in used: continue          # one question per paragraph
        used.add(r["context"])
        ex.append({"qid": r["id"], "question": r["question"],
                   "gold_answers": sorted(set(r["answers"]["text"])), "gold_pid": seen[r["context"]]})
        if len(ex) >= int(n * 1.3): break          # spare questions in case some get dropped
    return ex, pool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--out", default="artifacts/data")
    a = ap.parse_args()
    ex, pool = load_squad_v2(a.n, a.seed)
    rows, dropped = build_rows(ex, pool, a.k, a.seed)
    keep = [r["qid"] for r in rows if r["condition"] == "C1_gold"][: a.n]
    ks = set(keep)
    rows = [r for r in rows if r["qid"] in ks]
    errs = validate(rows)
    name = "benchmark_mini_v0.jsonl" if a.mini else "benchmark_v1.jsonl"
    with open(f"{a.out}/{name}", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    if not a.mini:
        json.dump(make_folds(keep, 5, a.seed), open(f"{a.out}/folds_v1.json", "w"), indent=1)
        json.dump(make_comparison_set(keep, 20, 1), open(f"{a.out}/comparison_set_v1.json", "w"), indent=1)
    json.dump({"rows": len(rows), "qids": len(keep), "dropped": dropped, "errors": errs},
              open(f"{a.out}/validation_report{'_mini' if a.mini else ''}.json", "w"), indent=1)
    print(f"rows={len(rows)} qids={len(keep)} dropped={len(dropped)} errors={len(errs)}")
