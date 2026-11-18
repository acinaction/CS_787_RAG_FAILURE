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

NUM_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "hundred", "thousand", "million", "billion", "half", "dozen"}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"}

def answer_type(a):
    """Coarse answer type so a counterfactual swap stays plausible:
    'date' (year/month), 'num' (other numbers), 'cap' (proper noun), 'low' (common phrase)."""
    toks = re.findall(r"[\w'%-]+", a)
    low = [t.lower() for t in toks]
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2})s?\b", a) or any(t in MONTHS for t in low):
        return "date"
    if re.search(r"\d", a) or any(t in NUM_WORDS for t in low):
        return "num"
    content = [t for t in toks if t.lower() not in {"the", "a", "an", "of", "and", "de", "in", "for"}]
    if content and all(t[0].isupper() for t in content):
        return "cap"
    return "low"

WH_KEYS = ["how many", "how much", "how long", "how old", "what year", "which year", "what percentage",
           "whom", "whose", "who", "when", "where", "why", "how", "which", "what"]

def wh_key(question):
    q = question.lower()
    hits = [(m.start(), -len(k), k) for k in WH_KEYS for m in [re.search(r"\b" + k + r"\b", q)] if m]
    return min(hits)[2] if hits else "other"

ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)

def _form(a):
    """surface form: possessive and plural markers of the last word."""
    last = a.split()[-1] if a.split() else a
    plural = last.lower().endswith("s") and not last.lower().endswith(("ss", "us", "is", "'s"))
    return (a.endswith("'s"), plural)

def _clean(a):
    a = a.strip().strip(",;:")
    balanced = all(a.count(o) == a.count(c) for o, c in ["()", "[]", "{}"]) and a.count('"') % 2 == 0
    return a if balanced and len(a) >= 2 and not re.search(r"^[^\w\"']|[,;:(]$", a) else None

def pick_fake(ans, pool_answers, rng, question=None, title=None, gold_text=""):
    """Choose a plausible replacement for C5 (v2 rule). Hard filters: clean string, not overlapping any gold
    answer, not already in the gold passage, same coarse type (date/num/proper noun/common phrase).
    Preference tiers (first non-empty wins, then a seeded random choice):
      1 same Wikipedia article + same question wh-word + same surface form (possessive/plural)
      2 same article + same form
      3 same wh-word + same form + similar length (+-1 word)
      4 same wh-word + same form
      5 same wh-word
      6 type only
    If the gold has no leading article, a leading 'the/a/an' is stripped from the candidate.
    pool_answers: list of str or tuples (answer, wh_key[, title])."""
    main = min(ans, key=len)
    t, n, f = answer_type(main), len(main.split()), _form(main)
    wk = wh_key(question) if question else None
    gold_art = bool(ARTICLE.match(main))
    gl = gold_text.lower()
    ok = []
    for item in pool_answers:
        a, w, ti = (item, None, None) if isinstance(item, str) else (tuple(item) + (None, None))[:3]
        a = _clean(a)
        if a and not gold_art:
            a = _clean(ARTICLE.sub("", a))
        if not a or answer_type(a) != t or contains_any(a, ans) or contains_any(" ".join(ans), [a]):
            continue
        if gl and re.search(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", gl):
            continue
        ok.append((a, w, ti, _form(a) == f, abs(len(a.split()) - n) <= 1))
    tiers = [[a for a, w, ti, fm, ln in ok if ti == title and w == wk and fm],
             [a for a, w, ti, fm, ln in ok if ti == title and fm],
             [a for a, w, ti, fm, ln in ok if w == wk and fm and ln],
             [a for a, w, ti, fm, ln in ok if w == wk and fm],
             [a for a, w, ti, fm, ln in ok if w == wk],
             [a for a, w, ti, fm, ln in ok]]
    for k, c in enumerate(tiers, 1):
        if c:
            pick_fake.last_tier = k
            return rng.choice(sorted(set(c)))
    pick_fake.last_tier = None
    return None

def apply_swap(text, golds, fake):
    """Replace every whole-word occurrence of each gold answer (longest first) by `fake`, fixing a/an before it
    and capitalising it at a sentence start."""
    out = text
    for g in sorted(golds, key=len, reverse=True):
        if not g.strip():
            continue
        pat = re.compile(r"(\b(?:a|an)\s+)?(?<!\w)" + re.escape(g) + r"(?!\w)", re.I)
        def rep(m, s=None):
            art, f = m.group(1), fake
            before = out[:m.start()]
            if art:
                want = "an" if re.match(r"[aeiou]", f, re.I) else "a"
                art = (want.capitalize() if art[0].isupper() else want) + art[len(art.rstrip()):]
            elif (m.start() == 0 or re.search(r"[.!?]\s+$", before)) and f[:1].islower():
                f = f[0].upper() + f[1:]
            return (art or "") + f
        out = pat.sub(rep, out)
    return out

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

def build_rows(examples, pool, k=3, seed=0, answer_pool=None):
    """examples: [{qid, question, gold_answers, gold_pid}], pool: [{pid, title, text}].
    answer_pool: candidate replacement answers for C5 (defaults to the examples' own answers)."""
    rng = random.Random(seed)
    idx = {p["pid"]: i for i, p in enumerate(pool)}
    ret = Retriever([p["text"] for p in pool])
    v1_answers = [a for e in examples for a in e["gold_answers"]]      # v1 replacement pool (rng replay only)
    all_answers = answer_pool or v1_answers
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
        # Keep C1-C4 identical to benchmark v1: v1 drew its C5 replacement from the SAME rng that draws the C2
        # random distractors, so we replay that draw exactly (result discarded) to keep the rng stream unchanged.
        v1_fakes = [a for a in v1_answers if not contains_any(a, ans) and not contains_any(" ".join(ans), [a])]
        if len(hard) < k or len(rand) < k or not v1_fakes:
            dropped.append((e["qid"], "not enough clean distractors", {"question": e["question"], "n_hard": len(hard), "n_random": len(rand)})); continue
        rng.choice(v1_fakes)
        # v2 C5 replacement: own per-question rng stream, same wh-word + answer type (see pick_fake)
        fake = pick_fake(ans, all_answers, uid_rng(seed, e["qid"] + "__C5_replacement"), e["question"],
                         e.get("title"), gold["text"])
        tier = pick_fake.last_tier
        if not fake:
            dropped.append((e["qid"], "no replacement entity of matching type", {"question": e["question"], "gold_answers": ans})); continue
        cf = apply_swap(gold["text"], ans, fake)
        if contains_any(cf, ans) or fake.lower() not in cf.lower():
            left = [g for g in ans if g.strip() and g.lower() in cf.lower()]
            ctx = ""
            if left:
                i = cf.lower().index(left[0].lower()); ctx = cf[max(0, i - 60): i + len(left[0]) + 60]
            dropped.append((e["qid"], "counterfactual swap failed", {
                "question": e["question"], "gold_answers": ans, "replacement": fake,
                "why": (f"gold string {left[0]!r} still present after word-boundary swap (it occurs inside a longer word/phrase)"
                        if left else "replacement entity not found in swapped passage"),
                "context_after_swap": ctx}))
            continue
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
                "swapped_entity": fake if c == "C5_counterfactual" else None,
                **({"swap_tier": tier} if c == "C5_counterfactual" else {})})
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
        ex.append({"qid": r["id"], "question": r["question"], "title": r["title"],
                   "gold_answers": sorted(set(r["answers"]["text"])), "gold_pid": seen[r["context"]]})
        if len(ex) >= int(n * 1.3): break          # spare questions in case some get dropped
    answer_pool = sorted({(a, wh_key(r["question"]), r["title"]) for r in ds["validation"] for a in r["answers"]["text"]
                          if len(a.split()) <= max_ans_words})
    return ex, pool, answer_pool

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--out", default="artifacts/data")
    ap.add_argument("--version", default="v2", help="output file suffix (v1 = legacy builder in src/data/legacy)")
    a = ap.parse_args()
    ex, pool, answer_pool = load_squad_v2(a.n, a.seed)
    rows, dropped = build_rows(ex, pool, a.k, a.seed, answer_pool)
    keep = [r["qid"] for r in rows if r["condition"] == "C1_gold"][: a.n]
    ks = set(keep)
    rows = [r for r in rows if r["qid"] in ks]
    errs = validate(rows)
    name = "benchmark_mini_v0.jsonl" if a.mini else f"benchmark_{a.version}.jsonl"
    with open(f"{a.out}/{name}", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    if not a.mini:
        json.dump(make_folds(keep, 5, a.seed), open(f"{a.out}/folds_{a.version}.json", "w"), indent=1)
        json.dump(make_comparison_set(keep, 20, 1), open(f"{a.out}/comparison_set_{a.version}.json", "w"), indent=1)
    json.dump({"rows": len(rows), "qids": len(keep), "dropped": dropped, "errors": errs},
              open(f"{a.out}/validation_report_{'mini' if a.mini else a.version}.json", "w"), indent=1)
    print(f"rows={len(rows)} qids={len(keep)} dropped={len(dropped)} errors={len(errs)}")
