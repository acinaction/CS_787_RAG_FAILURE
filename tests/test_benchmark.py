import unittest, random
from src.data.noise_builder import build_rows, validate, make_folds, make_comparison_set, CONDS
from src.data.prompt_templates import build_user

def synth(n_pool=80, n_q=15):
    pool = [{"pid": f"p{i}", "title": f"T{i}",
             "text": f"The capital of Country{i} is City{i}. Country{i} has river River{i} and mountain Mount{i}. "
                     f"Common words about geography history trade and culture appear here."} for i in range(n_pool)]
    ex = [{"qid": f"q{i}", "question": f"What is the capital of Country{i}?",
           "gold_answers": [f"City{i}"], "gold_pid": f"p{i}"} for i in range(n_q)]
    return ex, pool

class T(unittest.TestCase):
    def setUp(self):
        self.ex, self.pool = synth()
        self.rows, self.dropped = build_rows(self.ex, self.pool, k=3, seed=0)
    def test_valid(self):            self.assertEqual(validate(self.rows), [])
    def test_all_conditions(self):   self.assertEqual(len(self.rows), 5 * (15 - len(self.dropped)))
    def test_c4_no_gold(self):
        for r in self.rows:
            if r["condition"] == "C4_missing":
                self.assertFalse(any(p["role"] == "gold" for p in r["passages"]))
                self.assertFalse(r["answerable"])
    def test_c5_swap(self):
        for r in self.rows:
            if r["condition"] == "C5_counterfactual":
                cf = [p for p in r["passages"] if p["role"] == "counterfactual"][0]
                self.assertIn(r["swapped_entity"].lower(), cf["text"].lower())
                self.assertNotIn(r["gold_answers"][0].lower(), cf["text"].lower())
    def test_unique_uids(self):      self.assertEqual(len({r["uid"] for r in self.rows}), len(self.rows))
    def test_folds_group(self):
        f = make_folds([e["qid"] for e in self.ex]); self.assertEqual(set(f.values()), {0, 1, 2, 3, 4})
    def test_comparison_set(self):
        q = [e["qid"] for e in self.ex]
        c = make_comparison_set(q, n=5, seed=1)
        self.assertEqual(len(c["qids"]), 5); self.assertEqual(c, make_comparison_set(q, n=5, seed=1))
    def test_prompt_keeps_question(self):
        r = dict(self.rows[0]); r["passages"] = [{"text": "x" * 20000, "role": "gold"}]
        u = build_user(r)
        self.assertTrue(u.endswith("Question: " + r["question"])); self.assertLessEqual(len(u), 12000)
    def test_validator_catches_leak(self):
        bad = [dict(r) for r in self.rows]
        for r in bad:
            if r["condition"] == "C2_random":
                r["passages"] = [dict(p) for p in r["passages"]]
                [p for p in r["passages"] if p["role"] == "random"][0]["text"] += " " + r["gold_answers"][0]; break
        self.assertTrue(validate(bad))
if __name__ == "__main__": unittest.main()
