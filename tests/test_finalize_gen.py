"""Merge / validate logic of selective 256-token regeneration (no GPU, no model)."""
import unittest
from src.gen.finalize_gen import merge, validate, truncated, REQUIRED


def row(uid, ids, eos, t="std"):
    return {"uid": uid, "model": "m", "template": t, "prompt_version": "prompt_v1", "answer": " ".join(map(str, ids)),
            "n_answer_tokens": len(ids), "answer_token_ids": ids, "answer_token_logprobs": [-.1] * len(ids),
            "answer_token_entropies": [.5] * len(ids), "first_token_entropy": 1.0, "first_token_maxprob": .5,
            "prompt_len": 100, "gen_seconds": 1.0, "stopped_by_eos": eos}


class TestFinalize(unittest.TestCase):
    def setUp(self):
        self.base = [row("a", list(range(5)), True), row("b", list(range(128)), False), row("c", list(range(7)), True),
                     row("d", list(range(128)), False)]
        self.regen = [row("b", list(range(200)), True), row("d", list(range(10)) + [99] * 246, False)]

    def test_only_truncated_rows_replaced(self):
        out, st = merge(self.base, self.regen)
        by = {r["uid"]: r for r in out}
        self.assertEqual(by["a"], dict(self.base[0], regenerated=False)); self.assertEqual(by["c"]["regenerated"], False)
        self.assertEqual(by["b"]["n_answer_tokens"], 200); self.assertTrue(by["b"]["regenerated"])
        self.assertEqual(by["b"]["first_pass_tokens"], 128)
        self.assertEqual(st["prefix_match"], 1); self.assertEqual(st["prefix_MISMATCH"], 1)       # d differs after token 10
        self.assertEqual(st["still_truncated_at_regen_limit"], 1)

    def test_validation_flags_remaining_truncation_and_set_errors(self):
        out, _ = merge(self.base, self.regen)
        errs, rep = validate(out, ["a", "b", "c", "d"], "std")
        self.assertTrue(any("still truncated" in e for e in errs)); self.assertEqual(rep["still_truncated"], 1)
        errs, _ = validate(out[:3], ["a", "b", "c", "d"], "std"); self.assertTrue(any("uid set" in e for e in errs))
        errs, _ = validate(out + [out[0]], ["a", "b", "c", "d"], "std"); self.assertTrue(any("duplicate" in e for e in errs))
        bad = [dict(r) for r in out]; del bad[0]["stopped_by_eos"]
        errs, _ = validate(bad, ["a", "b", "c", "d"], "std"); self.assertTrue(any("missing fields" in e for e in errs))

    def test_clean_file_passes(self):
        regen = [row("b", list(range(200)), True), row("d", list(range(128)) + [7] * 20, True)]
        out, _ = merge(self.base, regen)
        errs, rep = validate(out, ["a", "b", "c", "d"], "std")
        self.assertEqual(errs, []); self.assertEqual(rep["regenerated_rows"], 2); self.assertEqual(truncated(out), [])

    def test_tier3_merge_replaces_still_truncated(self):
        out, _ = merge(self.base, self.regen)                       # d is still truncated at 256
        tier3 = [row("d", list(range(10)) + [99] * 246 + [5] * 100, True)]
        out2, st = merge(out, tier3)
        by = {r["uid"]: r for r in out2}
        self.assertEqual(by["d"]["n_answer_tokens"], 356); self.assertEqual(by["d"]["first_pass_tokens"], 128)
        self.assertEqual(st["prefix_match"], 1); self.assertEqual(truncated(out2), [])
        self.assertEqual(validate(out2, ["a", "b", "c", "d"], "std")[0], [])


if __name__ == "__main__":
    unittest.main()
