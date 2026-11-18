"""Guards against silent changes to the frozen experimental setup (artifacts/data/FROZEN_v2.json)."""
import hashlib, json, os, unittest
import src.data.prompt_templates as pt

D = "artifacts/data"


@unittest.skipUnless(os.path.exists(f"{D}/FROZEN_v2.json"), "not frozen yet")
class TestFrozen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = json.load(open(f"{D}/FROZEN_v2.json"))

    def test_files_unchanged(self):
        for name, want in self.f["files"].items():
            self.assertEqual(hashlib.sha256(open(f"{D}/{name}", "rb").read()).hexdigest(), want, f"{name} changed after freeze")

    def test_prompt_v1_unchanged(self):
        c = json.dumps({k: getattr(pt, k) for k in ["PROMPT_VERSION", "SYSTEM", "STD", "ABSTAIN_EXTRA", "MAX_CHARS"]}, sort_keys=True)
        self.assertEqual(hashlib.sha256(c.encode()).hexdigest(), self.f["prompt_v1_constants_sha256"])

    def test_benchmark_and_fold_integrity(self):
        rows = [json.loads(l) for l in open(f"{D}/benchmark_v2.jsonl", encoding="utf-8")]
        folds = json.load(open(f"{D}/folds_v2.json")); cmp_q = json.load(open(f"{D}/comparison_set_v2.json"))["qids"]
        qids = {r["qid"] for r in rows}
        self.assertEqual(len(rows), 750); self.assertEqual(len(qids), 150)
        by = {}
        for r in rows:
            by.setdefault(r["qid"], set()).add(r["condition"])
        self.assertTrue(all(len(c) == 5 for c in by.values()))
        self.assertEqual(set(folds), qids)                         # one fold per qid => all 5 conditions share a fold
        self.assertEqual(sorted(__import__("collections").Counter(folds.values()).values()), [30] * 5)
        self.assertEqual(len(cmp_q), 20); self.assertTrue(set(cmp_q) <= qids)


class TestConfigMatchesCode(unittest.TestCase):
    """The frozen decisions in experiment_config_v2.json must be what the code actually does by default."""
    @classmethod
    def setUpClass(cls):
        cls.c = json.load(open(f"{D}/experiment_config_v2.json"))

    def test_generation_limits(self):
        from src.gen import gen, finalize_gen
        self.assertEqual(gen.DEFAULT_MAX_NEW_TOKENS, 128)
        self.assertEqual(self.c["generation"]["max_new_tokens"], 128)
        self.assertEqual(finalize_gen.FIRST, 128); self.assertEqual(finalize_gen.REGEN, 256)
        self.assertEqual(finalize_gen.REGEN2, 512); self.assertEqual(finalize_gen.ALLOWED_TRUNC, 0)

    def test_model_and_ranking(self):
        import src.detect.fit_redeep as fr
        pin = json.load(open(f"{D}/model_pin.json"))
        self.assertEqual((self.c["model"], self.c["revision"]), (pin["repo"], pin["revision"]))
        self.assertEqual(self.c["head_layer_selection"]["primary"], "pearson")
        self.assertEqual(fr.RANK_METHOD, "pearson")                  # module default = frozen primary
        self.assertEqual(self.c["head_layer_selection"]["sensitivity_only"], "auroc")
        self.assertEqual(self.c["benchmark"], "benchmark_v2.jsonl")


if __name__ == "__main__":
    unittest.main()
