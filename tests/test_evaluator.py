"""Label rules (EXECUTION_PLAN 3.4): correct = answered and (gold contained on token boundaries or max-F1 >= 0.5)."""
import unittest
from src.data.evaluator import label_row, is_correct, is_correct_legacy, best_f1

B = lambda cond, golds, sw=None: {"qid": "q", "condition": cond, "gold_answers": golds, "swapped_entity": sw}

class TestEvaluator(unittest.TestCase):
    def test_boundary_fixes_substring_false_positive(self):
        self.assertFalse(is_correct("It started in 1991.", ["91"]))
        self.assertTrue(is_correct_legacy("It started in 1991.", ["91"]))                # Day-1 rule was wrong here
        self.assertFalse(is_correct("an artistic movement", ["art"]))
        self.assertTrue(is_correct("twenty-two counties", ["two"]))                     # whole token after normalisation
        self.assertFalse(is_correct("City10", ["City1"]))
    def test_containment_and_f1_paths(self):
        self.assertTrue(is_correct("Sure! Based on Passage 1, the capital is City1.", ["City1"]))
        self.assertTrue(is_correct("the main hall", ["main hall"]))                      # containment
        self.assertGreaterEqual(best_f1("Galilean transformation rule", ["Galilean transformation"]), 0.5)
    def test_c4_and_c5_logic_preserved(self):
        r = label_row({"answer": "City3"}, B("C4_missing", ["City1"]))
        self.assertEqual(r["failure"], 1)                                              # answered in C4 = failure
        r = label_row({"answer": "I don't know."}, B("C4_missing", ["City1"]))
        self.assertEqual((r["refusal"], r["failure"]), (True, 0))
        r = label_row({"answer": "The capital is City5."}, B("C5_counterfactual", ["City1"], "City5"))
        self.assertEqual((r["correct"], r["failure"], r["followed_counterfactual"]), (0, 1, True))
        r = label_row({"answer": "City1"}, B("C5_counterfactual", ["City1"], "City5"))
        self.assertEqual((r["failure"], r["followed_counterfactual"]), (0, False))
    def test_f1_field_kept(self):
        r = label_row({"answer": "the capital is City1"}, B("C1_gold", ["City1"]))
        self.assertIn("f1", r); self.assertAlmostEqual(r["f1"], 2 * (1 / 4) * 1 / (1 / 4 + 1), places=3)

if __name__ == "__main__":
    unittest.main()
