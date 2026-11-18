"""Frozen decision (2026-11-07): Pearson ranking by |r| with the original sign retained; AUROC only as sensitivity."""
import unittest
import numpy as np
from src.detect.fit_redeep import rank_features, Fitted, grid_scores, score_with
from src.detect.common import auc


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < .4).astype(int)
    ecs = rng.normal(0, 1, (n, 5)); pks = rng.normal(0, 1, (n, 4))
    ecs[:, 0] -= 1.2 * y          # paper direction: ECS falls with failure, strong
    ecs[:, 1] += 0.8 * y          # opposite direction: ECS RISES with failure
    ecs[:, 2] -= 0.2 * y          # weak
    pks[:, 0] += 1.0 * y          # paper direction
    pks[:, 1] -= 1.5 * y          # opposite direction, strongest
    return ecs, pks, y


class TestSignedPearson(unittest.TestCase):
    def test_rank_by_abs_r_and_sign_kept(self):
        ecs, pks, y = toy()
        hr, lr, hs, ls = rank_features(ecs, pks, y, "pearson")
        self.assertEqual(list(hr[:2]), [0, 1])                      # ranked by |r|: strong head, then opposite-sign head
        self.assertEqual((hs[0], hs[1]), (-1.0, 1.0))               # original signs retained
        self.assertEqual(lr[0], 1); self.assertEqual(ls[0], -1.0)   # strongest PKS layer is failure-DEcreasing here
        self.assertEqual(ls[1], 1.0)

    def test_auroc_is_sensitivity_with_paper_signs(self):
        ecs, pks, y = toy()
        hr, lr, hs, ls = rank_features(ecs, pks, y, "auroc")
        self.assertTrue(np.all(hs == -1) and np.all(ls == 1))       # fixed paper directions
        self.assertEqual(hr[0], 0); self.assertEqual(hr[-1], 1)     # authors' AUROC(ECS, 1-y) puts the flipped head last

    def test_oriented_score_ranks_like_paper_formula_when_signs_match(self):
        ecs, pks, y = toy()
        e, p = ecs[:, [0, 2]], pks[:, [0, 3]]                       # only paper-direction features
        f = Fitted(e, p, y, "pearson")
        self.assertTrue(np.all(f.h_sign == -1) and f.l_sign[0] == 1)
        En, Pn = f.norm_cum(e, p)
        raw = e[:, f.h_rank[0]]
        e_n = (raw - raw.min()) / (raw.max() - raw.min())           # paper's min-max ECS
        for b in (0.2, 1.0, 2.0):
            self.assertAlmostEqual(np.corrcoef(Pn[:, 0] + b * En[:, 0], Pn[:, 0] - b * e_n)[0, 1], 1.0, places=6)

    def test_retained_sign_uses_flipped_features_correctly(self):
        ecs, pks, y = toy()
        f = Fitted(ecs, pks, y, "pearson")
        En, Pn = f.norm_cum(ecs, pks)
        self.assertGreater(auc(En[:, 1], y), 0.75)                  # top-2 oriented heads (one has positive sign) separate failures
        f2 = Fitted(ecs, pks, y, "auroc")
        En2, _ = f2.norm_cum(ecs, pks)
        self.assertGreater(auc(En[:, 4], y), auc(En2[:, 4], y) - 1e-9)   # all 5 heads: oriented >= unoriented
        self.assertGreater(auc(score_with((2, 2, 1.0), En, Pn, "full"), y), 0.75)   # higher score = more failure

    def test_grid_shapes(self):
        ecs, pks, y = toy(600)
        f = Fitted(ecs, pks, y, "pearson"); En, Pn = f.norm_cum(ecs, pks)
        S, cfg = grid_scores(En, Pn, "full")
        self.assertEqual(S.shape[0], len(cfg)); self.assertEqual(S.shape[1], 600)


if __name__ == "__main__":
    unittest.main()
