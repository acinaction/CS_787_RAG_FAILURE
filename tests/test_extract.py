"""CPU tests for the ReDeEP extraction (tiny random Llama, no download).

Checks that the hook-captured residual is exactly the fork's `residual` (stream after attention,
before the FFN), that features have the right shapes / ranges, and that hs[-1] is post-final-norm
(the authors skip the extra norm on the last layer because of this)."""
import unittest
import numpy as np
import torch

try:
    from transformers import LlamaConfig, LlamaForCausalLM
    HAVE_TF = True
except Exception:
    HAVE_TF = False


@unittest.skipUnless(HAVE_TF, "transformers not installed")
class TestExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
        cfg._attn_implementation = "eager"
        cls.model = LlamaForCausalLM(cfg).eval()
        cls.L = 4

    def test_residual_hook_matches_fork_definition(self):
        from src.extract.redeep_extract import ResidualCatcher
        m = self.model
        c = ResidualCatcher(m.model.layers)
        ids = torch.randint(0, 128, (1, 20))
        with torch.no_grad():
            out = m(input_ids=ids, output_hidden_states=True)
        for l, layer in enumerate(m.model.layers):
            mid = c.store[l]
            # fork: hidden_out = residual + mlp(post_attention_layernorm(residual))
            recon = mid + layer.mlp(layer.post_attention_layernorm(mid))
            if l < self.L - 1:
                torch.testing.assert_close(recon, out.hidden_states[l + 1], atol=1e-5, rtol=1e-5)
            else:   # last hidden state is post-final-norm in HF
                torch.testing.assert_close(m.model.norm(recon), out.hidden_states[l + 1], atol=1e-5, rtol=1e-5)
        c.remove()

    def test_feature_shapes(self):
        from src.extract.redeep_extract import ResidualCatcher, features_for_row, probe_layer_indices
        m = self.model
        c = ResidualCatcher(m.model.layers)
        heads = [(0, 1), (2, 3), (3, 0)]
        ids = torch.randint(0, 128, (1, 30))
        with torch.no_grad():
            ans = m.generate(ids, max_new_tokens=5, do_sample=False, attention_mask=torch.ones_like(ids))[0, 30:].tolist()
        ecs, pks, probe, tfm = features_for_row(m, c, ids, ans, heads, probe_layer_indices(self.L))
        self.assertEqual(ecs.shape, (len(ans), 3))
        self.assertEqual(pks.shape, (len(ans), self.L))
        self.assertEqual(probe.shape, (3, 64))
        self.assertTrue(np.all(ecs <= 1.0001) and np.all(ecs >= -1.0001))
        self.assertTrue(np.all(pks >= 0))
        self.assertEqual(tfm, 1.0)    # greedy answer reproduced by teacher forcing
        e2, p2, pr2, t2 = features_for_row(m, c, ids, [], heads, probe_layer_indices(self.L))
        self.assertIsNone(e2); self.assertEqual(pr2.shape, (3, 64))
        c.remove()

    def test_jsd_matches_authors_formula(self):
        from src.extract.redeep_extract import jsd_authors
        import torch.nn.functional as F
        a, b = torch.randn(3, 50), torch.randn(3, 50)
        ours = jsd_authors(a, b)
        for i in range(3):   # authors' calculate_dist, verbatim
            sa, sb = F.softmax(a[i], -1), F.softmax(b[i], -1); M = 0.5 * (sa + sb)
            kl1 = F.kl_div(F.log_softmax(a[i], -1), M, reduction="none").mean(-1)
            kl2 = F.kl_div(F.log_softmax(b[i], -1), M, reduction="none").mean(-1)
            self.assertAlmostEqual(float(ours[i]), (0.5 * (kl1 + kl2)).item() * 10e5, places=2)


if __name__ == "__main__":
    unittest.main()
