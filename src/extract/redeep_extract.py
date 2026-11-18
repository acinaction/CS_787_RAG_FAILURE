"""P1: ReDeEP token-level feature extraction (ECS + PKS) + pre-generation hidden-state probe.

This re-implements, line for line, the feature computation of the authors'
`third_party/ReDEeP-ICLR/ReDeEP/token_level_detect.py` (commit 4d08191), but on STOCK transformers:
the authors' forked `modeling_llama.py` only adds `knowledge_layers=` to expose the residual stream
between attention and FFN; we capture the same tensor with a forward pre-hook on each layer's
`post_attention_layernorm` (its input is exactly the fork's `residual` after attention).
=> Report this as "authors' feature definitions, re-implemented with hooks" (adapted, not the fork).

Definitions (one teacher-forced forward pass over prompt + generated answer; P = prompt length,
positions t = P-1 ... T-2 are the positions that PREDICT each answer token, as in the authors' loop):

  ECS (external context score), per copy head (l, h) from topk_heads.json, in sorted (layer, head) order:
      a      = attention of head (l,h) at position t over the prompt tokens [0, P)
      idx    = top-10% prompt positions by a            (k = int(0.1 * P))
      ECS    = cos( mean_{i in idx} H_last[i],  H_last[t] )      H_last = final (normed) hidden states

  PKS (parametric knowledge score), per layer l = 0..L-1:
      p_post = softmax(lm_head(norm(h_{l+1}[t])))        (layer L-1: h_L is already normed -> no extra norm)
      p_mid  = softmax(lm_head(norm(r_l[t])))            r_l = residual after attention, before FFN of layer l
      PKS    = JSD_authors(p_post, p_mid) * 10e5
    where JSD_authors uses F.kl_div(log p, M, reduction='none').mean(-1) (mean over vocab), as shipped.

Probe: hidden_states[i][0, P-1] (last prompt token) for i in {floor(L/3), floor(L/2), floor(3L/4)}.
Sanity: `tf_match` = fraction of answer tokens equal to the argmax at their position (should be ~1.0
for greedy answers; < 1 flags a prompt/tokenisation mismatch between gen and extraction).

Outputs
  redeep_feats.parquet : uid, n_tokens, tf_match, ecs_tok [n_tok x n_heads], pks_tok [n_tok x L],
                         ecs_mean, ecs_first, pks_mean, pks_first
  redeep_meta.json     : head_list, n_layers, probe_layers, repo_commit, ...
  probe_feats.npy      : [N, 3, hidden] float16, row order = probe_index.json["uids"]
  redeep_tokens_cmp.json (optional, --cmp_set): per-token ECS/PKS + token strings for the comparison rows

Resumable: progress is appended to <out_feats>.parts/ (one .pkl per checkpoint); final files are
rebuilt from all parts at the end, so a Kaggle disconnect loses at most one checkpoint.

Usage (Kaggle, repo root):
  python -m src.extract.redeep_extract --model meta-llama/Llama-2-7b-chat-hf \
      --gen artifacts/runs/llama2-7b/gen_std.jsonl --bench artifacts/data/benchmark_v2.jsonl \
      --outdir artifacts/runs/llama2-7b --cmp_set artifacts/data/comparison_set_v2.json
"""
import argparse, glob, json, math, os, pickle, subprocess, time
from datetime import date
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.prompt_templates import encode_prompt

DEFAULT_TOPK_HEADS = "third_party/ReDEeP-ICLR/ReDeEP/log/test_llama2_7B/topk_heads.json"
TOP_RATE = 0.1          # authors: top 10% attended prompt tokens
JSD_SCALE = 10e5        # authors: js_divs * 10e5


def probe_layer_indices(n_layers):
    return [math.floor(n_layers / 3), math.floor(n_layers / 2), math.floor(3 * n_layers / 4)]


def jsd_authors(logits_a, logits_b):
    """Exact copy of the authors' calculate_dist, vectorised over rows. logits_*: [n, V] -> [n]."""
    pa, pb = F.softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1)
    M = 0.5 * (pa + pb)
    kl1 = F.kl_div(F.log_softmax(logits_a, dim=-1), M, reduction="none").mean(-1)
    kl2 = F.kl_div(F.log_softmax(logits_b, dim=-1), M, reduction="none").mean(-1)
    return 0.5 * (kl1 + kl2) * JSD_SCALE


def load_head_list(path, n_layers, n_heads):
    if path and os.path.exists(path):
        heads = [tuple(h) for h in json.load(open(path))]
        if all(l < n_layers and h < n_heads for l, h in heads):
            return sorted(heads), path      # authors iterate layers then heads -> sorted order
        print(f"[extract] WARNING: {path} does not fit this model ({n_layers}x{n_heads}); using fallback heads")
    # Fallback (smoke tests on other models only): one head per layer, spread evenly. NOT ReDeEP's heads.
    return sorted({(l, (7 * l) % n_heads) for l in range(n_layers)}), "fallback"


class ResidualCatcher:
    """Pre-hook on post_attention_layernorm: its input is the residual stream after attention (fork's `residual`)."""
    def __init__(self, layers):
        self.store, self.handles = {}, []
        for i, layer in enumerate(layers):
            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self._mk(i)))
    def _mk(self, i):
        def hook(mod, args):
            self.store[i] = args[0].detach()
        return hook
    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def features_for_row(model, catcher, prompt_ids, ans_ids, head_list, probe_layers):
    dev = prompt_ids.device
    P = prompt_ids.shape[1]
    full = torch.cat([prompt_ids, torch.tensor([ans_ids], device=dev, dtype=prompt_ids.dtype)], dim=1) if ans_ids else prompt_ids
    out = model(input_ids=full, output_attentions=bool(ans_ids), output_hidden_states=True, return_dict=True)
    hs = out.hidden_states                                  # L+1 tensors [1, T, d]; hs[-1] is post-final-norm
    L = len(hs) - 1
    probe = np.stack([hs[i][0, P - 1].float().cpu().numpy() for i in probe_layers]).astype(np.float16)
    # NOTE (multi-GPU, device_map="auto"): layers/attentions/hidden states live on different GPUs, so every
    # index tensor must be moved to the device of the tensor it indexes -> use `at(t)` below.
    if not ans_ids:
        return None, None, probe, float("nan")

    pos = torch.arange(P - 1, full.shape[1] - 1, device="cpu")        # predicts answer token t+1
    at = lambda t: pos.to(t.device)
    lm_head, norm = model.lm_head, model.model.norm
    lm_dev = lm_head.weight.device
    last_dev = hs[-1].device

    # ---- PKS -------------------------------------------------------------------------
    pks = []
    for l in range(L):
        post = hs[l + 1][0][at(hs[l + 1])].to(lm_dev)
        post = post if l == L - 1 else norm(post)
        mid = norm(catcher.store[l][0][at(catcher.store[l])].to(lm_dev))
        pks.append(jsd_authors(lm_head(post).float(), lm_head(mid).float()))
    pks = torch.stack([x.to(lm_dev) for x in pks], dim=1).cpu().numpy()   # [n_tok, L]

    # ---- ECS -------------------------------------------------------------------------
    last = hs[-1][0].float()                                # [T, d]
    k = max(1, int(P * TOP_RATE))
    ecs = []
    for (l, h) in head_list:
        att = out.attentions[l]
        a = att[0, h][at(att)][:, :P].float()               # [n_tok, P]
        idx = torch.topk(a, k, dim=-1).indices.to(last.device)   # [n_tok, k]
        attended = last[idx].mean(dim=1)                    # [n_tok, d]
        ecs.append(F.cosine_similarity(attended, last[at(last)], dim=-1))
    ecs = torch.stack([e.to(last_dev) for e in ecs], dim=1).cpu().numpy()             # [n_tok, n_heads]

    pred = out.logits[0][at(out.logits)].argmax(-1).cpu().tolist()
    tf_match = float(np.mean([p == t for p, t in zip(pred, ans_ids)]))
    return ecs.astype(np.float32), pks.astype(np.float32), probe, tf_match


def collect_parts(parts_dir):
    recs, probes = [], []
    for f in sorted(glob.glob(os.path.join(parts_dir, "part_*.pkl"))):
        r, p = pickle.load(open(f, "rb"))
        recs += r; probes += p
    return recs, probes


def main():
    ap = argparse.ArgumentParser(description="ReDeEP token-level feature extraction")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--outdir", default="artifacts/runs/llama2-7b")
    ap.add_argument("--topk_heads", default=DEFAULT_TOPK_HEADS)
    ap.add_argument("--cmp_set", default=None, help="comparison_set_v2.json -> also write redeep_tokens_cmp.json")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--checkpoint_every", type=int, default=25)
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    f_feats = os.path.join(a.outdir, "redeep_feats.parquet")
    parts_dir = f_feats + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    bench = {}
    for l in open(a.bench, encoding="utf-8"):
        r = json.loads(l); bench[r["uid"]] = r
    gen_rows = [json.loads(l) for l in open(a.gen, encoding="utf-8")]
    if a.n:
        gen_rows = gen_rows[: a.n]
    done = {r["uid"] for r in collect_parts(parts_dir)[0]}
    todo = [g for g in gen_rows if g["uid"] not in done]
    print(f"[extract] {len(done)} done, {len(todo)} to do")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    cuda = torch.cuda.is_available()
    from src.gen.gen import pinned_revision
    rev = a.revision or pinned_revision(a.model)
    print(f"[extract] revision: {rev or 'latest (NOT pinned)'}")
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=rev, dtype=(torch.float16 if cuda else torch.float32),
        device_map=("auto" if cuda else None), attn_implementation="eager").eval()
    cfg = model.config
    L, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_list, head_src = load_head_list(a.topk_heads, L, n_heads)
    probe_layers = probe_layer_indices(L)
    dev = model.get_input_embeddings().weight.device
    catcher = ResidualCatcher(model.model.layers)

    try:
        sha = subprocess.check_output(["git", "-C", "third_party/ReDEeP-ICLR", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        cf = "third_party/ReDEeP-ICLR/COMMIT"
        sha = open(cf).read().strip() if os.path.exists(cf) else "unknown"
    meta = {"model": a.model.split("/")[-1], "revision": rev, "head_list": [list(h) for h in head_list], "head_source": head_src,
            "n_layers": L, "n_heads": n_heads, "pks_layers": list(range(L)), "probe_layers": probe_layers,
            "top_rate": TOP_RATE, "jsd_scale": JSD_SCALE, "repo_commit": sha,
            "transformers": transformers.__version__, "implementation": "authors' definitions, stock transformers + hooks",
            "extraction_date": str(date.today())}
    json.dump(meta, open(os.path.join(a.outdir, "redeep_meta.json"), "w"), indent=2)
    print(f"[extract] L={L} heads={len(head_list)} ({head_src}) probe_layers={probe_layers}")

    buf_r, buf_p, t0 = [], [], time.time()
    part_id = len(glob.glob(os.path.join(parts_dir, "part_*.pkl")))

    def flush():
        nonlocal buf_r, buf_p, part_id
        if buf_r:
            pickle.dump((buf_r, buf_p), open(os.path.join(parts_dir, f"part_{part_id:05d}.pkl"), "wb"))
            part_id += 1
            buf_r, buf_p = [], []

    for i, g in enumerate(todo):
        b = bench[g["uid"]]
        ids = encode_prompt(tok, b, g["template"]).to(dev)
        if ids.shape[1] != g.get("prompt_len", ids.shape[1]):
            print(f"[extract] WARNING {g['uid']}: prompt_len {ids.shape[1]} != gen {g['prompt_len']}")
        ans = list(g["answer_token_ids"])
        ecs, pks, probe, tfm = features_for_row(model, catcher, ids, ans, head_list, probe_layers)
        rec = {"uid": g["uid"], "n_tokens": len(ans), "tf_match": tfm}
        if ecs is None:   # empty answer: no answer positions -> NaN features (dropped by detectors)
            rec.update(ecs_tok=[], pks_tok=[], ecs_mean=[np.nan] * len(head_list), ecs_first=[np.nan] * len(head_list),
                       pks_mean=[np.nan] * L, pks_first=[np.nan] * L)
        else:
            rec.update(ecs_tok=ecs.tolist(), pks_tok=pks.tolist(), ecs_mean=ecs.mean(0).tolist(),
                       ecs_first=ecs[0].tolist(), pks_mean=pks.mean(0).tolist(), pks_first=pks[0].tolist())
            if not (np.isfinite(ecs).all() and np.isfinite(pks).all()):
                print(f"[extract] WARNING {g['uid']}: non-finite features")
        buf_r.append(rec); buf_p.append((g["uid"], probe))
        if cuda:
            torch.cuda.empty_cache()
        if (i + 1) % a.checkpoint_every == 0:
            flush()
            el = time.time() - t0
            print(f"[extract] {len(done) + i + 1}/{len(gen_rows)}  {el / (i + 1):.2f}s/row  tf_match(last)={tfm:.2f}")
    flush()
    catcher.remove()

    # ---- rebuild final files from all parts -------------------------------------------
    recs, probes = collect_parts(parts_dir)
    order = {g["uid"]: k for k, g in enumerate(gen_rows)}
    keep = {}
    for r in recs:
        keep[r["uid"]] = r                      # last write wins
    recs = sorted(keep.values(), key=lambda r: order.get(r["uid"], 1e9))
    pmap = dict(probes)
    pd.DataFrame(recs).to_parquet(f_feats, index=False)
    uids = [r["uid"] for r in recs]
    np.save(os.path.join(a.outdir, "probe_feats.npy"), np.stack([pmap[u] for u in uids]).astype(np.float16))
    json.dump({"uids": uids, "layers": probe_layers}, open(os.path.join(a.outdir, "probe_index.json"), "w"))
    tfm = np.array([r["tf_match"] for r in recs], dtype=float)
    print(f"[extract] wrote {len(recs)} rows -> {f_feats}; mean tf_match={np.nanmean(tfm):.3f}")

    if a.cmp_set:
        cmp_q = set(json.load(open(a.cmp_set))["qids"])
        gmap = {g["uid"]: g for g in gen_rows}
        dump = {}
        for r in recs:
            if r["uid"].rsplit("__", 1)[0] in cmp_q:
                ids = gmap[r["uid"]]["answer_token_ids"]
                dump[r["uid"]] = {"tokens": tok.convert_ids_to_tokens(ids), "ecs_tok": r["ecs_tok"], "pks_tok": r["pks_tok"]}
        json.dump({"head_list": meta["head_list"], "rows": dump},
                  open(os.path.join(a.outdir, "redeep_tokens_cmp.json"), "w"))
        print(f"[extract] token dump for {len(dump)} comparison rows")


if __name__ == "__main__":
    main()
