import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.prompt_templates import encode_prompt, PROMPT_VERSION

PIN_PATH = "artifacts/data/model_pin.json"
_CFG = "artifacts/data/experiment_config_v2.json"
DEFAULT_MAX_NEW_TOKENS = (json.load(open(_CFG))["generation"]["max_new_tokens"] if os.path.exists(_CFG) else 128)


def pinned_revision(model_id):
    """Exact Hugging Face commit to load for the model of record (artifacts/data/model_pin.json)."""
    if os.path.exists(PIN_PATH):
        pin = json.load(open(PIN_PATH))
        if pin["repo"] == model_id:
            return pin["revision"]
    return None


def compute_first_token_stats(scores_0):
    """Return entropy and max-prob for the first generated token distribution."""
    p0 = torch.softmax(scores_0[0].float(), dim=-1)
    entropy = float(-(p0 * torch.log(p0 + 1e-12)).sum())
    maxprob = float(p0.max())
    return entropy, maxprob


def main():
    ap = argparse.ArgumentParser(description="Resumable greedy generation for rag-noise-probe")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--bench", required=True, help="Path to benchmark .jsonl file")
    ap.add_argument("--template", choices=["std", "abstain"], default="std")
    ap.add_argument("--out", required=True, help="Output .jsonl path (appended)")
    ap.add_argument("--n", type=int, default=None, help="Max rows to process (for smoke tests)")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="FROZEN: 128 (artifacts/data/experiment_config_v2.json). Rows that do not end with EOS are "
                         "regenerated with 256 by src.gen.finalize_gen. (32 truncated ~45% of answers, 64 ~5%.)")
    ap.add_argument("--revision", default=None, help="HF commit; default = pinned revision in artifacts/data/model_pin.json")
    ap.add_argument("--uids", default=None, help="optional file with one uid per line (sample runs)")
    ap.add_argument("--checkpoint_every", type=int, default=50,
                    help="Flush and print progress every N rows")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    # --- load benchmark ---
    rows = [json.loads(l) for l in open(a.bench, encoding="utf-8")]
    if a.uids:
        keep = {l.strip() for l in open(a.uids) if l.strip()}
        rows = [r for r in rows if r["uid"] in keep]
    if a.n:
        rows = rows[: a.n]

    # --- skip already-done uids (resumable) ---
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["uid"])
    todo = [r for r in rows if r["uid"] not in done]
    print(f"[gen] {len(done)} already done, {len(todo)} remaining")

    if not todo:
        print("[gen] Nothing to do — all uids present in output file.")
        return

    # --- load model ---
    print(f"[gen] Loading tokenizer: {a.model}")
    rev = a.revision or pinned_revision(a.model)
    print(f"[gen] revision: {rev or 'latest (NOT pinned)'}")
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev)
    print(f"[gen] Loading model: {a.model}  (dtype=float16, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=rev, dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map=("auto" if torch.cuda.is_available() else None)
    ).eval()
    name = a.model.split("/")[-1]
    import transformers
    json.dump({"model": a.model, "revision": rev, "template": a.template, "prompt_version": PROMPT_VERSION,
               "bench": a.bench, "decoding": "greedy", "do_sample": False, "max_new_tokens": a.max_new_tokens,
               "dtype": str(next(model.parameters()).dtype), "eos_token_id": tok.eos_token_id,
               "transformers": transformers.__version__, "torch": torch.__version__},
              open(a.out.replace(".jsonl", "_manifest.json"), "w"), indent=1)
    print(f"[gen] Model loaded on: {next(model.parameters()).device}")

    # --- generation loop ---
    with open(a.out, "a", encoding="utf-8") as f:
        for idx, r in enumerate(todo):
            t0 = time.time()

            ids = encode_prompt(tok, r, a.template).to(next(model.parameters()).device)

            with torch.no_grad():
                o = model.generate(
                    ids,
                    max_new_tokens=a.max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tok.eos_token_id,
                    attention_mask=torch.ones_like(ids),
                    temperature=None, top_p=None,
                )

            # strip prompt tokens, then strip EOS and anything after
            new_ids = o.sequences[0, ids.shape[1]:].tolist()
            stopped_by_eos = tok.eos_token_id in new_ids      # False = hit max_new_tokens (truncated)
            if tok.eos_token_id in new_ids:
                new_ids = new_ids[: new_ids.index(tok.eos_token_id)]

            # per-token log-probabilities
            lps, ents = [], []
            for i, t in enumerate(new_ids):
                lp = torch.log_softmax(o.scores[i][0].float(), dim=-1)
                lps.append(float(lp[t]))
                ents.append(float(-(lp.exp() * lp).sum()))   # predictive entropy at step i

            # first-token distribution stats
            if o.scores:
                ent, mxp = compute_first_token_stats(o.scores)
            else:
                ent, mxp = 0.0, 0.0

            rec = {
                "uid":                  r["uid"],
                "model":                name,
                "template":             a.template,
                "prompt_version":       PROMPT_VERSION,
                "answer":               tok.decode(new_ids, skip_special_tokens=True).strip(),
                "n_answer_tokens":      len(new_ids),
                "answer_token_ids":     new_ids,
                "answer_token_logprobs": lps,
                "answer_token_entropies": ents,
                "first_token_entropy":  ent,
                "first_token_maxprob":  mxp,
                "prompt_len":           int(ids.shape[1]),
                "stopped_by_eos":       stopped_by_eos,
                "gen_seconds":          round(time.time() - t0, 3),
            }
            f.write(json.dumps(rec) + "\n")

            # checkpoint: flush every N rows and print progress
            if (idx + 1) % a.checkpoint_every == 0:
                f.flush()
                os.fsync(f.fileno())
                done_total = len(done) + idx + 1
                print(f"[gen] checkpoint  {done_total}/{len(rows)} rows  "
                      f"last_uid={r['uid']}  last_secs={rec['gen_seconds']}")

    print(f"[gen] Done. Output: {a.out}")


if __name__ == "__main__":
    main()
