"""Pre-run gate (run on Kaggle, needs HF access): is the model we will load EXACTLY the pinned Meta release,
and does prompt_v1 render through its chat template as expected?

  python -m src.gen.model_check --bench artifacts/data/benchmark_v2.jsonl --out artifacts/runs/llama2-7b/model_check.json [--hash_weights]

Checks (all must pass, exit code 1 otherwise):
  1 every small file (config, generation_config, tokenizer*, special_tokens_map, index) at the pinned revision has
    the git-blob SHA-1 recorded in artifacts/data/model_pin.json; tokenizer.model also by SHA-256
  2 (--hash_weights) both safetensors shards match the recorded LFS SHA-256 (~1-2 min on Kaggle)
  3 the tokenizer has a chat template; its SHA-256 is recorded
  4 one std and one abstain prompt render with the expected Llama-2 prefix/suffix, contain the system message,
    instruction text and question verbatim, and start with exactly one BOS token
"""
import argparse, hashlib, json, os, sys
from src.data.prompt_templates import build_messages, encode_prompt, STD, ABSTAIN_EXTRA, SYSTEM, PROMPT_VERSION

PIN = "artifacts/data/model_pin.json"


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--hash_weights", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    pin = json.load(open(PIN)); repo, rev = pin["repo"], pin["revision"]
    res, ok = {"repo": repo, "revision": rev, "files": {}}, True
    for fn, exp in pin["files"].items():
        big = fn.endswith(".safetensors")
        if big and not a.hash_weights:
            continue
        p = hf_hub_download(repo, fn, revision=rev)
        got = sha256(p) if exp["lfs_sha256"] else git_blob_sha1(p)
        want = exp["lfs_sha256"] or exp["git_blob_sha1"]
        res["files"][fn] = {"match": got == want, "got": got}
        ok &= got == want
        print(f"  {'OK ' if got == want else 'BAD'} {fn}")

    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    tmpl = tok.chat_template or ""
    res["chat_template_sha256"] = hashlib.sha256(tmpl.encode()).hexdigest()
    res["has_chat_template"] = bool(tmpl); ok &= bool(tmpl)
    row = json.loads(open(a.bench, encoding="utf-8").readline())
    for t in ("std", "abstain"):
        text = tok.apply_chat_template(build_messages(row, t), tokenize=False, add_generation_prompt=True)
        ids = encode_prompt(tok, row, t)[0].tolist()
        checks = {"prefix": text.startswith(pin["expected_prompt_prefix"]), "suffix": text.endswith(pin["expected_prompt_suffix"]),
                  "system": SYSTEM in text, "instruction": (STD + (ABSTAIN_EXTRA if t == "abstain" else "")) in text,
                  "question": row["question"] in text, "single_bos": ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id}
        res[f"render_{t}"] = {**checks, "n_tokens": len(ids), "rendered_head": text[:160]}
        ok &= all(checks.values())
        print(f"  render {t}: {checks}")
    res["prompt_version"] = PROMPT_VERSION
    res["all_ok"] = bool(ok)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("MODEL CHECK", "PASSED" if ok else "FAILED", "->", a.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
