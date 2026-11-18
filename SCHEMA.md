# Schema Reference

All file schemas are **frozen** after Day 2. Changes require a version bump and team notification.

---

## benchmark_v1.jsonl  (one JSON object per line)

| Field | Type | Notes |
|-------|------|-------|
| uid | str | `{qid}__{condition}` |
| qid | str | Unique question ID |
| condition | str | One of CONDS list |
| question | str | Question text |
| gold_answers | list[str] | Accepted correct answer strings |
| answerable | bool | False only for C4_missing |
| passages | list[dict] | Each: `{pid, text, role}` |
| n_distractors | int | Count of non-gold passages |
| gold_position | str | `first / middle / last / none` |
| bm25_top1 | float | BM25 score of highest-scoring passage shown |
| bm25_margin | float | top1 − top2 BM25 score |
| swapped_entity | str\|null | Replacement entity (C5 only) |

---

## folds_v1.json

```json
{ "<qid>": <fold_int_0_to_4>, ... }
```

---

## comparison_set_v1.json

```json
{ "seed": 1, "qids": ["<qid>", ...] }   // exactly 20 qids
```

---

## gen_std.jsonl / gen_abstain.jsonl  (one JSON object per line)

| Field | Type | Notes |
|-------|------|-------|
| uid | str | Matches benchmark uid |
| model | str | Model short name (e.g. `Llama-2-7b-chat-hf`) |
| template | str | `std` or `abstain` |
| prompt_version | str | e.g. `prompt_v1` |
| answer | str | Decoded generated text |
| n_answer_tokens | int | Token count (after EOS strip) |
| answer_token_ids | list[int] | |
| answer_token_logprobs | list[float] | Per-token log-probability |
| answer_token_entropies | list[float] | Per-token predictive entropy (added 2026-11-07, for LN-entropy) |
| stopped_by_eos | bool | False = hit max_new_tokens (truncated) |
| regenerated / first_pass_tokens | bool / int | present after `finalize_gen`: row was regenerated at 256 after being truncated at 128 |
| first_token_entropy | float | Shannon entropy of first token distribution |
| first_token_maxprob | float | Max probability of first token |
| prompt_len | int | Input token count |
| gen_seconds | float | Wall time for this row |

---

## labeled_std.jsonl / labeled_abstain.jsonl

All gen fields **plus** `qid`, `condition` and:

| Field | Type | Notes |
|-------|------|-------|
| refusal | bool | Matched refusal regex |
| answered | bool | Not a refusal |
| f1 | float | Token F1 against gold |
| correct | int | 0/1 |
| failure | int | 0/1 (see label definitions) |
| followed_counterfactual | bool | C5 only |

---

## redeep_feats.parquet  (v1, 2026-11-07)

| Column | Type | Notes |
|--------|------|-------|
| uid | str | |
| n_tokens | int | answer tokens (0 = empty answer, features NaN) |
| tf_match | float | fraction of answer tokens reproduced by teacher forcing (should be ~1) |
| ecs_tok | list[list[float]] | [n_tokens][n_copy_heads], heads in `redeep_meta.head_list` order |
| pks_tok | list[list[float]] | [n_tokens][n_layers] |
| ecs_mean / ecs_first | list[float] | mean over tokens / first token |
| pks_mean / pks_first | list[float] | |

---

## redeep_meta.json

`model, head_list [[layer, head], ...] (sorted), head_source, n_layers, n_heads, pks_layers, probe_layers,
top_rate (0.1), jsd_scale (10e5), repo_commit, transformers, implementation, extraction_date`

## redeep_tokens_cmp.json

`{"head_list": [...], "rows": {uid: {"tokens": [...], "ecs_tok": [...], "pks_tok": [...]}}}` for the 100 comparison rows.

---

## probe_feats.npy

Shape: `[N, 3, hidden_dim]`  dtype: float16

Row order matches `probe_index.json`.

## probe_index.json

```json
{ "uids": ["<uid>", ...], "layers": [L1, L2, L3] }
```

---

## scores.parquet  (final merged table, out-of-fold)

Base columns: `uid, qid, condition, failure, answered, fold`. Score columns (higher = more likely failure):
`s_redeep_c1` (original), `s_redeep` (ours), `s_redeep_o1` (published hparams, if heads match), `s_ecs`, `s_pks`,
`s_lr_ecspks`, `s_lr_ecspks_bm25`, `s_mlp_ecspks`, `s_probe_L1/L2/L3`, `s_probe_best`,
`s_perplexity`, `s_lnentropy`, `s_firsttok_entropy`, `s_bm25_margin`, `s_cond_prior` (reference, not deployable).
Cross-noise (E7, evaluate only on other conditions): `x_redeep_trC2`, `x_probe_trC2`.

## thresholds.json

`{detector: {fold: tau}}` — flag/abstain when score >= tau of the row's fold.
