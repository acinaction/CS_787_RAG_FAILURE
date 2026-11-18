# EXECUTION PLAN — "Knowing When Not to Answer" (3 people, 7 days, async)

Base paper: ReDeEP (Sun et al., ICLR 2025). Code: github.com/Jeryi-Sun/ReDEeP-ICLR.
Everything below is written so that **nobody needs to be online at the same time**. Work moves between people only through **files with fixed names and fixed columns** (the "contracts" in section 3). If a file passes its DONE check, the next person can start without talking to anyone.

People: **P1** = GPU/extraction owner, **P2** = data/labels owner, **P3** = baselines/analysis owner. Each stage has a **backup owner** in case someone disappears for a day.

---

## 0. Ground rules (read once, follow always)

1. **Contracts are frozen.** Column names, file names, condition codes and prompt text in section 3 do not change after the freeze time in each card. If you must change one, bump the version (`v1`→`v2`), tell the others, and everything downstream is re-run. Do not edit silently.
2. **Every long script is resumable.** It appends one line per finished item and skips ids already present. Kaggle sessions die; this must not matter.
3. **Big files never go in git.** Git = code + small json/csv. Artifacts (jsonl/parquet/npy) go in one shared Google Drive folder `artifacts/` with the exact names below. After uploading, add one line to `MANIFEST.md` (name, rows, date, who).
4. **Status ping at the end of each work session** (template in section 8). This is how the others know what they can start.
5. **Debug small, run big.** Every GPU script is first tested on `--n 20` (laptop with a tiny model, or Kaggle), and only then launched fully.
6. **Time-box.** Each stage has a limit. At the limit you switch to that stage's Plan B. No heroics.
7. **Nobody is blocked waiting.** For every stage there is a "mock" input (section 3.9) so downstream people can build and test their code before real data exists.

---

## 1. Decisions already made (do not re-debate)

| Item | Decision |
|---|---|
| Method base | ReDeEP: External Context Score (ECS, copying heads) + Parametric Knowledge Score (PKS, FFN logit lens). Reuse authors' code and their provided `topk_heads.json`. |
| Models | Main: **Llama-2-7b-chat-hf**. Second (if time): **Meta-Llama-3-8B-Instruct**. Both have `topk_heads.json` in the repo. fp16, greedy decoding. |
| Compute | Kaggle (2×T4) for anything that loads a 7B model. Laptops: data, labels, baselines, probes, plots, and debugging with a tiny model. |
| Detection timing | **Draft-then-abstain**: the model writes a short answer, ReDeEP scores it (teacher-forced), we decide to show or abstain. Plus a **pre-generation probe** (last prompt token hidden state) as the "before generation" method. |
| Noise conditions | C1 gold only · C2 gold + 3 random · C3 gold + 3 hard negatives · C4 missing answer (3 hard negatives, no gold) · C5 counterfactual (gold + copy with answer entity swapped). Sweeps (C6) are stretch. |
| Unit of analysis | `uid = qid + "__" + condition`. Splits are by `qid` (60/20/20) so one question never appears in two splits. |
| Scale | 600 questions × 5 conditions = 3000 rows per model. Scale down to 300 if compute is short. |

## 2. Two things to verify in the first 30 minutes (Day 1, P2)

**V1. Does `nq_open` contain gold passages?** As I remember it, the HuggingFace `nq_open` has only `question` and `answer` (no passages) — please check by printing its columns. The TA guide says it has passages; if it doesn't:
- **Branch A (passages exist):** use NQ as in the guide.
- **Branch B (no passages, expected):** use **SQuAD 2.0** (`rajpurkar/squad_v2`) as the base: every answerable question comes with its gold paragraph, and the full paragraph pool (~20k paragraphs, train+dev) serves as the source of random and hard-negative distractors. Noise conditions are unchanged. Tell the TA "we use SQuAD-based passages for clean gold labels".
- Record the result in `PROJECT.md` under "Dataset decision".

**V2. Can the authors' fork be installed on Kaggle?** (P1, Day 1). `pip install -e transformers` from the cloned repo, then `python -c "import transformers; print(transformers.__version__)"` (expect `4.42.0.dev0`). If it fails, note the error and go to Plan B of stage S4.

---

## 3. Repo layout, naming, and contracts

```
rag-noise-probe/                 (git)
  PROJECT.md  SCHEMA.md  MANIFEST.md  STATUS.md
  src/data/        noise_builder.py  prompt_templates.py  evaluator.py  make_mock.py
  src/gen/         gen.py
  src/extract/     redeep_extract.py
  src/detect/      baselines.py  fit_redeep.py  fit_probes.py
  src/analysis/    metrics.py  make_tables.py  make_figs.py
  tests/
  third_party/ReDEeP-ICLR/       (git clone of the authors' repo)
artifacts/                        (Google Drive, NOT git)
  data/    benchmark_mini_v0.jsonl  benchmark_v1.jsonl  splits_v1.json
  runs/<model>/   gen_std.jsonl  gen_abstain.jsonl  labeled_std.jsonl  labeled_abstain.jsonl
                  redeep_feats.parquet  redeep_meta.json  probe_feats.npy  probe_index.json
  results/<model>/  scores.parquet  tables/*.csv  figs/*.png
```
`<model>` is `llama2-7b` or `llama3-8b`.

### 3.1 `benchmark_v1.jsonl` (produced by P2, one JSON per line)
| field | type | meaning |
|---|---|---|
| `uid` | str | `qid__condition` |
| `qid` | str | question id |
| `condition` | str | one of `C1_gold, C2_random, C3_hardneg, C4_missing, C5_counterfactual` |
| `question` | str | |
| `gold_answers` | list[str] | acceptable answers (original, never swapped) |
| `answerable` | bool | `false` only for C4 |
| `passages` | list[{pid, text, role}] | already shuffled; `role` ∈ gold, random, hardneg, counterfactual |
| `n_distractors` | int | |
| `gold_position` | str | first / middle / last / none |
| `bm25_top1`, `bm25_margin` | float | retriever-confidence features for the baseline tier 2 (score of top-1 passage and top-1 minus top-2, computed with BM25 for the question over the passages shown) |
| `swapped_entity` | str or null | only C5: the fake answer inserted |

Rules checked by tests: (a) C2/C3/C4 passages other than gold never contain any gold answer string; (b) C4 has no gold passage; (c) C5's counterfactual copy contains `swapped_entity` and not the original answer; (d) every `qid` has all 5 conditions; (e) no duplicate `uid`.

`splits_v1.json`: `{ "<qid>": "train|val|test" }`, assigned by hashing `qid` with seed 0 (60/20/20).

### 3.2 Frozen prompt (`prompt_templates.py`, version string `prompt_v1`)
- System: `You are a helpful assistant.`
- User (std): `Answer the question using the passages below. Give a short answer (a few words only).\n\nPassage 1: …\n\nPassage 2: …\n\nQuestion: …`
- User (abstain): same, plus `If the passages do not contain the answer, reply exactly: I don't know.`
- Built with the model's own `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Max prompt 12000 characters as in the authors' code.

### 3.3 `gen_<template>.jsonl` (produced by P1, `gen.py`)
`uid, model, template (std|abstain), prompt_version, answer, n_answer_tokens, answer_token_ids, answer_token_logprobs, first_token_entropy, first_token_maxprob, prompt_len, gen_seconds`. Greedy decoding, `max_new_tokens=32`.

### 3.4 `labeled_<template>.jsonl` (produced by P2, `evaluator.py`)
Everything from 3.3 plus: `refusal` (bool), `answered` (= not refusal), `f1` (max over gold answers), `correct` (answered and (normalized gold in normalized answer, or f1 ≥ 0.5)), **`failure`**:
- if `answerable`: `failure = not correct`
- if not answerable (C4): `failure = answered` (answering when it should abstain)
For C5, `correct` is judged against the original `gold_answers`; an answer equal to `swapped_entity` is a failure and is also flagged `followed_counterfactual=true`.

### 3.5 `redeep_feats.parquet` + `redeep_meta.json` (produced by P1, `redeep_extract.py`)
One row per `uid`: `ecs_mean` (list, length K = number of copy heads in `topk_heads.json`), `ecs_first` (same, value at first answer token), `pks_mean` (list, length L = number of layers), `pks_first`, `n_tokens`. `redeep_meta.json` stores the head list, layer count, model name, repo commit hash, JSD scaling used. ECS/PKS definitions follow the paper (Eq. 2–6) and the authors' `token_level_detect.py` — do not re-derive them.

### 3.6 `probe_feats.npy` + `probe_index.json` (produced by P1, same pass)
`probe_feats.npy`: float16 array `[N, 3, hidden_dim]` = hidden state at the **last prompt token** at layers ⌊L/3⌋, ⌊L/2⌋, ⌊3L/4⌋. `probe_index.json`: `{"uids": [...], "layers": [...]}` in the same row order.

### 3.7 `scores.parquet` (the single hand-off into analysis; produced by P3)
One row per `uid`: `uid, qid, split, condition, failure, answered` plus **one column per detector**, named `s_<method>`, where **higher = more likely failure**. Methods: `s_perplexity, s_lnentropy, s_firsttok_entropy, s_bm25_margin (negated), s_ecs, s_pks, s_redeep, s_lr_ecspks, s_mlp_ecspks, s_probe_L1/L2/L3`. Every detector writes its column into this table and nothing else. `analysis/*` reads only this table. That is how detector work and analysis work stay independent.

### 3.8 Metrics (frozen)
AUROC, AUPRC, Pearson r (score vs failure), risk–coverage curve and AURC (lower is better), all reported **per condition and pooled**, on the **test split only**. 95% CIs by bootstrap over `qid` (1000 resamples). Detector hyperparameters are tuned on `val` only.

### 3.9 Mock data (so nobody waits)
`make_mock.py` (P2, Day 1) writes `artifacts/data/mock/` with the same schemas as 3.3–3.7 but fake numbers (random features, failure correlated with a random signal). P1 uses it to test the extract-output writer, P3 uses it to build and test the whole detector + analysis pipeline before any real run exists. Swapping mock→real is just changing the folder path.

---

## 4. Pipeline at a glance

```
S0 setup (all) ─┬─> S1 data (P2) ──> benchmark_mini_v0 ─> S2 gen (P1) ─> S3 labels (P2) ─┐
                │                    benchmark_v1  ───────────┘             │            │
                │                                                           ▼            ▼
                ├─> S8 reproduction (P3, CPU, Day 1)       S4 redeep+probe feats (P1) ─> S5 baselines (P3)
                │                                                           │             │
                │                                                           ▼             ▼
                └────────────────────────────────────────────────> S6 detectors (P3) ─> S7 analysis (P3+P2) ─> S9 writing (all)
```
Critical path: S1 → S2 → S4 → S6 → S7. Everything else hangs off it with slack.

---

## 5. Stage cards

Each card: **Owner / Backup · Needs · Steps · Produces · DONE when · Time-box · Plan B**.

### S0 — Setup (all three, Day 1, ≈2 h each, can be done separately)
- **Everyone:** GitHub repo `rag-noise-probe` created (P3 creates, invites others); shared Drive folder `artifacts/` created (P2); Kaggle account with phone verification (needed for GPU and internet); HuggingFace account.
- **Model access (P1):** request Llama-2-7b-chat-hf and Meta-Llama-3-8B-Instruct on HuggingFace today (approval can take time). Meanwhile test with a mirror repo if one exists (verify the repo name before relying on it).
- **Clone** the authors' repo into `third_party/` (P1, P3).
- **DONE when:** repo exists with the folder skeleton; each person has run `python -c "import torch"` on their machine; HF access requests sent; STATUS.md created.
- **Plan B:** no gated access by Day 3 → use a mirror; if no mirror works, run Plan C (hooks version on an open Llama-architecture model; see S4).

### S8 — Reproduction on CPU (Owner **P3**, Backup P1, Day 1, ≈2 h, no GPU)
- **Needs:** authors' repo, dataset from their Google Drive links (there are two: `readme.md` and `dataset/readme.md`; get both).
- **Steps:** put the dataset files into `third_party/ReDEeP-ICLR/dataset/` (expected names `response.jsonl`, `source_info.jsonl`, plus the Dolly files). `cd third_party/ReDEeP-ICLR/ReDeEP` (scripts use relative paths like `../dataset` and `../transformers/src`, so run from that folder). Run `python chunk_level_reg.py --model_name llama2-7b --dataset ragtruth`. Read the script and note how scores are min–max normalized, how top-K heads/layers are selected and how β is tuned; write 10 lines in `notes/redeep_regression.md` (P3 reuses this logic in S6).
- **Produces:** `results/reproduction.md` = our number vs paper (paper: AUC 0.7458, PCC 0.4203 for Llama-2-7B chunk on RAGTruth; the repo's own log shows 0.7473/0.4207).
- **DONE when:** a number within about ±0.02 AUC of the paper is recorded.
- **Plan B:** dataset link dead or script errors → use the shipped `log/test_llama2_7B/ReDeEP(chunk).json` as the reported reference and state honestly "regression re-run not possible, used authors' logs". Do not burn more than 3 h here.

### S1 — Data (Owner **P2**, Backup P3)
- **Needs:** nothing (V1 decision first).
- **Steps:**
  1. Day 1: V1 check. Write `noise_builder.py` for the chosen branch, the tests in 3.1, and `make_mock.py`. Produce **`benchmark_mini_v0.jsonl`** (20 qids × 5 conditions = 100 rows) **first** — this unblocks P1.
  2. Day 1–2: scale to **`benchmark_v1.jsonl`** (600 qids × 5) and `splits_v1.json`. BM25 over the passage pool for hard negatives (`rank_bm25` or sklearn TF-IDF). Hard negatives = top-ranked passages that do **not** contain any gold answer string. Random noise = uniformly sampled passages from other questions. Counterfactual = replace the answer string in a copy of the gold passage with another question's answer.
  3. Print token-length stats of the longest prompts (so P1 knows memory needs).
  4. Freeze `prompt_templates.py` (`prompt_v1`) by end of Day 2.
- **Produces:** `benchmark_mini_v0.jsonl` (**due end of Day 1**), `benchmark_v1.jsonl` + `splits_v1.json` + mock folder (**due Day 2 noon**).
- **DONE when:** all 5 tests pass; 15 random rows hand-read and look right; condition counts equal.
- **Time-box:** 1.5 days. **Plan B:** BM25 install fails → sklearn TF-IDF. Too few clean hard negatives for some questions → drop those qids (document how many). C5 swap looks unnatural → keep it, but add a note that C5 measures blind copying, not realism.

### S2 — Generation (Owner **P1**, Backup P2)
- **Needs:** `benchmark_mini_v0.jsonl` (to start), `benchmark_v1.jsonl` (full), `prompt_templates.py`.
- **Steps:**
  1. Day 1 (laptop, tiny Llama-architecture model such as TinyLlama): write `gen.py`, run `--n 20`, check the output matches 3.3. Resumable: skip `uid`s already in the output file.
  2. Day 2 (Kaggle, Llama-2-7b-chat fp16, device_map auto): run `--n 20` on the mini file and **time it**. Extrapolate: 3000 rows × seconds/row. If > 4 h, reduce to 300 qids.
  3. Day 2–3: full run with template `std`, then template `abstain`. Checkpoint every 50 rows to `/kaggle/working`, download/upload to Drive at the end of each session.
- **Produces:** `runs/llama2-7b/gen_std.jsonl`, `gen_abstain.jsonl`.
- **DONE when:** row count = rows in `benchmark_v1` (or the agreed reduced set); no empty answers beyond 2%; three answers hand-read per condition.
- **Time-box:** Day 3 evening. **Plan B:** OOM on long prompts → truncate each passage to 150 words and re-freeze the benchmark as `v1.1` (tell everyone). Kaggle quota out → switch to another teammate's account (this is why the backup exists) or Colab. Model access blocked → mirror → Plan C.

### S3 — Labels (Owner **P2**, Backup P3)
- **Needs:** `gen_*.jsonl`, `benchmark_v1.jsonl`.
- **Steps:** `evaluator.py` implements 3.4. Refusal detector = regex list (`I don't know`, `cannot`, `not mentioned`, `no information`, …). Hand-check **40 rows** (8 per condition): compare machine `correct`/`refusal` with your own judgement. Tune rules until agreement ≥ 90%. Then label everything.
- **Produces:** `labeled_std.jsonl`, `labeled_abstain.jsonl` (first version on `mini` as soon as it exists; full as soon as S2 ends).
- **DONE when:** agreement ≥ 90% on the 40 hand-checked rows; a one-table summary (accuracy, refusal rate, failure rate per condition) is posted in STATUS.md.
- **Time-box:** half a day after S2. **Plan B:** agreement stuck below 90% → keep the rules, report the measured agreement and note label noise as a limitation (and use bootstrap CIs everywhere).

### S4 — ReDeEP + probe feature extraction (Owner **P1**, Backup P2)
- **Needs:** `labeled_std.jsonl` (answers), `benchmark_v1.jsonl`, authors' fork + `topk_heads.json`.
- **Steps:**
  1. Read `token_level_detect.py` closely (it is the reference). Write `redeep_extract.py` as a copy of its core loop with three changes: (i) read prompt + answer from our files instead of RAGTruth, (ii) response-level score = mean over answer tokens (and also keep the first token's value), (iii) also save the hidden state at the last prompt token (3 layers) for the probe. Keep their JSD function **exactly** (their GitHub issue #2 discusses a scaling quirk).
  2. Test on mock/mini (`--n 20`) on Kaggle. Check shapes: `len(ecs_mean) == len(topk_heads)`, `len(pks_mean) == num_layers`.
  3. Full run, resumable, appending parts (`part_000.parquet`, …) merged at the end.
  4. Sanity check before shipping: plot mean PKS per layer for correct vs failed answers (paper's Fig. 4d: late layers higher for hallucinations) and mean ECS correct vs failed. Seeing the same direction in C1 is the sign that extraction is right.
- **Produces:** `redeep_feats.parquet`, `redeep_meta.json`, `probe_feats.npy`, `probe_index.json` (Llama-2-7B). Repeat for Llama-3-8B only if time allows.
- **DONE when:** rows = rows in labeled file; no NaN; the sanity plot looks like the paper in C1.
- **Time-box:** Day 3 night → results Day 4 morning. **Plan B ladder:**
  - **B1:** fork won't install → I write the same extraction with hooks on a modern `transformers` (eager attention; hooks on each layer's attention output and MLP input/output; logit lens = final norm + `lm_head`). Debug it locally with a tiny Llama-architecture model first.
  - **B2:** extraction too slow → reduce to 300 qids; save only first-token + mean features; skip Llama-3.
  - **B3:** ECS/PKS bugs not fixed by Day 4 noon → freeze at ECS/PKS "approximate" version and spend remaining effort on the pre-generation probe + baselines (which need only `gen_*` and `probe_feats`). Report ReDeEP honestly as partially reproduced.

### S5 — Baselines (Owner **P3**, Backup P2)
- **Needs:** `labeled_std.jsonl` (token logprobs, first-token entropy), `benchmark_v1.jsonl` (`bm25_*`), `labeled_abstain.jsonl`.
- **Steps:** `baselines.py` writes columns `s_perplexity` (negative mean token logprob → exp), `s_lnentropy`, `s_firsttok_entropy`, `s_bm25_margin` (negated margin), and for the prompt-based abstention baseline records the refusal rate / failure rate of the abstain template per condition (this is a single operating point, not a score). Optional: `P(True)` (needs another GPU pass → only if time).
- **Produces:** columns in `scores.parquet`; `results/<model>/prompt_abstention.csv`.
- **DONE when:** each baseline's AUROC is computed on the mock/mini data without error, then on real data.
- **Time-box:** build on mock Day 1–2; real run takes minutes once S3 is done. **Plan B:** skip P(True); everything else is CPU-cheap.

### S6 — Detectors (Owner **P3**, Backup P1)
- **Needs:** `redeep_feats.parquet`, `probe_feats.npy`, `labeled_std.jsonl`, `splits_v1.json`.
- **Steps:**
  1. **ReDeEP score** `s_redeep = Σ_{l∈F} α·PKS_l − Σ_{(l,h)∈A} β·ECS_{l,h}` (higher = failure). α = 1; β grid 0.1…2.0 step 0.1; choose K_heads and K_layers (1…32) by greedy search on **val** AUROC (the paper's Appendix J procedure). Follow the normalization read in S8.
  2. `s_ecs` (ECS only) and `s_pks` (PKS only) — the paper's Table 6 ablation.
  3. `s_lr_ecspks`, `s_mlp_ecspks`: logistic regression / small MLP on [ECS vector, PKS vector], trained on train, tuned on val.
  4. `s_probe_L1/L2/L3`: logistic regression on `probe_feats` per layer (standardize features; strong L2 regularization; tune C on val).
  5. All fits are done **pooled across conditions** and also **per condition** (train on one condition, test on all) — needed for the cross-noise experiment.
- **Produces:** `results/<model>/scores.parquet` with all `s_*` columns.
- **DONE when:** every method has val and test AUROC logged and the scores table has no missing values.
- **Time-box:** Day 4–5. **Plan B:** no ECS/PKS yet → run steps 4 and the baselines on whatever exists; the code for steps 1–3 is already tested on mock data.

### S7 — Analysis and figures (Owner **P3**, with **P2** doing the qualitative part; Backup P1)
Experiments (each is one table or figure):
1. **E1 Noise hurts:** accuracy / refusal / failure rate per condition (both templates).
2. **E2 Detector comparison:** AUROC/AUPRC/PCC per condition + pooled, all methods, with CIs.
3. **E3 Sign flip:** per condition, AUROC of `s_ecs` and `s_pks` and the sign of their correlation with failure; highlight C4/C5 vs C1.
4. **E4 Ablation:** only-ECS, only-PKS, full (paper Table 6 style); probe layer L1/L2/L3.
5. **E5 Abstention:** risk–coverage curves and AURC for `s_redeep`, best probe, perplexity/entropy, BM25 margin; the prompt-based abstain template as a single point on the same plot.
6. **E6 Cross-noise generalization:** train on C2, test on C3/C4/C5 (and the reverse); heatmap.
7. **E7 Qualitative (P2):** 15 cases: failures caught by internal signals but missed by output confidence, and the reverse; 5 C4 cases where the model answered anyway.
8. Stretch: second model (Llama-3-8B), C6 sweeps (distractor count 0/1/3/5, gold position first/middle/last), second dataset.
- **DONE when:** `tables/*.csv` and `figs/*.png` for E1–E5 exist and are regenerated by one command (`make_tables.py && make_figs.py`).
- **Time-box:** Day 5–6. **Plan B:** if internal signals show no advantage, the finding is "where and why they fail under noise" (E3 and E6 become the story). If E6 is impossible (too few failures in a condition), drop it and say why.

### S9 — Writing (all; each writes what they produced; Day 6–7)
See section 9.

---

## 6. Day-by-day with owners (what each person delivers and what they can start)

Because you are not available together, each cell is something one person can do alone, in order. "Starts when" is the dependency.

| Day | P1 (GPU) | P2 (data/labels) | P3 (baselines/analysis) | End-of-day artifact |
|---|---|---|---|---|
| **1** | S0; V2 (fork install); write `gen.py` and test with tiny model locally. | S0; V1 dataset check; `noise_builder.py` + tests; **`benchmark_mini_v0` by evening**; `make_mock.py`. | S0; **S8 CPU reproduction**; read `chunk_level_reg.py`; start `metrics.py` using mock data. | `benchmark_mini_v0.jsonl`, `reproduction.md`, mock folder |
| **2** | Kaggle: gen on mini, time it; start `redeep_extract.py` (copy of authors' loop) and test on mini. | Scale to `benchmark_v1` + splits; freeze prompts; start `evaluator.py` on mini outputs. | `baselines.py` + `fit_redeep.py` + `fit_probes.py` written and passing on mock. | `benchmark_v1.jsonl`, `splits_v1.json`, prompt freeze |
| **3** | Launch full `gen_std` then `gen_abstain` (Llama-2-7B). Then launch `redeep_extract` on the finished std outputs (night run). | Finish `evaluator.py`, 40-row hand-check, label `std` as soon as it lands. | Run baselines on real labeled data when it appears. | `gen_*.jsonl`, `labeled_std.jsonl` |
| **4** | Collect extraction output; sanity plot (C1); upload. Start Llama-3-8B gen if everything is green. | Label `abstain`; E1 table; start E7 case selection. | S6 detectors on real features; val tuning. | `redeep_feats`, `probe_feats`, `scores.parquet` v1 |
| **5** | Llama-3-8B extraction (stretch) **or** C6 sweep generation (stretch); otherwise help P3 with E3/E6. | E7 qualitative cases; check label noise in C5. | E2–E6 tables and figures. | first full set of tables/figs |
| **6** | Ablation reruns on request; TriviaQA/second-model stretch; fill the "Method" and "Setup" sections. | "Dataset & noise" and "Labels" sections; polish E7 table. | "Results/Ablations" sections; bootstrap CIs; final figures freeze. | figure freeze, report draft v1 |
| **7** | Slides: method + setup. No new runs. | Slides: data + qualitative. | Slides: results + conclusions; rehearsal. | final report + slides |

**If you only have ~2 hours in a session,** pick the first undone item in your column. If it needs a file that isn't there yet, use the mock folder and tell the others in your status ping.

## 7. Risk and Plan-B master table

| Risk | Trigger | Plan B |
|---|---|---|
| `nq_open` has no passages | V1 check | Branch B: SQuAD 2.0 base (section 2). |
| Llama access delayed | No approval by Day 3 | Mirror repo → else Plan C hooks version on an open Llama-architecture model. |
| Fork won't install / old-stack conflicts | V2 fails on Day 1–2 | S4-B1: modern-transformers hooks version (I write it). |
| Out of GPU memory | OOM on long prompts | Truncate passages (benchmark `v1.1`), then reduce qids, then 4-bit (document that numbers differ from paper). |
| Kaggle quota/session limits | Quota hit, session killed | Resumable scripts; switch account (backup owner); Colab. |
| Extraction slower than planned | > 4 s/row after timing | 300 qids; mean features only; drop Llama-3. |
| ECS/PKS show no signal in noisy conditions | AUROC ≈ 0.5 in C2–C5 | Report as finding; probes + baselines carry E5; E3 becomes the main story. |
| Labels noisy | < 90% agreement | Keep, report agreement, rely on CIs; remove ambiguous C5 rows only if documented. |
| Reproduction can't be re-run | Drive link dead | Cite authors' logs; state limits plainly. |
| Someone unavailable | Misses a stage's time-box | Backup owner takes the stage using this doc; the artifacts and DONE checks make it possible. |
| Everything late on Day 6 | — | Cut order: C6 sweeps → second dataset → Llama-3 → MLP. Never cut: S8, E1, E2, E3, E5. |

## 8. Communication templates

**Status ping (end of each session):**
```
[P#] done: <stage/file, rows, where in Drive>
[P#] next: <what I do next>
[P#] blocked on: <file or nothing>
[P#] heads-up: <anything others must know / any contract change>
```
**Handoff message (when a stage is DONE):**
```
HANDOFF S<n> → P<m>
file: artifacts/<path> (rows=<n>, version=<v>)
checked: <the DONE checks I ran, results>
known issues: <or none>
you can start: <exact next step>
```
**Git conventions:** branches `p1/...`, `p2/...`, `p3/...`; small commits; main only updated by pull request or by agreement; `STATUS.md` and `MANIFEST.md` edited by everyone (append only).

## 9. Report and slides: who writes what

| Section | Writer | Source artifacts |
|---|---|---|
| Abstract, Intro, Related work (ReDeEP, SAPLMA/SEP, knowledge conflict, selective prediction) | P3 | notes from S8, paper |
| Method: ECS, PKS, ReDeEP score, probes, draft-then-abstain | P1 | `redeep_meta.json`, authors' code |
| Dataset, noise construction, labels, prompts | P2 | `benchmark_v1`, evaluator, hand-check table |
| Experimental setup (models, hardware, metrics, splits) | P1 + P3 | section 3.8 |
| Results E1 | P2 | |
| Results E2–E6 + ablations | P3 | `scores.parquet`, tables/figs |
| Qualitative (E7), failure analysis | P2 | |
| Discussion, limitations, conclusion | all (P3 drafts) | |

Centerpiece results for the presentation: **E3** (sign flip), **E2** (detector comparison per noise type), **E5** (risk–coverage vs prompt-based abstention), **reproduction row**, and **one E7 example**.

## 10. Minimum submission checklist (if everything goes wrong, this is enough)
- [ ] Reproduction row (S8).
- [ ] Llama-2-7B, 600 or 300 questions, conditions C1–C5, labeled.
- [ ] ECS, PKS, ReDeEP score and at least 2 baselines with per-condition AUROC and CIs.
- [ ] One risk–coverage figure with the prompt-based abstain point.
- [ ] Honest limitations section.

## 11. Things I could not verify (check them, don't assume)
- Whether the authors' fork installs on current Kaggle images (V2).
- Whether `nq_open` has passages (V1).
- Real seconds-per-row on a T4 (measure on 20 rows).
- Exact names of ungated mirror repos on HuggingFace.
- Whether the TA guide's claimed existing code (schema, noise builder, tests) is real; this plan assumes nothing exists, which matches what you told me.
