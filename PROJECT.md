# rag-noise-probe — Project Record

## Research Question
Do ReDeEP's internal signals (ECS, PKS) continue to predict incorrect answers when retrieval
context contains noise, misleading information, or missing evidence?  Can we use these signals
to determine when an LLM should abstain rather than answer?

## Dataset Decision (made 2026-11-07)
**SQuAD 2.0 (`rajpurkar/squad_v2`)**: each answerable question ships with its gold paragraph, and the
~20k unique paragraphs (≤220 words, train+validation) form the distractor / BM25 pool.
150 validation questions (one per paragraph, answers ≤5 words), seed 0. `nq_open` was not used (no passages).

Build: `python -m src.data.noise_builder --n 150 --out artifacts/data` (~45 s on CPU).
Result: 750 rows, 0 validation errors; 6 spare questions dropped because the C5 swap failed (see `validation_report.json`).

**C5 counterfactual construction (changed during hand check):** the first version swapped in a random
answer of any type (e.g. "U.N. Secretary **30%**", climate → "**two**"), which a model can trivially ignore.
The replacement is now drawn from all SQuAD-validation answers with the **same question wh-word**
(who/when/where/how many/…) **and the same surface type** (date / number / proper noun / common phrase), with
similar length; swaps use word boundaries. "who/when/how many" swaps are now plausible (Newton → "Jean Ribault"
style); "what" questions remain loosely matched — **limitation**, no NER used.

Hand-read: 4 questions × 5 conditions read on 2026-11-07 (gold present, hard negatives on-topic without the
answer, C4 has no gold). Plan asks for 15–20 rows: **P2 still to finish**.

## Model
- Primary: `meta-llama/Llama-2-7b-chat-hf`  FP16, greedy decoding, `max_new_tokens=32` (Kaggle T4 x2)
- CPU smoke tests: `HuggingFaceTB/SmolLM2-135M-Instruct` (Llama architecture, fallback head list)

## Retrieval Conditions
| ID | Name | Description | failure = |
|----|------|-------------|-----------|
| C1_gold | Gold | Gold passage only | not correct |
| C2_random | Random | Gold + 3 random distractors | not correct |
| C3_hardneg | Hard negatives | Gold + 3 BM25 hard negatives (no answer string) | not correct |
| C4_missing | Missing | 3 hard negatives, gold removed | model answered (should abstain) |
| C5_counterfactual | Counterfactual | Gold + copy with answer swapped | not correct w.r.t. ORIGINAL answer |

## E0 — reproduction of the base paper (done, CPU)
`python -m src.repro.e0_reproduce` → `artifacts/results/e0_reproduction.json`

| | AUC | PCC |
|---|---|---|
| Paper, ReDeEP(chunk), Llama-2-7B, RAGTruth | 0.7458 | 0.4203 |
| Authors' shipped result file | 0.7473 | 0.4207 |
| **Our CPU run of the authors' `chunk_level_reg.py` on their shipped features** | **0.7473** | **0.4207** |
| Paper, ReDeEP(token) (features not shipped → reference only) | 0.7325 | 0.3979 |

Two shims, both documented in the JSON: (1) stub `source_info.jsonl` (only feeds an unused `task_type`);
(2) **bug in the authors' script**: `df.iloc[:, :int(df.shape[1]*0.5)]` slices columns and crashes with
KeyError; we use the first half of rows (`df.iloc[:int(df.shape[0]*0.5)]`). Ranking on all rows gives the same AUC.
Authors' repo commit: `4d08191`.

## ReDeEP implementation (what we can claim)
`src/extract/redeep_extract.py` re-implements the feature code of the authors' `token_level_detect.py`
**on stock transformers**: their fork only adds `knowledge_layers=` to expose the residual stream between
attention and FFN; we capture the same tensor with a pre-hook on `post_attention_layernorm` (unit test
`tests/test_extract.py` proves `h_{l+1} = r_l + FFN(r_l)`). ECS = cosine(final hidden state, mean final hidden
state of the top-10% attended prompt tokens) for the authors' 32 copy heads (`topk_heads.json`); PKS = authors'
JSD(×10e5) between logit-lens of the stream after and before each FFN. Fitting (`src/detect/fit_redeep.py`)
follows `token_level_reg.py` (AUC ranking, top-K sums, min-max, `PKS − β·ECS`, mean over tokens).
→ Report as **"authors' definitions and fitting procedure, re-implemented with hooks (adapted, not the fork)"**.
Our answers are teacher-forced from the generated token ids; `tf_match` checks they are reproduced (1.0 in smoke test).

## Frozen Scope
- 150 questions × 5 conditions = 750 benchmark rows
- Prompt version: `prompt_v1` (see `src/data/prompt_templates.py`)
- Evaluation: 5-fold grouped cross-validation (`folds_v1.json`), inner 3-fold for hyperparameters/τ,
  bootstrapped CIs (n=1000, grouped by qid); AUROC n/a if <10 failures or successes
- 20-prompt comparison set: `comparison_set_v1.json` (seed 1, drawn before any model output)
- Final deadline: **November 17, 2026**

## Open questions (ask mentor)
- "20 standard prompts": ours (20 random benchmark questions) or the paper's RAGTruth prompts? (plan §8)

## Team
| Person | Primary Role |
|--------|-------------|
| P1 | GPU, model generation, ReDeEP extraction |
| P2 | Dataset, noise construction, labeling |
| P3 | Reproduction, baselines, detectors, analysis |
