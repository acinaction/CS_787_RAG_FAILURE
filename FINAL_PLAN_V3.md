# FINAL PLAN (V3) — Knowing When Not to Answer
**Predicting RAG failure from LLM internal states under retrieval noise**
Base paper: ReDeEP — Sun et al., ICLR 2025 · Team: P1, P2, P3 · Core schedule: **3 days** (7-day extension listed in section 10)

This is the document to follow and to show the mentor/TA. `EXECUTION_PLAN.md` stays as the operating manual (file contracts, stage cards, Plan Bs, message templates). **Where the two differ, this file wins.** The differences are listed in section 11.

---

## 1. The question and the two sides we compare

**Research question.** ReDeEP detects RAG hallucinations from internal signals when the retrieved context is *accurate and relevant* (the paper states this scope in its introduction). Do those signals still predict wrong answers when retrieval is noisy, and can we use them to decide when to abstain?

The mentor's comparison is "what the original paper does" vs "what we add". We define both sides precisely, so the comparison is fair:

| | **ORIGINAL — ReDeEP as published** | **OURS — noise-aware extension** |
|---|---|---|
| Assumption | Retrieved context is clean | Context may be noisy, misleading or missing the answer |
| Signals | ECS (copy heads) + PKS (FFN logit lens), same extraction code | Same signals, same extraction code |
| Score | `H = Σ α·PKS − Σ β·ECS` | Same formula |
| Fitting data | **Clean condition (C1) only**: choose top-K heads, top-K layers, β, and threshold τ using clean training rows | **All conditions pooled**, with no condition label given to the detector |
| Extra inputs | none | Retriever features (BM25 margin) as an optional variant (`s_lr_ecspks_bm25`), pre-generation probe (`s_probe_best`) |
| Decision | Flag as "hallucination" when `H > τ` | Abstain when score > τ, and risk–coverage curve over all thresholds |
| Evaluated on | Same rows, conditions C1–C5 | Same rows, conditions C1–C5 |

Only the **fitting data, optional extra features and the use as an abstention rule** differ. Model, prompt, signals, scoring formula, splits and metrics are identical, so any gap comes from our extension.

Three reference points for "as published" (use as many as feasible):
1. **Paper numbers (reference only):** RAGTruth, Llama-2-7B — ReDeEP(token) AUC 0.7325, PCC 0.3979; ReDeEP(chunk) AUC 0.7458, PCC 0.4203.
2. **Our reproduction (CPU, Day 1):** run the authors' `chunk_level_reg.py` on their shipped Llama-2-7B features; report our number next to the paper's.
3. **O2 — published method, clean-fit on our data (PRIMARY "original" side):** the paper's own procedure (rank heads/layers, greedy top-K search, α = 1, β grid 0.1–2.0, validation-selected threshold), run on C1 rows only. This is `s_redeep_c1`.
   Optional **O1 — published hyperparameters as-is:** for Llama-2-7B the paper's Appendix J lists top-1 head, top-10 FFN layers, α = 1, β = 0.2, threshold 0.6 (token level). Apply them without refitting. Treat this as a sanity check only: our prompts (short-answer QA) differ from RAGTruth (long responses) and the score normalisation must match the authors' code, so these values may not transfer.

## 2. What "implementing the paper" means here (and what we will not claim)
- ECS, PKS and the combined score come from the **authors' code** (`github.com/Jeryi-Sun/ReDEeP-ICLR`, with their `topk_heads.json` and forked `transformers`), adapted to read our prompts and answers. We understand and explain each equation (paper Eq. 2–6) and say plainly which parts are reused.
- **Token-level ReDeEP** is the mandatory path (our answers are ≤32 tokens, so it is cheap). **Chunk-level** is optional.
- If the authors' fork cannot run and we write our own hooks, the report says "adapted implementation", never "exact reproduction". An approximate ECS/PKS is never called ReDeEP.
- We **do not** replace ReDeEP with a simple classifier. The hidden-state probe is an additional pre-generation method, not the core.

## 3. Scope

| Mandatory | If time allows | Dropped for the 3-day version |
|---|---|---|
| Llama-2-7B-chat, fp16, greedy | Logistic-regression probe on last-prompt-token hidden states | Llama-3-8B |
| 150 questions × 5 conditions = 750 rows | `s_lr_ecspks_bm25`, small MLP | Second dataset (TriviaQA) |
| ECS + PKS + ReDeEP score (token level) | Cross-noise C2→C3, C2→C5 | C6 distractor/position sweeps |
| Original (C1-fit) vs ours (pooled) | Chunk-level ReDeEP | P(True) |
| Baselines: perplexity, LN-entropy, first-token entropy, BM25 margin, abstain-prompt | O1 published hyperparameters | GPU reproduction on RAGTruth |
| Cross-noise C2→C4 | | |
| 20-prompt comparison | | |
| CPU reproduction row | | |

## 4. Data and splits
- **Source:** decide on Day 1. Check whether `nq_open` contains gold passages. As I remember it, it does not; if so use **SQuAD 2.0** (gold paragraph included; paragraph pool for distractors). Record the decision in `PROJECT.md`.
- **Conditions:** C1 gold · C2 gold + 3 random · C3 gold + 3 BM25 hard negatives (no answer string) · C4 3 hard negatives, no gold (correct behaviour: abstain) · C5 gold + a copy with the answer entity swapped (judged against the **original** answer).
- **Validation checks (P2):** no gold answer string in non-gold passages of C2/C3/C4; C5 copy contains the swapped entity and not the original; every qid has all five conditions; 15–20 rows hand-read.
- **Evaluation protocol: 5-fold cross-validation grouped by `qid`** instead of a fixed 90/30/30 split. A 30-question test set gives 30 rows per condition and very wide intervals. With grouped CV every question is tested exactly once, using predictions made by a model that never saw it (out-of-fold). All five variants of a question always sit in the same fold.
  - In each outer fold, hyperparameters and thresholds are chosen **only on that fold's training part** (inner 3-fold grouped CV). Never on the held-out fold.
  - All reported numbers are computed on the pooled out-of-fold predictions, with bootstrap confidence intervals over `qid` (1000 resamples).

## 5. Pipeline, owners and handoffs (summary; full contracts in `EXECUTION_PLAN.md` §3)

| Artifact | Producer → consumer | Check before handoff |
|---|---|---|
| `benchmark_v1.jsonl`, `folds_v1.json`, `comparison_set_v1.json` | P2 → P1, P3 | validation checks above; 15 rows read by hand |
| `gen_std.jsonl`, `gen_abstain.jsonl` | P1 → P2, P3 | row count = benchmark; empty answers < 2% |
| `labeled_std.jsonl`, `labeled_abstain.jsonl` | P2 → P3 | ≥ 90% agreement on 40 hand-checked rows |
| `redeep_feats.parquet`, `redeep_meta.json`, `probe_feats.npy`, `redeep_tokens_cmp.json` | P1 → P3 | shapes match head/layer lists; late-layer PKS higher for failures in C1 (paper Fig. 4d direction) |
| `scores.parquet` (out-of-fold, one `s_<method>` column per detector, higher = more likely failure) | P3 → analysis | no missing values |
| `comparison_20.csv`, `comparison_20.png`, tables, figures | P3 + P2 → report | 20×5 grid complete, disagreements annotated |

Mock data (same formats, fake numbers) is produced by P2 on Day 1 so P3 can build and test everything before real data exists.

## 6. Detectors and the exact fitting recipe

All detectors output a score where **higher = more likely failure**.

| Column | Fitting data | Procedure |
|---|---|---|
| `s_redeep_c1` (**original**) | C1 training rows only | Rank copy heads by correlation of ECS with failure and layers by correlation of PKS with failure; greedy search over K_heads ∈ [1,32], K_layers ∈ [1,32]; α = 1; β ∈ {0.1,…,2.0}; pick the best inner-CV AUROC; τ = F1-optimal threshold on inner-CV scores. Follow the normalisation used in the authors' `chunk_level_reg.py`. |
| `s_redeep` (**ours**) | All training rows, pooled, no condition label | Same procedure. |
| `s_ecs`, `s_pks` | pooled | ECS-only / PKS-only (paper Table 6 ablation). |
| `s_lr_ecspks`, `s_lr_ecspks_bm25`, `s_mlp_ecspks` | pooled | Logistic regression / small MLP on the ECS and PKS vectors (second variant adds BM25 margin and top-1 score). The condition label is **never** a feature (it is not known at inference time). |
| `s_probe_L1/L2/L3`, `s_probe_best` | pooled | Standardised logistic regression on last-prompt-token hidden states at layers ⌊L/3⌋, ⌊L/2⌋, ⌊3L/4⌋; best by inner CV. |
| `s_perplexity`, `s_lnentropy`, `s_firsttok_entropy`, `s_bm25_margin` | none | Output-uncertainty and retriever-confidence baselines. |
| `s_cond_prior` (**reference, not deployable**) | pooled | Failure rate of the row's condition in training. Shows how much of a pooled AUROC is just "recognising the noise type". |

Prompt-based abstention is **not** a score: it is a single operating point (refusal rate and failure rate when the model is told to say "I don't know"). It appears as one point on the risk–coverage plot.

**Fairness rules for original vs ours:** same model, prompt, features, folds and metrics; the only changes are the fitting data and optional extra features. Both use the same threshold rule (F1-optimal on training-fold scores).

## 7. Experiments

| ID | What | Why it matters |
|---|---|---|
| E0 | **Reproduction row** (paper vs our CPU run) | Shows we can reproduce the base paper |
| E1 | Accuracy, refusal rate, failure rate per condition, both prompt templates | How much noise hurts |
| E2 | Per-condition and pooled AUROC/AUPRC/PCC for every detector, with CIs | Do signals survive noise? |
| E3 | **Sign analysis:** per condition, correlation of mean ECS (selected heads) and mean PKS (selected layers) with failure, no fitting | Does ECS change meaning when context is misleading (C4, C5)? |
| E4 | Ablation: ECS-only, PKS-only, full; probe layer | Which signal carries information |
| E5 | **Selective abstention:** risk–coverage curves and AURC for `s_redeep_c1`, `s_redeep`, best probe, baselines; prompt-abstain as one point | Can signals decide when to answer? |
| E6 | **Original vs ours, all rows:** `s_redeep_c1` vs `s_redeep` vs `s_probe_best`, per condition with CIs | Does noise-aware fitting help? |
| E7 | Cross-noise: train C2 → test C4 (priority), C3, C5 (if time) | Do signals generalise across noise types? |
| E8 | **20-prompt side-by-side** (section 8) | Mentor's request |
| E9 | Qualitative: 15 cases, plus 5 C4 cases where the model answered anyway | Failure analysis |

**Rigor notes:** (a) report **within-condition** AUROC as the main result; pooled AUROC can be inflated by simply detecting the condition (compare with `s_cond_prior`). (b) If a condition has fewer than 10 failures or fewer than 10 successes, report AUROC as "n/a" and say why. (c) Label noise is reported (hand-check agreement) and never hidden.

## 8. The 20-prompt comparison (mentor's request)

**Selection:** 20 questions drawn at random from all 150 with `seed = 1` on Day 1, **before any model output exists** (no cherry-picking). All 5 conditions per question → 100 rows. Predictions are the out-of-fold ones.

**Per (prompt, condition) cell:** apply each side's threshold τ and record one of: `kept-correct`, `caught-failure`, `missed-failure`, `false-abstain`.

**Outputs:**
1. `comparison_20.csv`: `qid, question, condition, answer, gold, correct, failure, orig_score, orig_flag, orig_outcome, ours_score, ours_flag, ours_outcome, probe_score`.
2. `comparison_20.png`: two heatmaps (Original | Ours): rows = the 20 prompts, columns = C1…C5, colour = outcome.
3. Report/slide table: `Prompt | Condition | Original says | Ours says | Truth | Why`, with a one-line reason for **every row where the two sides disagree** (at least 8), using the token-level ECS/PKS dump `redeep_tokens_cmp.json`.
4. Summary counts of the four outcomes per side per condition. Always shown next to E6 so the 20 prompts illustrate and the full set proves.

**If the mentor meant the paper's own prompts:** the original side uses 20 RAGTruth QA examples scored from the authors' shipped features (CPU), and the ours side uses our noisy versions. **Ask the mentor on Day 1**, one message.

## 9. Core 3-day schedule

The compute is small (rough, unmeasured: about 20 min per generation pass and 40 min extraction for 750 rows on a T4). **Time 20 rows first.** The real risks are the fork install, Llama access and label checking.

| | **P1 — GPU / ReDeEP** | **P2 — data / labels** | **P3 — detectors / analysis** |
|---|---|---|---|
| **Day 1** | HF access (or mirror). Install fork on Kaggle. Run the authors' script on 5 examples. Write `gen.py`, test with a tiny model, then on 20 rows. | V1 dataset check. `noise_builder.py` + tests. **Mini benchmark (20 qids) by noon.** Full 150 × 5, folds and `comparison_set_v1.json` by evening. Mock data. | CPU reproduction (E0). Read `chunk_level_reg.py` and write the 10-line recipe note. Write metrics and detector code on mock data. |
| **Day 2** | Full generation (std + abstain). Then ReDeEP + probe extraction (+ token dump for the 100 comparison rows). Upload. | `evaluator.py`, 40-row hand check (≥ 90% agreement), label everything. E1 table. | Baselines on real data. Fit `s_redeep_c1`, `s_redeep`, probes as soon as features land. `make_comparison.py` on mock. |
| **Day 3** | Method + setup sections, slides. Help with reruns. | Qualitative notes (E9), annotate disagreement rows (E8), data section. | E2–E7 tables and figures, comparison figure, results sections. **No new GPU runs after noon.** |

**Gates:**
- **G1 (Day 1 noon):** Llama loads and the fork runs, or switch to the hooks-based version (adapted, not exact).
- **G2 (Day 1 end):** benchmark frozen and valid; `gen.py` works on 20 rows.
- **G3 (Day 2 evening):** features exist and C1 sanity check (late-layer PKS higher for failures) passes.
- **G4 (Day 3 noon):** scores table complete; after this only plots and writing.

**If a gate fails:** cut optional items in this order: MLP → cross-noise beyond C2→C4 → probes → chunk-level. Never cut: ECS/PKS, the original-vs-ours comparison, per-condition AUROC, risk–coverage, the reproduction row.

## 10. If you have 7 days instead
Add in this order: Llama-3-8B (authors provide its `topk_heads.json`), TriviaQA, 300 questions, chunk-level ReDeEP, C6 sweeps, GPU reproduction on short RAGTruth QA prompts, O1 published hyperparameters.

## 11. Differences from `EXECUTION_PLAN.md` (this file wins)
- Evaluation is **5-fold grouped CV** with out-of-fold scores, not a fixed 60/20/20 split.
- `splits_v1.json` becomes `folds_v1.json`; the 20-prompt set is drawn from **all 150** questions, not only a test split.
- Added `s_cond_prior`, `s_lr_ecspks_bm25`, and the O1/O2 definition of "ReDeEP as published".
- Scale is 150 questions × 5 conditions; Llama-3 is optional.
- Metrics are reported on pooled out-of-fold rows, not on a "test split only".

## 12. Deliverables checklist
- [ ] E0 reproduction row (paper vs ours).
- [ ] 750-row benchmark, validated, with hand-check notes.
- [ ] Generation and labels for both prompt templates, with measured label agreement.
- [ ] ECS, PKS, `s_redeep_c1` (original) and `s_redeep` (ours) with per-condition AUROC and CIs.
- [ ] Baselines: perplexity, entropy, BM25 margin, abstain-prompt.
- [ ] Risk–coverage figure including the prompt-abstain point.
- [ ] 20-prompt comparison (CSV, figure, annotated table).
- [ ] Limitations: 150 questions, one model, one dataset, label noise, adapted vs exact code, noise type not known at inference.

## 13. Five lines for the mentor
1. Base paper: ReDeEP, which detects RAG hallucination from attention (ECS) and FFN (PKS) signals under clean context.
2. We reproduce its core signals with the authors' code and show our reproduction next to theirs.
3. Our addition: five controlled retrieval-noise conditions and a noise-aware fit, evaluated by within-condition AUROC and abstention.
4. We compare "ReDeEP as published" (fitted on clean data) with our version on 20 fixed random prompts and on the full set.
5. Scope is scaled down (one model, 150 questions) and stated honestly in the limitations.

## 14. Not verified (check, don't assume)
- Whether the authors' fork installs on current Kaggle images.
- Whether `nq_open` has passages.
- Real seconds per example; exact mirror repo names for the Llama weights.
- That the paper's published hyperparameters (O1) transfer to our prompts.
- What the mentor means exactly by "20 standard prompts".
