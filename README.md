# Knowing When Not to Answer

**Predicting RAG failure from LLM internal states under retrieval noise** — CS787 course project. An implementation and stress test of ReDeEP (Sun et al., ICLR 2025) on Llama-2-7B-chat.

**Team:** Sarthak Agrawal · Arpit Singh · Arsh Chauhan

## Deliverables
| What | Where |
|---|---|
| **Project report (PDF, 9 pages)** | [`CS787_Project_Report.pdf`](CS787_Project_Report.pdf) |
| **Detailed results report (Markdown, with appendices)** | [`REPORT.md`](REPORT.md) |
| **Presentation slides (PPTX)** | [`presentation/Project_Presentation_Final.pptx`](presentation/Project_Presentation_Final.pptx) |
| **Video presentation (Google Drive)** | https://drive.google.com/file/d/1AtW8b7WiF4AgIZwVRnCGGNL69KE_1wB7/view?usp=sharing |
| Plan followed | [`FINAL_PLAN_V3.md`](FINAL_PLAN_V3.md) · changes from the plan: [`DEVIATIONS.md`](DEVIATIONS.md) |

## External resources
- Base paper code (ReDeEP): https://github.com/Jeryi-Sun/ReDEeP-ICLR (commit `4d08191`)
- Data: SQuAD 2.0 — https://huggingface.co/datasets/rajpurkar/squad_v2 (RAGTruth features from the ReDeEP repo for the reproduction)
- Model: `meta-llama/Llama-2-7b-chat-hf`, revision `f5db02db724555f92da89c216ac04704f23d4590`
- Kaggle (GPU runs): dataset `acinaction/rag-noise-probe-v2`, notebooks `rag-noise-probe-b-full-run`, `rag-noise-probe-b2-resume`, `rag-noise-probe-a-prerun-validation`, `rag-noise-probe-a2-extraction-check` (the notebook sources are in this repo: `kaggle_*.ipynb`)

## Headline results (Llama-2-7B-chat, 150 questions × 5 retrieval conditions = 750 prompts)
- Noise raises the failure rate from 10.7% (gold passage) to 21.3% (hard negatives), 32.7% (counterfactual) and 98.0% (missing evidence).
- Within each condition both ReDeEP versions (fitted on clean context, or on all conditions) are at chance: AUROC 0.46–0.54, all 95% intervals include 0.5.
- Probes and simple uncertainty baselines do better in places; pooled metrics mostly reward recognising missing evidence (the condition label alone reaches AUROC 0.82).
- Reproduction of the paper on its released data: AUC 0.7473 / PCC 0.4207 vs 0.7458 / 0.4203.

## Demo (no GPU needed)
`python demo.py` replays the saved results for one question: the passages in each of the 5 conditions, Llama-2's answer with the standard prompt and with the abstain-allowed prompt, whether it was right, and what each detector (ReDeEP original / noise-aware, probe, LN-entropy) decided (answer or abstain).
```bash
python demo.py                      # the Amazon-drought example, all 5 conditions
python demo.py --list               # the 20 comparison questions
python demo.py --pick 3             # question 3 from that list
python demo.py --pick 3 --cond C4_missing
```
It does not call the model live: generation needs a GPU and the gated Llama-2 weights (see the Kaggle notebooks).

## Repository layout
```
src/data/        benchmark builder (noise_builder.py), prompts, labelling (evaluator.py), hand-check tools
src/gen/         generation (gen.py), regeneration + validation (finalize_gen.py), model identity check
src/extract/     ECS / PKS / hidden-state extraction with forward hooks (redeep_extract.py)
src/detect/      ReDeEP fitting (fit_redeep.py), LR/MLP, probes, baselines, nested grouped CV
src/analysis/    tables, figures, 20-prompt comparison, appendices
src/repro/       E0 reproduction of the paper
src/run_cpu_pipeline.py   one command that runs every CPU step on the extracted features
tests/           30 unit tests
artifacts/data/  frozen benchmark v2 (750 rows), folds, 20-prompt set, config, checksums (FROZEN_v2.json)
artifacts/runs/  generations, labels and extracted features of the full run
artifacts/results/llama2-7b/   all tables, figures and comparison files
kaggle_*.ipynb   the notebooks that were run on Kaggle (2 × T4)
```

## Reproduce
```bash
pip install -r requirements.txt   # or: pip install torch transformers pandas pyarrow scikit-learn rank_bm25 matplotlib tabulate datasets
python -m pytest tests -q                                   # 30 tests
python -m src.run_cpu_pipeline --run artifacts/runs/llama2-7b --skip_label   # tables, figures, comparison (CPU, minutes)
```
The GPU steps (generation and feature extraction) are in `kaggle_B_full_run.ipynb` and `kaggle_B2_resume.ipynb` and need access to the gated Llama-2 weights. The authors' ReDeEP repository is only needed for the reproduction (E0) and the head list `topk_heads.json`, which is included under `third_party/`.

## Notes
- No language model was trained: Llama-2 is used frozen for inference; only the small detectors (ReDeEP score, logistic regression, MLP, probes) are fitted, on CPU.
- Compute: generation took about 72 minutes and feature extraction about 10 minutes on two NVIDIA T4 GPUs (Kaggle); total notebook time about 1.5 hours.
- Results under `artifacts/results/mock/` were synthetic test data and are not included.
