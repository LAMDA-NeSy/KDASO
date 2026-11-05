# KDASO: Knowledge-Data Dual-Driven Automated Skill Optimization for Liability Adjudication Tasks

Liability adjudication asks whether a party should bear liability for an event — a ride-hailing cancellation, a property-fee dispute, a legal charge, a logical entailment. Modern LLM adjudicators reason under a library of **domain skills**: executable descriptions of preconditions, adjudication logic and execution templates. KDASO is a framework that **optimizes the skills instead of retraining the model**. It continuously induces new skills from recently adjudicated data, refines their formulation until the LLM's execution trajectory matches the expert's reasoning path, and routes only the applicable skills into the context budget. The result is an adjudicator that keeps pace with a shifting distribution, follows expert logic faithfully, and degrades gracefully as the skill library grows — without a single gradient step.

| [📄 Paper](NeurIPS-KDASO.pdf) | [🌐 Project Page](https://lamda-nesy.github.io/KDASO/) | [🧪 Demo Dataset](data/test_data) |
|---|---|---|
| NeurIPS 2026 | Nanjing University · DiDi Chuxing | Ready to run |

---

## Overview

Skill-based LLM adjudication is attractive because reasoning is explicit and interpretable, yet it inherits three tensions from the paradigm itself (Section 3.2 of the paper):

- **Skill coverage under distribution shift.** Skill libraries are curated from historical cases and therefore lag behind emerging behaviours (e.g. GPS-spoofing "virtual displacement" fraud). Inducing a skill from one case over-fits idiosyncratic noise; inducing from the whole dataset blows up the context window.
- **Skill execution alignment.** A skill that is logical to a human expert may be executed inconsistently by an LLM: divergent priors and ambiguous operator precedence create a semantic execution gap, so the label can be right while the reasoning path is wrong. Empirically, skill description quality dominates adjudication precision (Figure 1).
- **Skill scalability under context budget.** Omission of skills correlates positively with the number of skills in the prompt (R² = 0.82, Figure 2), and precision decays with it — an effect of the selectivity of self-attention rather than of the model's knowledge.

KDASO answers each tension with one module, and the three modules form a closed loop that keeps the library accurate, compact and adaptable:

| Module | Mechanism | Guarantee |
| --- | --- | --- |
| Neighborhood-Conditioned Skill Induction | Embeds cases with a BGE encoder, retrieves the top-k (k = 4) most similar *recent* adjudicated cases as a local chart on the data manifold, and induces a skill from that neighborhood; every induced skill is then validated on real data and either kept, merged or discarded | New skills track the evolving distribution without single-case over-fitting, and induction stays far below the context limit |
| Execution-Aligned Skill Refinement | Samples G stochastic execution trajectories per skill, scores each against the human reference trajectory, and rewrites the skill's execution scaffolding for the misaligned ones (iterative, capped at 20 rounds) | Closes the semantic execution gap; abstract instructions become step-wise, evidence-grounded checking procedures |
| Context-Budgeted Skill Routing | Trains one lightweight XGBoost applicability classifier per skill on top of case embeddings, unions classifier hits with direct structural matches, then truncates to the largest set that fits the context budget | Removes skill omission and attentional interference; keeps a stable ~12 skills per case while cutting input tokens by ~73% |

The skill library therefore behaves like a versioned artefact of the deployed system: skills induced in one scenario transfer directly to concept-similar scenarios (**+8.96%**), monthly rolling optimization keeps improving performance, and **40%** of the generated skills have been accepted by DiDi's adjudication team and merged into the production skill base.

---

## Installation

```bash
git clone https://github.com/LAMDA-NeSy/KDASO.git
cd KDASO

conda create --name kdso python=3.12
conda activate kdso
pip install -r requirements.txt
```

`requirements.txt` pins the versions that matter: `torch==2.6.0`, `scikit-learn==1.5.2` (sklearn 1.6 conflicts with XGBoost 2.1.3), plus `sentence-transformers`, `xgboost`, `aiohttp`, `pandas`, `numpy`, `requests`.

> ⚠️ Every model call in this repository goes through an OpenAI-compatible `chat/completions` endpoint. No endpoint or key is hard-coded: fill in `src/setting/config.py` before running.

---

## Model Configuration

All four stages share one configuration file, `src/setting/config.py`:

```python
LLM_PATH = "your-served-model-name"                       # "model" field of the chat request
LLM_URL  = "http://<host>:<port>/v1/chat/completions"     # OpenAI-compatible endpoint

LLM_PARAMS = {
    "max_tokens": 12000,
    "temperature": 0,
    "top_k": 1,
    "top_p": 0.8,
    "repetition_penalty": 1,
    "stream": False
}

MODEL_NAME = "BAAI/bge-large-zh-v1.5"                     # embedding model for neighborhoods and routing
```

- `LLM_PATH` / `LLM_URL` / `LLM_PARAMS` are used by the generator, the refiner rollouts and the final adjudicator.
- `MODEL_NAME` is loaded by `sentence-transformers` and powers both the neighborhood construction (Stage 1) and the router features (Stage 3).

---

## Data Setup

Each dataset is a folder `./data/<dataset_name>/` holding four files:

| File | Description |
| --- | --- |
| `train.json` | Labelled adjudication cases used for skill induction and refiner rollouts |
| `test.json` | Cases to adjudicate; the same schema as `train.json` |
| `rule.json` | Seed skill library: `{"R0": "skill text", "R1": "skill text", ...}` |
| `info.json` | Column mapping, see below |

```json
{
  "SEARCH_LIST": ["context", "question"],
  "INFO_COLUMN": "context",
  "QUESTION_COLUMN": "question",
  "RULE_COLUMN": "rule",
  "ANSWER_COLUMN": "answers"
}
```

- `SEARCH_LIST` — fields concatenated into the text used for similarity matching.
- `INFO_COLUMN` / `QUESTION_COLUMN` — case context and the adjudication prompt shown to the LLM.
- `RULE_COLUMN` — the skill name a training case belongs to (used to label positives for each router).
- `ANSWER_COLUMN` — ground-truth label of a training case.

`./data/test_data` is the shipped demonstrator (9 training cases, 2 test cases, 2 seed skills `R0`/`R1`), taken from a logical-reasoning style adjudication task. The ride-hailing, LawBench, LegalBench, ReClor and AR-LSAT datasets used in the paper are not redistributable — download them from their official sources and convert them into the four-file layout above.

---

## Reproducing the Pipeline

```bash
python run_experiment.py --experiment_name test_data --dataset_name test_data
```

`run_experiment.py` runs the four stages in order and writes every artefact into `./result/{experiment_name}/`:

| # | Stage | Implementation | Produces |
| --- | --- | --- | --- |
| 1 | Neighborhood-Conditioned Skill Induction | `src/rule_generator.py` | `generate_guidance.json` (retrieved neighbors + raw output), `generate_rule.json` (skill library with the newly induced skills appended as `Rn`) |
| 2 | Execution-Aligned Skill Refinement | `src/rule_refiner.py`, `src/experience.py` | `rule_refiner/stats.json` and `rule_refiner/step_*/{shuffled_data.jsonl, rollout.jsonl, single_rollout_summary.json, single_query_critique.json, batch_update.json, experiences.json}` |
| 3 | Context-Budgeted Skill Routing | `src/rule_retriever.py` | `final_rule_base.json`, `rule_retriever/predictions.csv` (case × skill 0/1 matrix), `rule_retriever/answer.csv` (per-skill validation precision/recall), `rule_retriever/model/*.pkl` |
| 4 | Liability Adjudication | `src/run.py` | `final_output.json` — per test case: original record, constructed prompt, raw model output |

**Refinement loop in one glance.** The refiner seeds `step_0/experiences.json` with the current library (`result/{experiment_name}/kdaro_rule.json` if a previous induction run left one, otherwise `data/{dataset_name}/rule.json`), then walks the training set in batches: `grpo_n` rollouts per case are sampled, `<result>` is verified against the label, the trajectories are summarised, a critique proposes `add` / `modify` / `merge` operations, and the updated experience repository is written to `step_{n+1}/experiences.json`. Because the router consumes the *last* step, the stage always hands the most refined library to Stage 3.

> The released router exposes the full case × skill applicability matrix. The paper's final assembly step additionally unions structural matches and truncates to the context budget; in this code base that selection is left to the deployment, so the matrix can be re-thresholded without retraining.

**Resume for free.** `shuffled_data.jsonl`, `rollout.jsonl`, `single_rollout_summary.json`, `single_query_critique.json`, `batch_update.json`, `experiences.json` and `stats.json` are all cached and re-used, so an interrupted run (or an expensive rollout) never has to be repeated.

### Command-Line Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--experiment_name` | str | `test_data<timestamp>` | Output directory under `./result/` |
| `--dataset_name` | str | `test_data` | Folder under `./data/` |
| `--generator_top_k` | int | `4` | Neighbors per induced skill (paper: k = 4 is optimal, k ≥ 6 overflows the context) |
| `--refiner_epochs` | int | `2` | Passes over the annotated training set |
| `--refiner_batchsize` | int | `4` | Cases per refinement step |
| `--refiner_truncate` | int | `300` | Keep only the first N training cases |
| `--refiner_grpo_n` | int | `2` | Rollouts sampled per case |
| `--refiner_concurrency` | int | `128` | Concurrent rollout requests (dominant GPU-memory knob) |
| `--refiner_temperature` | float | `0.7` | Sampling temperature of the rollouts |
| `--refiner_max_tokens` | int | `5172` | Generation budget per rollout |
| `--refiner_ground_truth` | str | `"True"` | Feed ground-truth labels into the critique |
| `--true_sample_ratio` | int | `1` | Negative:positive ratio when assembling each router's training set |
| `--use_clustering` | flag | clustering **on** | Store-false flag: *omit* to keep KMeans-centred negative sampling, *pass* to switch to random negatives |
| `--skip_generator` / `--skip_refiner` / `--skip_retriever` | flag | off | Reuse the artefacts of an earlier run |
| `--stop_on_error` | flag | off | Abort instead of continuing with the remaining stages |

> `refiner_concurrency` and `refiner_batchsize` dominate GPU memory: lower them first if a rollout wave exhausts the server.

---

## Benchmarks

KDASO is evaluated on three real-world adjudication domains:

| Domain | Tasks | Source |
| --- | --- | --- |
| Ride-hailing adjudication | Cancel–Appeal (CA), Cancel–Reservation (CR), Cancel–Passenger-cancel (CPC), Cancel–Driver-cancel (CDC), Fee–Detour (FD) | Five closed-source DiDi datasets (relatively hard cases) |
| Legal liability determination | LA-ITI, LA-CP; LE-OPP, LE-UN, LE-LH | LawBench, LegalBench |
| Logical reasoning judgment | ReClor, AR-LSAT (Test / Test-E / Test-H) | ReClor, AR-LSAT |

Compared systems span four families: **proprietary LLMs** (Gemini 3 Pro, GPT-4o); **liable-adjudicated general-purpose algorithms** (NL2FOL, NeoBERT, ModernBERT, SIM-RAG, NodeRAG); **domain-specific models** (Lawyer-LLAMA, LexiLaw, ChatLaw; MERIt, LReasoner, IDOL, AMR-LDA); and **open-source general LLMs** (DeepSeek-R1, Qwen3-8B, QwQ-32B), the last of which also serve as KDASO backbones. Ride-hailing is scored with ordinary-liability and bad-faith precision/recall (O-Precision, O-Recall, B-Precision, B-Recall); legal and logic tasks use the official benchmark metrics.

---

## Repository Layout

```
KDASO/
├── run_experiment.py          End-to-end driver: generator → refiner → retriever → adjudication
├── example.sh                  Ready-to-copy commands
├── requirements.txt
├── data/test_data/             Demonstrative dataset
│   ├── train.json  test.json  rule.json  info.json
├── src/
│   ├── rule_generator.py       Stage 1 — neighborhood-conditioned skill induction
│   ├── rule_refiner.py         Stage 2 — execution-aligned skill refinement (rollout loop)
│   ├── experience.py           Trajectory summary → critique → experience-repository update
│   ├── rule_retriever.py       Stage 3 — per-skill applicability routers + prediction matrix
│   ├── run.py                  Stage 4 — adjudication under the routed skill subset
│   └── setting/
│       ├── config.py           Endpoint, model names, embedding model
│       └── prompts.py          GENERATE_PROMPT, PROBLEM_WITH_EXPERIENCE_TEMPLATE, REFINE_PROMPT_1/2
├── evaluation/                 Legalbench.py, AR-LSAT.py, Reclor.py
├── result/                     Experiment artefacts (created at runtime)
├── image/KDASO.jpg             Framework figure
├── index.html  styles.css      Project page
└── NeurIPS-KDASO.pdf           Paper
```

---

## Evaluation

- **LawBench** — evaluate with the official task code: <https://github.com/open-compass/LawBench>
- **LegalBench** — `python evaluate/Legalbench.py` (update the input/output paths at the top of the script)
- **AR-LSAT** — `python evaluate/AR-LSAT.py` (update the input/output paths)
- **ReClor** — `python evaluate/Reclor.py` converts the JSON output into a `.npy` prediction file; submit it to <https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347>

The scripts expect the `final_output.json` produced by Stage 4, flattened into a data frame with an `all_answer` column; they parse `<result>`, `<rule>` and `<analysis>` style tags, tolerate `</think>` prefixes, and fall back through several regexes before counting an extraction failure.

---

## Extending KDASO

1. **New dataset.** Create `./data/<name>/` with `train.json`, `test.json`, `rule.json`, `info.json` (see [Data Setup](#data-setup)), then run `python run_experiment.py --dataset_name <name>`.
2. **New LLM backbone.** Edit `LLM_PATH`, `LLM_URL`, `LLM_PARAMS` in `src/setting/config.py`. Any OpenAI-compatible server works — the code only reads `choices[0].message.content`.
3. **New embedding model.** Edit `MODEL_NAME` in `src/setting/config.py` (paper default: `BAAI/bge-large-zh-v1.5`).
4. **New router.** Replace `XGBClassifier()` in `src/rule_retriever.py` with your own applicability classifier; only `fit` / `predict` / `joblib` serialisation are assumed.
5. **New prompts.** `GENERATE_PROMPT`, `PROBLEM_WITH_EXPERIENCE_TEMPLATE`, `REFINE_PROMPT_1` and `REFINE_PROMPT_2` live in `src/setting/prompts.py`. `GENERATE_PROMPT` drives skill induction (`src/rule_generator.py`), `PROBLEM_WITH_EXPERIENCE_TEMPLATE` drives the refiner rollouts (`src/rule_refiner.py`), and the two `REFINE_PROMPT`s wrap the adjudication instructions used by both the refiner and `src/run.py`.

---

## Citation

```bibtex
@inproceedings{cheng2026kdaso,
  title     = {KDASO: Knowledge-Data Dual-Driven Automated Skill Optimization for Liability Adjudication Tasks},
  author    = {Cheng, Zi-Jian and Wu, Weiming and Meng, Jie and Peng, Zhen and Huang, Shan and Li, Qun and Wu, Guobing and Guo, Lan-Zhe},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
