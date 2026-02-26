# Do Deployment Constraints Make LLMs Hallucinate Citations?

Replication package for the paper:

> **Do Deployment Constraints Make LLMs Hallucinate Citations? An Empirical Study across Four Models and Five Prompting Regimes**
>
> Submitted to [EASE 2026](https://conf.researchr.org/home/ease-2026) (Short Papers and Emerging Results track)

## Overview

Large language models frequently fabricate academic citations—generating references that look plausible but do not correspond to real scholarly works. This project provides a curated claim dataset, an automated three-way citation verification framework, and all experimental artifacts needed to study how **deployment constraints** affect citation hallucination across proprietary and open-source models.

We evaluate four LLMs under five prompting conditions using deterministic decoding (temperature = 0), with no retrieval augmentation or tool use—all citations are generated entirely from parametric knowledge. The verification pipeline classifies each citation as **Existing**, **Ambiguous**, or **Fabricated** by querying Crossref and Semantic Scholar.

### Key Findings

- **No model verifies a majority of its citations** under any condition (best: 47.5%).
- **Temporal constraints** cause the steepest decline in verifiability, while models maintain surface compliance with the year window (violation rates ≤ 2.7%).
- **Survey-style prompting** widens the gap between proprietary and open-source models.
- **Privacy instructions** shift errors from fabrication into harder-to-detect ambiguity.
- **Ambiguous citations dominate** across all conditions (43–61%), a category most prior work does not distinguish.

### Models

| Short Name | Model Identifier | Type |
|------------|-----------------|------|
| Claude Sonnet | `claude-sonnet-4-5-20250929` | Proprietary (Anthropic) |
| GPT-4o | `gpt-4o` | Proprietary (OpenAI) |
| LLaMA 3.1-8B | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | Open-source (Meta) |
| Qwen 2.5-14B | `Qwen/Qwen2.5-14B-Instruct` | Open-source (Alibaba) |

### Prompting Conditions

| Condition | Description | Citations |
|-----------|-------------|-----------|
| `baseline` | Write one academic paragraph with citations | 5 |
| `temporal` | Baseline + all cited papers must fall within a specified publication-year window | 5 |
| `survey` | Write a related-work section organized into 3–4 approach categories | 8 |
| `privacy` | Baseline + instruction not to reveal or rely on memorized training documents | 5 |
| `combo` | Survey + temporal + privacy combined | 8 |

All conditions use deterministic decoding (temperature = 0) so that observed differences reflect the prompting condition and the model, not sampling variability.

## Repository Structure

```
.
├── src/                        # Core library
│   ├── schema.py               # Dataclasses: Citation, MatchResult, VerificationResult
│   ├── prompts.py              # Prompt templates for all 5 experimental conditions
│   ├── parser.py               # Regex-based citation block parser
│   ├── normalize.py            # Text normalization (titles, authors, venues, DOIs)
│   ├── match.py                # Fuzzy matching with weighted scoring (Eq. 1 in paper)
│   ├── label.py                # Three-way label assignment (Existing / Ambiguous / Fabricated)
│   ├── aggregate.py            # Claim-level metric aggregation
│   ├── clients.py              # Crossref and Semantic Scholar API clients with JSONL caching
│   └── __init__.py
├── scripts/
│   ├── build_runs_from_claims.py       # Step 1: Generate prompts from claims CSV
│   ├── generate_runs.py                # Step 2: Call LLMs and collect outputs
│   ├── verify_runs.py                  # Step 3: Verify citations against scholarly databases
│   └── sample_for_manual_validation.py # Sample citations for human annotation
├── analysis/
│   └── summarize_metrics.py    # Step 4: Bootstrap CIs and LaTeX table generation
├── data/
│   ├── claims.csv              # 144 research claims across 30 subdomains (input)
│   ├── claude-sonnet-4-5-20250929_full_runs.jsonl
│   ├── gpt-4o_full_runs.jsonl
│   ├── llama3_8b_full_runs.jsonl
│   └── qwen2_5_14b_full_runs.jsonl
├── out/
│   ├── *_full_runs_result.jsonl        # Raw model outputs (720 runs per model)
│   ├── verify/
│   │   ├── citations.jsonl             # Citation-level verification results
│   │   └── run_metrics.jsonl           # Claim-level aggregated metrics
│   └── analysis/
│       ├── summary_table.csv           # Summary statistics with bootstrap CIs
│       └── summary_table.tex           # LaTeX table for paper inclusion
├── appendix/
│   ├── claims_candidate_pool.csv       # Full candidate pool before sampling (240 prompts)
│   └── domain_counts_120.csv           # Domain distribution reference
├── manual_validation_100.csv           # 100 human-annotated citations for pipeline validation
└── README.md
```

## Claim Dataset

The benchmark uses **144 question-style prompts** spanning 6 major academic fields and 30 subdomains. Claims are phrased as information-seeking questions (e.g., "What evidence supports…", "How do methods compare…") rather than declarative statements, aligning with survey-style scholarly writing and allowing multiple valid citation sets per prompt.

**Construction methodology:** We built a candidate pool of **240 prompts** from publicly accessible academic materials (textbook headings, survey section titles, syllabus topics), pre-screened for suitability, stratified by domain, and randomly sampled the target 144 claims using a fixed seed to reduce selection bias.

Each claim record includes a research question, domain label, optional temporal window (for `temporal` and `combo` conditions), and optional seed anchors (keyword-only topic hints that the model is explicitly instructed not to cite).

| Field | Subdomains | Claims |
|-------|-----------|--------|
| Computer Science | AI/ML, Data/HCI, Security, Software Engineering, Systems | 24 |
| Humanities | Cultural Studies, History, Linguistics, Literature, Philosophy | 24 |
| Interdisciplinary | Digital Society, Research Ethics, STS, Sustainability, Tech Policy | 24 |
| Medicine & Health | Clinical Medicine, Epidemiology, Health Informatics, Mental Health, Pharmacology | 24 |
| Natural Sciences | Biology, Chemistry, Earth Science, Environmental Science, Physics | 24 |
| Social Sciences | Economics, Policy/Law/Education, Political Science, Psychology, Sociology | 24 |

## Pipeline

The pipeline has four stages, each implemented as a standalone script:

```
claims.csv ──▶ build_runs_from_claims.py ──▶ generate_runs.py ──▶ verify_runs.py ──▶ summarize_metrics.py
   (144           (720 prompts/model)        (LLM responses)     (citation-level      (bootstrap CIs,
   claims)                                                        verification)         LaTeX table)
```

### Step 1: Build Prompts

```bash
python scripts/build_runs_from_claims.py
```

Reads `data/claims.csv` and generates 5 prompt variants per claim (one per condition), producing 720 prompts per model. Prompt templates are defined in `src/prompts.py`. Output: `data/<model>_full_runs.jsonl`.

### Step 2: Generate Model Outputs

```bash
# Set API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export TOGETHER_API_KEY="..."       # for LLaMA (Together AI)
export SILICONFLOW_API_KEY="..."    # for Qwen (SiliconFlow)

python scripts/generate_runs.py --topics data/gpt-4o_full_runs.jsonl --out out/gpt-4o_full_runs_result.jsonl
```

Calls each LLM with the rendered prompt at temperature 0. Supports `--resume` to skip completed runs and `--conditions baseline,temporal` to filter conditions.

### Step 3: Verify Citations

```bash
python scripts/verify_runs.py --runs out/gpt-4o_full_runs_result.jsonl --out_dir out/verify
```

For each citation in the model output:

1. **Parse** the structured citation block into fields: title, authors, venue, year, DOI.
2. **Normalize** all text fields for comparison (lowercasing, venue alias resolution, DOI cleaning).
3. **Retrieve candidates** from Crossref (DOI lookup + title search, *k* = 5) and Semantic Scholar (title search, *k* = 5).
4. **Score** each candidate against the parsed citation using a weighted combination:

   *s* = 0.60 · *t* + 0.20 · *a* + 0.15 · *y* + 0.05 · *v*

   where *t* = fuzzy title similarity (token-set ratio), *a* = author last-name overlap, *y* = year agreement (1.0 exact, 0.5 if off by one, 0 otherwise), *v* = venue similarity (partial ratio).

5. **Label** based on the best-match score:
   - **Existing** (score ≥ 0.85): high-confidence match to a real paper
   - **Ambiguous** (0.60 ≤ score < 0.85): partial or conflicting evidence
   - **Fabricated** (score < 0.60, or no candidate found): no convincing match

6. **Flag temporal violations** for time-constrained conditions (citations whose year falls outside the prompt-specified window).

The existence threshold is set at 0.85 rather than 1.0 because even genuine citations rarely achieve perfect metadata agreement across databases, due to variations in venue naming, author transliterations, and preprint-vs-final year discrepancies.

To verify all model outputs at once, omit `--runs`:

```bash
python scripts/verify_runs.py --out_dir out/verify --clear
```

### Step 4: Summarize Results

```bash
python analysis/summarize_metrics.py \
    --in_jsonl out/verify/run_metrics.jsonl \
    --out_csv out/analysis/summary_table.csv \
    --out_tex out/analysis/summary_table.tex
```

Computes per-model, per-condition means with 95% bootstrap confidence intervals (1,000 resamples, fixed seed). Outputs both CSV and a LaTeX table formatted for ACM sigconf.

**Reported metrics** (per claim; existence, fabrication, and ambiguity rates sum to 1):

| Metric | Description |
|--------|-------------|
| Existence Rate | Proportion of claims where all citations verified as Existing |
| Fabrication Rate | Proportion of claims with at least one Fabricated citation |
| Ambiguity Rate | Remaining claims (at least one Ambiguous, no Fabricated) |
| Temporal Violation Rate | Citations whose year falls outside the prompt-specified window |
| Avg. #Citations | Mean number of citations generated per claim |

## Manual Validation

To assess pipeline reliability, we sampled 100 citations stratified by model and label for manual human annotation.

### Sampling

```bash
python scripts/sample_for_manual_validation.py \
    --citations out/verify/citations.jsonl \
    --out manual_validation_100.csv \
    --n 100 \
    --seed 42
```

The script uses proportional stratified sampling across all (model, label) strata.

### Agreement Results

Overall pipeline–human agreement: **75%** (75/100).

| Pipeline Label | Agreement | Rate |
|---------------|-----------|------|
| Existing | 31/32 | 96.9% |
| Fabricated | 29/33 | 87.9% |
| Ambiguous | 15/35 | 42.9% |

The high agreement on Existing and Fabricated labels confirms the pipeline's reliability at the extremes. The lower agreement on Ambiguous citations is expected—these are inherently borderline cases where metadata partially matches, making them difficult for both automated systems and human reviewers to classify definitively.

### Annotation Schema

| Column | Description |
|--------|-------------|
| `pipeline_label` | Automated label (EXISTS, AMBIGUOUS, FABRICATED) |
| `pipeline_score` | Best-match confidence score |
| `human_label` | Manual annotation by human reviewer |
| `agree` | Whether pipeline and human labels match |
| `reasoning` | Annotator's justification for the human label |

## Output Schema

### `citations.jsonl` (citation-level)

Each line is a JSON object containing:

| Field | Description |
|-------|-------------|
| `citation_id` | Unique identifier (`{run_id}_c{index}`) |
| `run_id` | Parent run identifier |
| `model`, `condition`, `topic_id` | Experimental metadata |
| `label` | EXISTS, AMBIGUOUS, or FABRICATED |
| `confidence` | Best-match score (0.0–1.0) |
| `error_type` | `fabricated`, `temporal`, or `null` |
| `parsed` | Extracted metadata: title, authors, venue, year, DOI |
| `canonical` | Matched record metadata and source (Crossref or Semantic Scholar) |

### `run_metrics.jsonl` (claim-level)

Each line contains all fields from the input run plus:

| Field | Description |
|-------|-------------|
| `num_citations` | Number of citations parsed from the model output |
| `exists_rate` | Proportion labeled Existing |
| `fabricated_rate` | Proportion labeled Fabricated |
| `ambiguous_rate` | Proportion labeled Ambiguous |
| `temporal_violation_rate` | Proportion of citations outside the specified time window |

## Dependencies

- Python 3.9+
- `openai`, `anthropic` — LLM API clients
- `requests` — HTTP for Crossref / Semantic Scholar
- `rapidfuzz` — fuzzy string matching
- `tqdm` — progress bars
- `numpy`, `pandas` — analysis and bootstrap CIs

```bash
pip install openai anthropic requests rapidfuzz tqdm numpy pandas
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o) |
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude Sonnet) |
| `TOGETHER_API_KEY` | Together AI API key (LLaMA 3.1-8B) |
| `SILICONFLOW_API_KEY` | SiliconFlow API key (Qwen 2.5-14B) |
| `RESUME` | Set to `1` to enable resume mode in generate/verify scripts |

## Reproducibility

All model outputs, verification results, and analysis artifacts are included in this repository. To reproduce from scratch:

1. **Build prompts:** `python scripts/build_runs_from_claims.py`
2. **Generate outputs:** Run `scripts/generate_runs.py` for each model (requires API keys)
3. **Verify citations:** `python scripts/verify_runs.py --out_dir out/verify --clear`
4. **Compute statistics:** `python analysis/summarize_metrics.py --in_jsonl out/verify/run_metrics.jsonl --out_csv out/analysis/summary_table.csv --out_tex out/analysis/summary_table.tex`

Note: Steps 2–3 require API access and will incur costs. The pre-computed outputs in `out/` allow steps 3–4 to be run without regenerating model responses.

## Data Availability

This repository is released in compliance with the [SIGSOFT Open Science Policies](https://github.com/acmsigsoft/open-science-policies). The claim dataset, prompt templates, verification pipeline source code, and all raw results files are included.

## License

This project is provided for research purposes.
