# Citation Hallucination Benchmark

A systematic benchmark for measuring citation hallucination rates in large language models (LLMs). The pipeline prompts models to write citation-backed academic text under controlled conditions, then automatically verifies each citation against Crossref and Semantic Scholar.

## Overview

LLMs frequently fabricate academic citations when asked to produce referenced text. This project quantifies that behavior across four models and five prompting conditions, using 144 research claims spanning six domain groups.

**Models tested:** Claude Sonnet (Anthropic), GPT-4o (OpenAI), LLaMA 3.1-8B (Meta), Qwen 2.5-14B (Alibaba)

**Conditions:**

| Condition | Description |
|-----------|-------------|
| `baseline` | Write a paragraph with 5 citations |
| `temporal` | Baseline + all papers must fall within a specified time window |
| `survey` | Write a related-work section with 8 citations |
| `privacy` (Non-Disc.) | Baseline + instruction not to rely on memorized training data |
| `combo` | Survey + temporal + privacy combined |

## Repository Structure

```
.
├── src/                        # Core library
│   ├── schema.py               # Dataclasses: Citation, MatchResult, VerificationResult
│   ├── prompts.py              # Prompt templates for 5 experimental conditions
│   ├── parser.py               # Regex-based citation block parser
│   ├── normalize.py            # Text normalization (titles, authors, venues, DOIs)
│   ├── match.py                # Fuzzy matching with weighted scoring
│   ├── label.py                # Label assignment (EXISTS / AMBIGUOUS / FABRICATED)
│   ├── aggregate.py            # Run-level metric aggregation
│   └── clients.py              # Crossref and Semantic Scholar API clients with caching
├── scripts/
│   ├── build_runs_from_claims.py   # Step 1: Generate prompts from claims CSV
│   ├── generate_runs.py            # Step 2: Call LLMs and collect outputs
│   ├── verify_runs.py              # Step 3: Verify citations against academic databases
│   └── sample_for_manual_validation.py  # Stratified sampling for human annotation
├── analysis/
│   ├── summarize_metrics.py        # Step 4: Bootstrap CIs and LaTeX table generation
│   ├── compute_dual_metrics.py     # Claim-level and citation-level metric aggregation
│   ├── export_fig2a_data.py        # Export per-claim verification fractions for Figure 2a
│   └── plot_fig2a_boxplot.py       # Generate Figure 2a boxplot from exported CSV
├── manual_validation_100.csv        # 100-citation stratified sample, human-annotated (batch 1)
├── manual_validation_200_batch2.csv # 100 additional citations, zero overlap (batch 2, pending annotation)
├── data/
│   ├── claims.csv                  # 144 research claims (input)
│   └── *_full_runs.jsonl           # Generated prompts per model
├── appendix/
│   ├── claims_candidate_pool.csv   # Full candidate pool (240 claims)
│   └── domain_counts_120.csv       # Domain distribution across 6 major fields
└── out/
    ├── *_full_runs_result.jsonl     # Raw model outputs
    ├── verify/
    │   ├── citations.jsonl          # Citation-level verification results
    │   └── run_metrics.jsonl        # Run-level aggregated metrics
    └── analysis/
        ├── summary_table.csv        # Summary statistics
        ├── summary_table.tex        # LaTeX table for paper inclusion
        ├── doi_completeness.csv     # DOI completeness rates by model and condition
        └── fig2a_frac_existing.csv  # Per-claim verification fractions (Figure 2a source data)
```

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

Reads `data/claims.csv` and generates 5 prompt variants per claim (one per condition). Output: `data/<model>_full_runs.jsonl`.

### Step 2: Generate Model Outputs

```bash
# Set API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export TOGETHER_API_KEY="..."       # for Llama
export SILICONFLOW_API_KEY="..."    # for Qwen

python scripts/generate_runs.py --topics data/gpt-4o_full_runs.jsonl --out out/gpt-4o_full_runs_result.jsonl
```

Supports `--resume` to skip completed runs and `--conditions baseline,temporal` to filter conditions.

### Step 3: Verify Citations

```bash
python scripts/verify_runs.py --runs out/gpt-4o_full_runs_result.jsonl --out_dir out/verify
```

For each citation in the model output:
1. Parse the structured citation block (title, authors, venue, year, DOI)
2. Normalize all text fields for comparison
3. Query Crossref (DOI lookup + title search) and Semantic Scholar (title search)
4. Score candidates using a weighted combination: title similarity (60%), author overlap (20%), year match (15%), venue similarity (5%)
5. Assign a label based on the best match score:
   - **EXISTS** (score >= 0.85): citation matches a real paper
   - **AMBIGUOUS** (score >= 0.60): partial match, possibly a real paper with metadata errors
   - **FABRICATED** (score < 0.60): no convincing match found
6. Flag temporal violations for time-constrained conditions

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

Computes per-model, per-condition means with 95% bootstrap confidence intervals (1000 resamples). Outputs both CSV and a LaTeX table ready for paper inclusion.

### Export Figure Data

```bash
python analysis/export_fig2a_data.py
```

Exports a tidy CSV of per-claim verification fractions (`frac_existing = #Existing / #Parsed`) for all 2,880 runs (4 models × 5 conditions × 144 claims). The output at `out/analysis/fig2a_frac_existing.csv` is the source data for the boxplot in Figure 2a of the paper.

### Plot Figure 2a

```bash
python analysis/plot_fig2a_boxplot.py \
    --in_csv  out/analysis/fig2a_frac_existing.csv \
    --out_png out/analysis/fig2a_frac_existing_boxplot.png
```

Reads the exported CSV and generates a 1×5 horizontal strip of boxplots (one per condition, four models each). Requires `matplotlib` and `pandas`.

### Manual Validation

To sample citations for human annotation:

```bash
python scripts/sample_for_manual_validation.py \
    --citations out/verify/citations.jsonl \
    --out manual_validation_100.csv \
    --n 100 \
    --seed 42
```

Performs stratified sampling by `(model, label)` to ensure representation across all models and outcome categories (EXISTS, AMBIGUOUS, FABRICATED). The output CSV includes the pipeline's automated label and blank columns for the human annotator (`human_label`, `agree`, `reasoning`).

Two batches are provided:

| File | N | Seed | Status |
|------|---|------|--------|
| `manual_validation_100.csv` | 100 | 42 | Annotated (batch 1) — 75% agreement, Cohen's κ = 0.627 |
| `manual_validation_200_batch2.csv` | 100 | 137 | Pending annotation (batch 2, zero overlap with batch 1) |

## Dataset

The benchmark uses 144 research claims balanced across 6 major academic fields (24 claims each):

| Field | Domains |
|-------|---------|
| Computer Science | AI/ML, Data/HCI, Security, Software Engineering, Systems |
| Humanities | Cultural Studies, History, Linguistics, Literature, Philosophy |
| Interdisciplinary | Digital Society, Research Ethics, STS, Sustainability, Tech Policy |
| Medicine & Health | Clinical Medicine, Epidemiology, Health Informatics, Mental Health, Pharmacology |
| Natural Sciences | Biology, Chemistry, Earth Science, Environmental Science, Physics |
| Social Sciences | Economics, Policy/Law/Education, Political Science, Psychology, Sociology |

Each claim includes a research question, domain label, suggested number of papers, a temporal window (for the `temporal` and `combo` conditions), and seed anchor keywords to scope the topic.

## Dependencies

- Python 3.9+
- `openai`, `anthropic` (LLM API clients)
- `requests` (HTTP for Crossref / Semantic Scholar)
- `rapidfuzz` (fuzzy string matching)
- `tqdm` (progress bars)
- `numpy`, `pandas` (analysis)
- `matplotlib` (figure generation)

Install:

```bash
pip install openai anthropic requests rapidfuzz tqdm numpy pandas matplotlib
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key (GPT-4o) |
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude) |
| `TOGETHER_API_KEY` | Together AI API key (Llama) |
| `SILICONFLOW_API_KEY` | SiliconFlow API key (Qwen) |
| `RESUME` | Set to `1` to enable resume mode in generate/verify scripts |

## Output Schema

### `citations.jsonl` (citation-level)

Each line contains:
- `citation_id`, `run_id`, `model`, `condition`, `topic_id`
- `label`: EXISTS, AMBIGUOUS, or FABRICATED
- `confidence`: best match score (0.0--1.0)
- `error_type`: `fabricated`, `temporal`, or `null`
- `parsed`: extracted metadata (title, authors, venue, year, DOI)
- `canonical`: matched record metadata and source (Crossref or Semantic Scholar)

### `run_metrics.jsonl` (run-level)

Each line contains all fields from the input run plus:
- `num_citations`: number of citations parsed
- `exists_rate`, `fabricated_rate`, `ambiguous_rate`: label proportions
- `temporal_violation_rate`: fraction of citations outside the specified time window

## License

This project is provided for research purposes.
