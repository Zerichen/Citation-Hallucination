# Citation Hallucination Benchmark

A systematic benchmark for measuring **citation hallucination** in large language models (LLMs) under deployment-motivated prompting constraints. The pipeline:

1. Prompts four LLMs (Claude Sonnet 4.5, GPT-4o, LLaMA 3.1-8B, Qwen 2.5-14B) to write citation-backed academic text on 144 question-style claims spanning six domains, under five regimes (Baseline / Temporal / Survey / Non-Disclosure / Combo).
2. Parses each generated reference and verifies it against **Crossref** and **Semantic Scholar** with a deterministic three-way label scheme:
   - **Existing** (`EXISTS`): a real, indexed paper matches.
   - **Unresolved** (`AMBIGUOUS`): partial match; cannot be confirmed without manual review.
   - **Fabricated** (`FABRICATED`): no plausible match found.
3. Ships a minimal **retrieval-augmented (RAG) baseline** for Claude Sonnet on Baseline + Temporal cells, to probe how much of the closed-book failure is recoverable by grounding (added for the DeLTA 2026 camera-ready in response to Reviewer R5).
4. Releases the verification logic as a small installable Python package, **`citecheck`**, with a CLI and a programmatic API.

📄 Paper: *Do Deployment Constraints Make LLMs Hallucinate Citations? An Empirical Study Across Four Models and Five Prompting Regimes* (DeLTA 2026).

---

## Quick start

```bash
git clone https://github.com/Zerichen/Citation-Hallucination
cd Citation-Hallucination
pip install -e .

# Verify a JSONL file of citation records
citecheck verify examples/sample_references.jsonl --output out/example_results.jsonl
```

Each input line is a JSON object of the form:

```json
{"title": "...", "authors": ["..."], "venue": "...", "year": 2024, "doi": "..."}
```

The output JSONL contains the assigned label, the best-match score, and the canonical record retrieved from Crossref or Semantic Scholar.

### Python API

```python
from citecheck import verify_citation

result = verify_citation({
    "title": "Attention Is All You Need",
    "authors": ["Vaswani"],
    "venue": "NeurIPS",
    "year": 2017,
})
# result.label = "EXISTS", result.confidence = 0.97
```

---

## Install

```bash
pip install -e .
```

Requires Python 3.9+. The base install pulls only the verification dependencies (`requests`, `rapidfuzz`, `tqdm`, `numpy`, `pandas`). Optional extras:

| Command | Adds | Needed for |
|---|---|---|
| `pip install -e .` | (core only) | `citecheck verify`, `scripts/verify_runs.py`, most of `analysis/` |
| `pip install -e ".[dev]"` | `pytest` | `pytest tests/` |
| `pip install -e ".[llm]"` | `openai`, `anthropic` | `scripts/generate_runs.py`, `scripts/run_rag.py` (LLM calls) |
| `pip install -e ".[dev,llm]"` | both | full reproduction + tests |

---

## How verification works

### Scoring formula (`src/citecheck/match.py`)

Each parsed citation is scored against every candidate retrieved from Crossref / Semantic Scholar:

```
s = 0.60 · title_similarity        (token-set ratio)
  + 0.20 · author_overlap          (last-name set overlap)
  + 0.15 · year_match              (1.0 exact, 0.5 off-by-one, 0 else)
  + 0.05 · venue_similarity        (partial ratio)
```

Title gets the highest weight because it is the most discriminative field in practice. Weights are defined in [`src/citecheck/match.py`](src/citecheck/match.py).

### Label thresholds (`src/citecheck/label.py`)

```python
EXISTS_TH = 0.85   # ≥ 0.85 → Existing
AMBIG_TH  = 0.60   # 0.60 ≤ s < 0.85 → Unresolved (AMBIGUOUS)
                   # s < 0.60 or no candidate → Fabricated
```

To **re-run the labeling with different thresholds** (e.g., to reproduce the threshold-sensitivity analysis in §5 of the paper), use the dedicated script:

```bash
python scripts/threshold_sensitivity.py
#   → out/analysis/threshold_sensitivity.json
```

It re-labels the existing per-citation scores in `out/verify/citations.jsonl` under four perturbations and recomputes model rankings + paired-bootstrap Δ's — no API calls or LLM re-runs needed. The four perturbations reported in the paper are:

| Perturbation | `EXISTS_TH` | `AMBIG_TH` | What it tests |
|---|---|---|---|
| (a) | 0.80 | 0.60 | More permissive Existing cut |
| (b) | 0.90 | 0.60 | More restrictive Existing cut |
| (c) | 0.85 | 0.65 | Shrink the Unresolved band |
| (d) | 0.85 | 0.55 | Widen the Unresolved band |

Result (from the paper): 0 sign flips across 84 perturbation-comparisons; 17 of 18 statistically meaningful Δ's preserve significance.

### Naming note

The data files use `EXISTS` / `AMBIGUOUS` / `FABRICATED`; the paper uses *Existing* / *Unresolved* / *Fabricated*. They refer to the same label classes.

---

## Closed-book experiment (full reproduction)

The five-regime, four-model study from §3–§5 of the paper.

### 1. Generate model outputs

```bash
# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Build the prompts for each (claim, condition) pair
python scripts/build_runs_from_claims.py --claims data/claims.csv

# Call the LLMs (writes out/<model>_full_runs.jsonl)
python scripts/generate_runs.py
```

Each run uses temperature 0, max_tokens 2048, and one prompt template per condition. The full design yields **144 × 5 × 4 = 2,880 runs producing 17,443 individual citations**.

### 2. Verify citations

```bash
python scripts/verify_runs.py
```

Output:
- `out/verify/citations.jsonl` — per-citation labels with best-match scores and supporting evidence
- `out/verify/run_metrics.jsonl` — per-run aggregates (existence rate, fabrication rate, etc.)

### 3. Aggregate + analyze

```bash
python analysis/summarize_metrics.py
python analysis/compute_dual_metrics.py
python analysis/domain_metrics.py
python analysis/plot_fig2a_boxplot.py
```

Outputs land in `out/analysis/` (summary tables, per-domain breakdowns, boxplot data).

---

## Retrieval-Augmented (RAG) baseline

A minimal RAG comparison added for the DeLTA 2026 camera-ready — Reviewer R5 asked for a grounded baseline to probe how much of the closed-book failure is recoverable through retrieval.

**Scope (intentionally minimal):**

| Axis | Closed-book | RAG |
|---|---|---|
| Models | Claude Sonnet 4.5, GPT-4o, LLaMA 3.1-8B, Qwen 2.5-14B | Claude Sonnet 4.5 only |
| Conditions | 5 regimes | Baseline + Temporal only |
| Claims | 144 | 144 (same set) |
| Retrieval | none | Crossref top-5 by claim-keyword query (year-windowed under Temporal) |
| Verification | existing pipeline | same pipeline, unchanged |

### How `run_rag.py` works

1. **Replays the same 144 claims** the closed-book study used (read back from `out/verify/run_metrics.jsonl`).
2. **Builds a Crossref query** from each claim's `seed_anchors` (or falls back to the question keywords).
3. **Fetches top-5 records** from `https://api.crossref.org/works` (free, no auth; sends `mailto` parameter as a polite identifier). Year-filtered under the Temporal condition.
4. **Caches the raw Crossref responses** under `cache/crossref_rag/queries.jsonl` so the experiment can be reproduced offline.
5. **Injects the 5 records** into the prompt before the citation-format block — everything else (system prompt, output schema, temperature 0, max_tokens 2048) is **byte-identical to closed-book**.
6. **Writes outputs** in the same JSONL schema as the closed-book runs, plus a `retrieval` sidecar with the candidate records used.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_rag.py
#   → out/rag/claude-sonnet-4-5-20250929_rag_runs.jsonl   (288 runs)
#   → out/rag/run_config.json                              (reproducibility metadata)
#   → cache/crossref_rag/queries.jsonl                     (cached responses)
```

### Verify the RAG runs

The same verification pipeline that handles closed-book also handles RAG outputs:

```bash
python scripts/verify_runs.py --input out/rag/claude-sonnet-4-5-20250929_rag_runs.jsonl \
                              --output out/verify/rag_citations.jsonl
#   → 1,359 verified citations (out of 288 runs × ~5 cits/run)
```

### Analyze RAG vs. closed-book

```bash
python scripts/analyze_rag.py
#   → out/rag/deliverables/rag_metrics.json    (cell rates + paired bootstrap Δs)
#   → out/rag/deliverables/rag_results.md      (interpretation + tables)
#   → out/rag/deliverables/rag_table_row.tex   (LaTeX rows for Table 3)
#   → out/rag/deliverables/rag_paragraph.tex   (LaTeX paragraph for §5)
```

### Headline numbers (from the paper)

| Cell | Existing ↑ | Fabricated ↓ | Δ vs. closed-book |
|---|---|---|---|
| Claude closed-book Baseline | 0.381 | 0.157 | — |
| **Claude + RAG Baseline** | **0.699** | **0.054** | **+0.318 [0.254, 0.375]** |
| Claude closed-book Temporal | 0.119 | 0.347 | — |
| **Claude + RAG Temporal** | **0.818** | **0.008** | **+0.699 [0.649, 0.746]** |

Two caveats bound the interpretation:
1. The injected candidates are by construction real Crossref records, so part of the gain reflects faithful reuse of supplied references rather than improved unaided recall.
2. 5 of 288 RAG runs abstained (the model explicitly declined when no candidate looked adequate); counted as zero-existing — the conservative choice.

---

## Repository structure

```
Citation-Hallucination/
├── src/citecheck/                # The installable package
│   ├── prompts.py                # Prompt templates (closed-book + RAG variant)
│   ├── clients.py                # Anthropic / OpenAI client wrappers + caching
│   ├── verify.py                 # Top-level verify() entrypoint
│   ├── label.py                  # ⭐ EXISTS_TH / AMBIG_TH thresholds
│   ├── match.py                  # ⭐ Scoring formula (0.6 / 0.2 / 0.15 / 0.05)
│   ├── parser.py                 # Citation parsing from raw model output
│   ├── normalize.py              # Title/author/venue normalization
│   ├── aggregate.py              # Bootstrap CI computation
│   ├── schema.py                 # Pydantic models
│   └── cli.py                    # `citecheck verify ...`
│
├── scripts/
│   ├── build_runs_from_claims.py # Build (claim, condition) input tuples
│   ├── generate_runs.py          # Closed-book LLM calls
│   ├── run_rag.py                # ⭐ RAG generation (Crossref retrieval + Claude)
│   ├── verify_runs.py            # Verify any model outputs (closed-book or RAG)
│   ├── analyze_rag.py            # ⭐ RAG vs. closed-book analysis
│   └── sample_for_manual_validation.py
│
├── analysis/
│   ├── summarize_metrics.py      # Per-cell tables for Table 2
│   ├── compute_dual_metrics.py   # DOI completeness, count compliance, etc.
│   ├── domain_metrics.py         # Per-domain existence rates (Figure 4)
│   ├── plot_fig2a_boxplot.py     # Boxplot generation (Figure 3)
│   └── export_fig2a_data.py
│
├── data/
│   ├── claims.csv                # 144 claims (the experimental dataset)
│   └── <model>_full_runs.jsonl   # Pre-generated closed-book outputs (committed)
│
├── out/
│   ├── verify/
│   │   ├── citations.jsonl       # 17,443 closed-book citation labels
│   │   ├── run_metrics.jsonl     # Per-run aggregates (closed-book)
│   │   ├── rag_citations.jsonl   # 1,359 RAG citation labels
│   │   └── rag_run_metrics.jsonl # Per-run aggregates (RAG)
│   ├── rag/
│   │   ├── claude-sonnet-4-5-20250929_rag_runs.jsonl
│   │   ├── run_config.json
│   │   └── deliverables/
│   │       ├── rag_metrics.json
│   │       ├── rag_results.md
│   │       ├── rag_table_row.tex
│   │       └── rag_paragraph.tex
│   └── analysis/
│       ├── domain_metrics.csv
│       ├── fig2a_frac_existing.csv
│       └── ...
│
├── cache/
│   ├── crossref_doi.jsonl        # Crossref DOI lookups (closed-book verify)
│   ├── crossref_title.jsonl      # Crossref title searches (closed-book verify)
│   ├── s2_title.jsonl            # Semantic Scholar title searches
│   └── crossref_rag/             # Crossref keyword searches for RAG
│
├── manual_validation_*.csv       # Human-audit annotations (200 citations, 3 batches)
├── appendix/                     # Full 240-item candidate pool, subdomain map
├── examples/
├── tests/
├── pyproject.toml
└── README.md
```

⭐ = files most relevant to the RAG baseline + threshold-sensitivity reproduction.

---

## Reproducibility

| Component | How it's pinned |
|---|---|
| Model versions | Exact identifiers logged in `out/<model>_full_runs.jsonl` and `out/rag/run_config.json` (`claude-sonnet-4-5-20250929`, `gpt-4o` API alias, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, `Qwen/Qwen2.5-14B-Instruct`) |
| Crossref / Semantic Scholar responses | Cached to disk before normalization (`cache/`). Labels can be regenerated from cached snapshots without API access |
| Bootstrap CIs | 1,000 cluster-bootstrap resamples, seed=42 |
| Sampling | Fixed seeds throughout |
| Decoding | temperature 0, max_tokens 2048 |

---

## Manual audit (label validation)

A stratified 200-citation human audit was performed across two batches by two annotators with reconciliation:

- `manual_validation_100.csv` — Batch 1 (n=100)
- `manual_validation_200_batch2.csv` — Batch 2 (n=100)
- `manual_validation_300_batch3.csv` — Batch 3 (additional annotations for replication)

Combined pipeline-vs-human agreement: **68%, Cohen's κ = 0.52** (moderate). Per-class precision: 0.93 (Existing), 0.41 (Unresolved), 0.86 (Fabricated). Detail in §4 of the paper.

---

## Citation

If you use this benchmark or the `citecheck` tool, please cite:

```bibtex
@inproceedings{zhao2026citation,
  author    = {Zhao, Chen and Tang, Yuan and Qian, Yitian},
  title     = {{Do Deployment Constraints Make LLMs Hallucinate Citations? An Empirical Study Across Four Models and Five Prompting Regimes}},
  booktitle = {Proceedings of the 7th International Conference on Deep Learning Theory and Applications (DeLTA 2026)},
  year      = {2026},
  publisher = {Springer}
}
```

## License

Apache 2.0.
