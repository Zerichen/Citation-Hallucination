# Citation Hallucination Benchmark

A benchmark for measuring **citation hallucination** in LLMs under deployment-motivated prompting constraints.

Four models (Claude Sonnet 4.5, GPT-4o, LLaMA 3.1-8B, Qwen 2.5-14B) write citation-backed text on 144 claims across six domains, under five regimes (Baseline / Temporal / Survey / Non-Disclosure / Combo). Each generated reference is verified against **Crossref** and **Semantic Scholar** and labeled:

- **Existing** (`EXISTS`) — a real, indexed paper matches.
- **Unresolved** (`AMBIGUOUS`) — partial match; not confirmable without manual review.
- **Fabricated** (`FABRICATED`) — no plausible match.

The repo also ships a minimal **retrieval-augmented (RAG) baseline** (Claude, Baseline + Temporal) added for the DeLTA 2026 camera-ready, and releases the verification logic as an installable package, **`citecheck`**.

📄 *Do Deployment Constraints Make LLMs Hallucinate Citations? An Empirical Study Across Four Models and Five Prompting Regimes* (DeLTA 2026).

---

## Quick start

```bash
git clone https://github.com/Zerichen/Citation-Hallucination
cd Citation-Hallucination
pip install -e .

citecheck verify examples/sample_references.jsonl --output out/example_results.jsonl
```

Each input line is a citation record; each output line adds the label, best-match score, and the canonical record retrieved:

```json
{"title": "...", "authors": ["..."], "venue": "...", "year": 2024, "doi": "..."}
```

Or from Python:

```python
from citecheck import verify_citation

result = verify_citation({"title": "Attention Is All You Need",
                          "authors": ["Vaswani"], "venue": "NeurIPS", "year": 2017})
# result.label == "EXISTS", result.confidence == 0.97
```

### Install variants

Requires Python 3.9+. Base install pulls only verification deps (`requests`, `rapidfuzz`, `tqdm`, `numpy`, `pandas`).

| Command | Adds | Needed for |
|---|---|---|
| `pip install -e .` | (core only) | `citecheck verify`, `verify_runs.py`, most of `analysis/` |
| `pip install -e ".[dev]"` | `pytest` | `pytest tests/` |
| `pip install -e ".[llm]"` | `openai`, `anthropic` | `generate_runs.py`, `run_rag.py` (LLM calls) |
| `pip install -e ".[dev,llm]"` | both | full reproduction + tests |

---

## How verification works

**Scoring** ([`src/citecheck/match.py`](src/citecheck/match.py)) — each citation is scored against every retrieved candidate:

```
s = 0.60 · title_similarity   (token-set ratio)
  + 0.20 · author_overlap     (last-name set overlap)
  + 0.15 · year_match         (1.0 exact, 0.5 off-by-one, else 0)
  + 0.05 · venue_similarity   (partial ratio)
```

Title carries the most weight as the most discriminative field.

**Labels** ([`src/citecheck/label.py`](src/citecheck/label.py)):

```python
EXISTS_TH = 0.85   # s ≥ 0.85           → Existing
AMBIG_TH  = 0.60   # 0.60 ≤ s < 0.85    → Unresolved
                   # s < 0.60 / no cand → Fabricated
```

> **Naming:** data files use `EXISTS` / `AMBIGUOUS` / `FABRICATED`; the paper uses *Existing* / *Unresolved* / *Fabricated*. Same classes.

### Threshold sensitivity (§5)

`scripts/threshold_sensitivity.py` re-labels the frozen per-citation scores in `out/verify/citations.jsonl` under four threshold perturbations and recomputes rankings + paired-bootstrap Δ's — no API calls or LLM re-runs.

| | `EXISTS_TH` | `AMBIG_TH` | Tests |
|---|---|---|---|
| (a) | 0.80 | 0.60 | More permissive Existing cut |
| (b) | 0.90 | 0.60 | More restrictive Existing cut |
| (c) | 0.85 | 0.65 | Shrink the Unresolved band |
| (d) | 0.85 | 0.55 | Widen the Unresolved band |

```bash
python scripts/threshold_sensitivity.py        # → out/analysis/threshold_sensitivity.json
```

The output is committed as a frozen artifact ([`out/analysis/threshold_sensitivity.json`](out/analysis/threshold_sensitivity.json), bootstrap=200, seed=42). Headline: of **18** Δ's meaningful at the original thresholds, across all four perturbations there are **0** sign flips and **2** significance changes (72 perturbation-comparisons total) — rankings are stable to threshold choice.

---

## Full reproduction (closed-book, §3–§5)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# 1. Build (claim, condition) prompts, then call the LLMs
python scripts/build_runs_from_claims.py --claims data/claims.csv
python scripts/generate_runs.py            # → out/<model>_full_runs.jsonl

# 2. Verify
python scripts/verify_runs.py
#   → out/verify/citations.jsonl      per-citation labels + scores + evidence
#   → out/verify/run_metrics.jsonl    per-run aggregates

# 3. Aggregate + analyze (→ out/analysis/)
python analysis/summarize_metrics.py
python analysis/compute_dual_metrics.py
python analysis/domain_metrics.py
python analysis/plot_fig2a_boxplot.py
```

Each run: temperature 0, max_tokens 2048, one template per condition. Full design = **144 × 5 × 4 = 2,880 runs → 17,443 citations**.

---

## RAG baseline

A minimal grounded comparison (Reviewer R5) probing how much closed-book failure is recoverable through retrieval. Intentionally narrow:

| Axis | Closed-book | RAG |
|---|---|---|
| Models | all four | Claude Sonnet 4.5 only |
| Conditions | 5 regimes | Baseline + Temporal |
| Claims | 144 | 144 (same) |
| Retrieval | none | Crossref top-5 by claim keywords (year-windowed under Temporal) |
| Verification | pipeline | same pipeline, unchanged |

`run_rag.py` replays the same 144 claims (from `run_metrics.jsonl`), builds a Crossref query per claim's `seed_anchors`, fetches and caches the top-5 records, and injects them into an otherwise **byte-identical** prompt (same system prompt, schema, temperature 0, max_tokens 2048).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_rag.py
#   → out/rag/claude-sonnet-4-5-20250929_rag_runs.jsonl   (288 runs)
#   → out/rag/run_config.json                             (repro metadata)
#   → cache/crossref_rag/queries.jsonl                    (cached responses)

# Verify (same pipeline) and analyze
python scripts/verify_runs.py --input out/rag/claude-sonnet-4-5-20250929_rag_runs.jsonl \
                              --output out/verify/rag_citations.jsonl   # 1,359 citations
python scripts/analyze_rag.py
#   → out/rag/deliverables/{rag_metrics.json, rag_results.md, rag_table_row.tex, rag_paragraph.tex}
```

**Headline numbers:**

| Cell | Existing ↑ | Fabricated ↓ | Δ vs. closed-book |
|---|---|---|---|
| Claude closed-book Baseline | 0.381 | 0.157 | — |
| **Claude + RAG Baseline** | **0.699** | **0.054** | **+0.318 [0.254, 0.375]** |
| Claude closed-book Temporal | 0.119 | 0.347 | — |
| **Claude + RAG Temporal** | **0.818** | **0.008** | **+0.699 [0.649, 0.746]** |

Two caveats: (1) injected candidates are by construction real Crossref records, so part of the gain reflects faithful reuse rather than improved unaided recall; (2) 5 of 288 RAG runs abstained (no adequate candidate) and are counted as zero-existing — the conservative choice.

---

## Repository structure

```
src/citecheck/         Installable package
  match.py             ⭐ Scoring formula (0.6 / 0.2 / 0.15 / 0.05)
  label.py             ⭐ EXISTS_TH / AMBIG_TH thresholds
  prompts.py           Prompt templates (closed-book + RAG)
  clients.py           Anthropic / OpenAI wrappers + caching
  verify.py            Top-level verify() entrypoint
  parser.py            Citation parsing from raw model output
  normalize.py         Title/author/venue normalization
  aggregate.py         Bootstrap CI computation
  schema.py · cli.py   Pydantic models · `citecheck verify`

scripts/
  build_runs_from_claims.py   Build (claim, condition) tuples
  generate_runs.py            Closed-book LLM calls
  run_rag.py                  ⭐ RAG generation (Crossref + Claude)
  verify_runs.py              Verify any outputs (closed-book or RAG)
  analyze_rag.py              ⭐ RAG vs. closed-book analysis
  threshold_sensitivity.py    ⭐ §5 threshold re-labeling

analysis/              Per-cell / per-domain tables + figures (Tables 2, Figs 3–4)

data/                  claims.csv (144 claims) + pre-generated closed-book runs
out/verify/            citations.jsonl, run_metrics.jsonl (+ rag_*)
out/rag/               RAG runs, run_config.json, deliverables/
out/analysis/          summary tables, figure data, threshold_sensitivity.json
cache/                 Crossref / Semantic Scholar responses (offline repro)
manual_validation_*.csv   Human-audit annotations (300 citations, 3 batches)
appendix/ examples/ tests/ pyproject.toml
```

⭐ = most relevant to the RAG baseline + threshold-sensitivity reproduction.

---

## Reproducibility

| Component | How it's pinned |
|---|---|
| Model versions | Logged in `out/<model>_full_runs.jsonl` and `out/rag/run_config.json` (`claude-sonnet-4-5-20250929`, `gpt-4o`, `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`, `Qwen/Qwen2.5-14B-Instruct`) |
| Crossref / S2 responses | Cached to `cache/` before normalization; labels regenerate from snapshots without API access |
| Bootstrap CIs | 1,000 cluster-bootstrap resamples (200 for threshold sensitivity), seed=42 |
| Decoding | temperature 0, max_tokens 2048 |

---

## Manual audit

Stratified 200-citation human audit, two annotators with reconciliation: `manual_validation_100.csv` (batch 1) + `manual_validation_200_batch2.csv` (batch 2), with `manual_validation_300_batch3.csv` adding replication annotations.

Pipeline-vs-human agreement: **68%, Cohen's κ = 0.52** (moderate). Per-class precision: 0.93 (Existing), 0.41 (Unresolved), 0.86 (Fabricated). Detail in §4.

---

## Citation

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
