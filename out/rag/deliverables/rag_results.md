# Retrieval-Augmented Baseline: Results (Claude Sonnet 4.5)

Minimal RAG experiment requested by Reviewer R5. Same 144 claims as the closed-book study; Claude Sonnet 4.5 only; Baseline + Temporal only; top-5 Crossref candidates injected before the citation-format block; identical verification pipeline, thresholds, and bootstrap methodology (seed=42, 1000 resamples over the 144 topic_ids).

## Interpretation

Grounding Claude Sonnet 4.5 with top-5 Crossref candidates substantially raises the Baseline existence rate, from 0.381 closed-book to 0.699 with retrieval (Δ = +0.318, 95% CI [0.254, 0.375]), while cutting fabrication from 0.157 to 0.054. On the Temporal condition—the worst closed-book cell, where existence collapses from 0.381 to 0.119—retrieval more than fully offsets this collapse: RAG's Temporal existence rate (0.818) not only erases the drop (Δ = +0.699, 95% CI [0.649, 0.746]) but exceeds even the unconstrained closed-book Baseline (0.381), with the temporal-violation rate staying low (0.035); year-windowed retrieval thus supplies in-window, verifiable references rather than trading verifiability for out-of-window citations. Two caveats temper the magnitude: the injected candidates are by construction real Crossref records, so part of the gain reflects faithful reuse of supplied references rather than improved unaided recall; and 4 Baseline and 1 Temporal RAG run(s) abstained entirely (the model declined to cite when it judged no candidate adequate), counted here as zero existing.

## Table 1 — Cell-level rates (95% bootstrap CI)

| Cell | Existing $\uparrow$ | Fabricated $\downarrow$ | Unresolved | Avg #Cit | Temporal Viol. |
|---|---|---|---|---|---|
| Claude+RAG Base | 0.699 [0.652, 0.744] | 0.054 [0.028, 0.083] | 0.219 [0.182, 0.260] | 4.76 | n/a |
| Claude+RAG Temp | 0.818 [0.777, 0.859] | 0.008 [0.003, 0.015] | 0.166 [0.131, 0.207] | 4.67 | 0.035 |
| Claude closed-book Base | 0.381 [0.336, 0.428] | 0.157 [0.125, 0.189] | 0.462 [0.419, 0.504] | 5.00 | n/a |
| Claude closed-book Temp | 0.119 [0.089, 0.156] | 0.347 [0.301, 0.390] | 0.533 [0.493, 0.575] | 5.00 | 0.015 |

*Abstentions:* 4 RAG-Base and 1 RAG-Temp run(s) produced **no** citations (the model explicitly declined to cite, e.g. "[No citations provided — no relevant peer-reviewed works ... were found]"). These are counted as 0 existing / 0 fabricated (conservative; they do not inflate the RAG existence rate). Closed-book has a 1.000 parse rate (0 abstentions). Excluding abstentions, RAG-Base existence = 0.719 and RAG-Temp existence = 0.824.

## Table 2 — Pairwise Δ in existence rate (paired 95% bootstrap CI)

| Comparison | Δ | 95% CI | CI excludes 0? |
|---|---|---|---|
| Claude+RAG Base − Claude Base | +0.318 | [0.254, 0.375] | yes |
| Claude+RAG Temp − Claude Temp | +0.699 | [0.649, 0.746] | yes |
| Claude+RAG Temp − Claude+RAG Base | +0.119 | [0.071, 0.166] | yes |

