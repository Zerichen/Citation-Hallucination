#!/usr/bin/env python3
"""Analyze the RAG vs. closed-book comparison for the DeLTA 2026 camera-ready.

Produces, for Claude Sonnet 4.5 only, Baseline + Temporal:
  1. Cell-level rates (existing / fabricated / unresolved) with 95% bootstrap CIs,
     plus avg #citations and (Temporal only) temporal-violation rate, for:
       - Claude+RAG Base / Temp           (from rag_citations.jsonl)
       - Claude closed-book Base / Temp    (recomputed from citations.jsonl)
  2. Pairwise Δ in existence rate with paired 95% bootstrap CIs:
       - Claude+RAG Base  − Claude Base
       - Claude+RAG Temp  − Claude Temp
       - Claude+RAG Temp  − Claude+RAG Base
  3. A short data-driven interpretation.

Methodology mirrors analysis/compute_dual_metrics.py: per-run (per-claim)
citation-level rates, averaged across the 144 claims; bootstrap resamples the
144 topic_ids. Seed = 42, 1000 resamples (paired for the Δ comparisons, since
the cells share the same 144 topic_ids).

Deliverables written to:
    <delta_dir>/rag_results.md
    <delta_dir>/rag_table_row.tex
    <delta_dir>/rag_paragraph.tex
    <delta_dir>/SUMMARY.txt
    <delta_dir>/rag_metrics.json   (machine-readable)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL = "claude-sonnet-4-5-20250929"
SEED = 42
N_BOOT = 1000
ALPHA = 0.05

DEFAULT_DELTA_DIR = (
    "/Users/chenzhao/Documents/Meta/Green Card/NIW/Background Improvement/"
    "Delta 2026/rag_results"
)


# ---- IO ----------------------------------------------------------------------
def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---- Per-topic rates ---------------------------------------------------------
def per_topic_rates(
    citations: List[dict], model: str, condition: str
) -> Dict[str, Dict[str, float]]:
    """Group per-citation labels into per-topic (per-run) rates.

    Returns {topic_id: {exists, fabricated, unresolved, temporal_viol, n_cit}}.
    One run per topic per cell; topic_id is the join key.
    """
    by_topic: Dict[str, List[dict]] = {}
    for c in citations:
        if c.get("model") != model or c.get("condition") != condition:
            continue
        by_topic.setdefault(c["topic_id"], []).append(c)

    out: Dict[str, Dict[str, float]] = {}
    for tid, cits in by_topic.items():
        n = len(cits)
        if n == 0:
            continue
        n_exist = sum(1 for c in cits if c["label"] == "EXISTS")
        n_fab = sum(1 for c in cits if c["label"] == "FABRICATED")
        n_amb = sum(1 for c in cits if c["label"] == "AMBIGUOUS")
        n_tv = sum(1 for c in cits if c.get("error_type") == "temporal")
        out[tid] = {
            "exists": n_exist / n,
            "fabricated": n_fab / n,
            "unresolved": n_amb / n,
            "temporal_viol": n_tv / n,
            "n_cit": float(n),
        }
    return out


# ---- Bootstrap ---------------------------------------------------------------
def bootstrap_mean_ci(
    values: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT, alpha: float = ALPHA
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def bootstrap_paired_delta_ci(
    a: np.ndarray, b: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT, alpha: float = ALPHA
) -> Tuple[float, float, float]:
    """Paired bootstrap of mean(a) - mean(b); a and b aligned by topic index."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(a[idx].mean() - b[idx].mean())
    deltas = np.array(deltas)
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return float(a.mean() - b.mean()), lo, hi


# ---- Cell summary ------------------------------------------------------------
def summarize_cell(
    rates: Dict[str, Dict[str, float]], topic_ids: List[str], condition: str
) -> dict:
    def col(metric: str) -> np.ndarray:
        return np.array([rates[t][metric] for t in topic_ids], dtype=float)

    n_cit_vals = col("n_cit")
    summary = {
        "n_topics": len(topic_ids),
        "avg_num_citations": float(n_cit_vals.mean()),
    }
    for metric in ["exists", "fabricated", "unresolved"]:
        mean, lo, hi = bootstrap_mean_ci(col(metric))
        summary[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    if condition == "temporal":
        mean, lo, hi = bootstrap_mean_ci(col("temporal_viol"))
        summary["temporal_viol"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    else:
        summary["temporal_viol"] = None
    return summary


# ---- Formatting helpers ------------------------------------------------------
def fmt3(x: float) -> str:
    return f"{x:.3f}"


def fmt_ci(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"


def tex_rate(d: dict) -> str:
    """`.NNN{\\scriptsize[.LLL,.HHH]}` — leading zero dropped, bracketed CI."""
    def lead(v: float) -> str:
        s = f"{abs(v):.3f}"
        s = s[1:] if s.startswith("0") else s  # drop leading 0
        return ("-" if v < 0 else "") + s
    return (
        f"${lead(d['mean'])}$\\,{{\\scriptsize$[{lead(d['ci_lo'])},{lead(d['ci_hi'])}]$}}"
    )


def tex_delta(mean: float, lo: float, hi: float) -> Tuple[str, str, bool]:
    def lead_signed(v: float) -> str:
        s = f"{abs(v):.3f}"
        s = s[1:] if s.startswith("0") else s
        sign = "-" if v < 0 else "+"
        return sign + s

    def lead_ci(v: float) -> str:
        s = f"{abs(v):.3f}"
        s = s[1:] if s.startswith("0") else s
        return ("-" if v < 0 else "") + s

    excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    delta_str = lead_signed(mean)
    ci_str = f"[{lead_ci(lo)},\\,{lead_ci(hi)}]"
    return delta_str, ci_str, excludes_zero


# ---- Main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--citations", default=os.path.join(ROOT_DIR, "out/verify/citations.jsonl"))
    ap.add_argument("--rag_citations", default=os.path.join(ROOT_DIR, "out/verify/rag_citations.jsonl"))
    ap.add_argument("--delta_dir", default=DEFAULT_DELTA_DIR)
    args = ap.parse_args()

    os.makedirs(args.delta_dir, exist_ok=True)

    cb = read_jsonl(args.citations)
    rag = read_jsonl(args.rag_citations)

    # Per-cell per-topic rates
    cb_base = per_topic_rates(cb, MODEL, "baseline")
    cb_temp = per_topic_rates(cb, MODEL, "temporal")
    rag_base = per_topic_rates(rag, MODEL, "baseline")
    rag_temp = per_topic_rates(rag, MODEL, "temporal")

    # The shared 144 topic set (defined by closed-book Claude baseline).
    topic_ids = sorted(cb_base.keys())
    if len(topic_ids) != 144:
        print(f"[WARN] closed-book baseline has {len(topic_ids)} topics, expected 144")

    # A run that produces no parseable citations (e.g. the model abstains and
    # writes "[No citations provided ...]") has no rows in citations.jsonl. We
    # keep such a topic in the cell and count it as 0 existing / 0 fabricated /
    # 0 unresolved (the model produced zero verifiable citations and zero
    # fabrications). This is the conservative choice: it does NOT inflate the
    # RAG existence rate, and it is applied identically to both conditions
    # (closed-book has a 1.000 parse rate, so it is unaffected). Abstentions are
    # reported explicitly below.
    ZERO = {"exists": 0.0, "fabricated": 0.0, "unresolved": 0.0, "temporal_viol": 0.0, "n_cit": 0.0}

    def fill(rates: Dict[str, Dict[str, float]]):
        abstain = sorted(t for t in topic_ids if t not in rates)
        for t in abstain:
            rates[t] = dict(ZERO)
        return abstain

    abstain_counts = {
        "cb_base": fill(cb_base),
        "cb_temp": fill(cb_temp),
        "rag_base": fill(rag_base),
        "rag_temp": fill(rag_temp),
    }
    for name, abst in abstain_counts.items():
        if abst:
            print(f"[INFO] {name}: {len(abst)} abstention run(s) imputed as 0: {abst}")

    cells = {
        "rag_base": summarize_cell(rag_base, topic_ids, "baseline"),
        "rag_temp": summarize_cell(rag_temp, topic_ids, "temporal"),
        "cb_base": summarize_cell(cb_base, topic_ids, "baseline"),
        "cb_temp": summarize_cell(cb_temp, topic_ids, "temporal"),
    }

    # Annotate each cell with abstention info and an "existence excluding
    # abstentions" mean (existence over the topics that produced >=1 citation).
    rate_dicts = {"rag_base": rag_base, "rag_temp": rag_temp, "cb_base": cb_base, "cb_temp": cb_temp}
    for key, cell in cells.items():
        rates = rate_dicts[key]
        n_abstain = len(abstain_counts[key])
        cell["n_abstain"] = n_abstain
        cell["abstain_topics"] = abstain_counts[key]
        nonzero = [rates[t]["exists"] for t in topic_ids if rates[t]["n_cit"] > 0]
        cell["exists_excl_abstain_mean"] = float(np.mean(nonzero)) if nonzero else float("nan")

    def exists_col(rates):
        return np.array([rates[t]["exists"] for t in topic_ids], dtype=float)

    # Pairwise Δ (existence rate), paired bootstrap over shared topic_ids.
    comparisons = [
        ("Claude+RAG Base $-$ Claude Base", exists_col(rag_base), exists_col(cb_base)),
        ("Claude+RAG Temp $-$ Claude Temp", exists_col(rag_temp), exists_col(cb_temp)),
        ("Claude+RAG Temp $-$ Claude+RAG Base", exists_col(rag_temp), exists_col(rag_base)),
    ]
    deltas = []
    for label, a, b in comparisons:
        mean, lo, hi = bootstrap_paired_delta_ci(a, b)
        deltas.append({"label": label, "mean": mean, "ci_lo": lo, "ci_hi": hi})

    # ---- machine-readable dump ----
    metrics_out = {
        "model": MODEL,
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_topics": len(topic_ids),
        "cells": cells,
        "deltas": deltas,
    }
    with open(os.path.join(args.delta_dir, "rag_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    # ---- interpretation (data-driven) ----
    d_base = deltas[0]
    d_temp = deltas[1]
    cb_temp_drop = cells["cb_base"]["exists"]["mean"] - cells["cb_temp"]["exists"]["mean"]

    def sig(d):
        return (d["ci_lo"] > 0 and d["ci_hi"] > 0) or (d["ci_lo"] < 0 and d["ci_hi"] < 0)

    base_help = "substantially raises" if d_base["mean"] > 0 and sig(d_base) else (
        "does not significantly change" if not sig(d_base) else "lowers")
    rag_temp_e = cells["rag_temp"]["exists"]["mean"]
    cb_base_e = cells["cb_base"]["exists"]["mean"]
    cb_temp_e = cells["cb_temp"]["exists"]["mean"]
    frac_recovered = (d_temp["mean"] / cb_temp_drop) if cb_temp_drop else float("nan")
    overshoots = rag_temp_e >= cb_base_e  # RAG Temporal beats even unconstrained closed-book

    if overshoots:
        temp_clause = (
            f"retrieval more than fully offsets this collapse: RAG's Temporal existence rate "
            f"({rag_temp_e:.3f}) not only erases the drop "
            f"(Δ = {d_temp['mean']:+.3f}, 95% CI [{d_temp['ci_lo']:.3f}, {d_temp['ci_hi']:.3f}]) "
            f"but exceeds even the unconstrained closed-book Baseline ({cb_base_e:.3f})"
        )
    else:
        temp_clause = (
            f"retrieval recovers about {frac_recovered*100:.0f}% of the closed-book Temporal "
            f"drop (Δ = {d_temp['mean']:+.3f}, 95% CI "
            f"[{d_temp['ci_lo']:.3f}, {d_temp['ci_hi']:.3f}]), reaching {rag_temp_e:.3f}"
        )

    interp = (
        f"Grounding Claude Sonnet 4.5 with top-5 Crossref candidates {base_help} the Baseline "
        f"existence rate, from {cb_base_e:.3f} closed-book to "
        f"{cells['rag_base']['exists']['mean']:.3f} with retrieval "
        f"(Δ = {d_base['mean']:+.3f}, 95% CI [{d_base['ci_lo']:.3f}, {d_base['ci_hi']:.3f}]), "
        f"while cutting fabrication from {cells['cb_base']['fabricated']['mean']:.3f} to "
        f"{cells['rag_base']['fabricated']['mean']:.3f}. "
        f"On the Temporal condition—the worst closed-book cell, where existence collapses "
        f"from {cb_base_e:.3f} to {cb_temp_e:.3f}—{temp_clause}, with the temporal-violation "
        f"rate staying low ({cells['rag_temp']['temporal_viol']['mean']:.3f}); year-windowed "
        f"retrieval thus supplies in-window, verifiable references rather than trading "
        f"verifiability for out-of-window citations. "
        f"Two caveats temper the magnitude: the injected candidates are by construction real "
        f"Crossref records, so part of the gain reflects faithful reuse of supplied references "
        f"rather than improved unaided recall; and {cells['rag_base']['n_abstain']} Baseline and "
        f"{cells['rag_temp']['n_abstain']} Temporal RAG run(s) abstained entirely (the model "
        f"declined to cite when it judged no candidate adequate), counted here as zero existing."
    )

    # ---- rag_results.md ----
    md = []
    md.append("# Retrieval-Augmented Baseline: Results (Claude Sonnet 4.5)\n")
    md.append(
        f"Minimal RAG experiment requested by Reviewer R5. Same 144 claims as the "
        f"closed-book study; Claude Sonnet 4.5 only; Baseline + Temporal only; top-{5} "
        f"Crossref candidates injected before the citation-format block; identical "
        f"verification pipeline, thresholds, and bootstrap methodology (seed={SEED}, "
        f"{N_BOOT} resamples over the {len(topic_ids)} topic_ids).\n"
    )
    md.append("## Interpretation\n")
    md.append(interp + "\n")

    md.append("## Table 1 — Cell-level rates (95% bootstrap CI)\n")
    md.append(
        "| Cell | Existing $\\uparrow$ | Fabricated $\\downarrow$ | Unresolved | Avg #Cit | Temporal Viol. |"
    )
    md.append("|---|---|---|---|---|---|")
    row_order = [
        ("Claude+RAG Base", "rag_base"),
        ("Claude+RAG Temp", "rag_temp"),
        ("Claude closed-book Base", "cb_base"),
        ("Claude closed-book Temp", "cb_temp"),
    ]
    for label, key in row_order:
        c = cells[key]
        tv = fmt3(c["temporal_viol"]["mean"]) if c["temporal_viol"] else "n/a"
        md.append(
            f"| {label} | {fmt_ci(c['exists'])} | {fmt_ci(c['fabricated'])} | "
            f"{fmt_ci(c['unresolved'])} | {c['avg_num_citations']:.2f} | {tv} |"
        )
    md.append("")
    n_ab_base = cells["rag_base"]["n_abstain"]
    n_ab_temp = cells["rag_temp"]["n_abstain"]
    if n_ab_base or n_ab_temp:
        md.append(
            f"*Abstentions:* {n_ab_base} RAG-Base and {n_ab_temp} RAG-Temp run(s) produced "
            f"**no** citations (the model explicitly declined to cite, e.g. \"[No citations "
            f"provided — no relevant peer-reviewed works ... were found]\"). These are counted "
            f"as 0 existing / 0 fabricated (conservative; they do not inflate the RAG existence "
            f"rate). Closed-book has a 1.000 parse rate (0 abstentions). Excluding abstentions, "
            f"RAG-Base existence = {cells['rag_base']['exists_excl_abstain_mean']:.3f} and "
            f"RAG-Temp existence = {cells['rag_temp']['exists_excl_abstain_mean']:.3f}.\n"
        )

    md.append("## Table 2 — Pairwise Δ in existence rate (paired 95% bootstrap CI)\n")
    md.append("| Comparison | Δ | 95% CI | CI excludes 0? |")
    md.append("|---|---|---|---|")
    for d in deltas:
        excl = "yes" if sig(d) else "no"
        md.append(
            f"| {d['label'].replace('$-$','−')} | {d['mean']:+.3f} | "
            f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | {excl} |"
        )
    md.append("")
    with open(os.path.join(args.delta_dir, "rag_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # ---- rag_table_row.tex (rows for Table 3 / tab:delta) ----
    tex = []
    tex.append("% Two new rows for Table 3 (tab:delta): retrieval-augmented vs. closed-book.")
    tex.append("% Generated by scripts/analyze_rag.py. Bold = CI excludes zero; (ns) = CI overlaps zero.")
    tex.append("% Insert after the existing Claude/GPT/LLaMA/Qwen comparison blocks.")
    tex.append("\\midrule")
    tex.append("\\multicolumn{3}{@{}l}{\\emph{Retrieval-augmented vs.\\ closed-book (Claude Sonnet)}} \\\\")
    rag_labels = {
        "Claude+RAG Base $-$ Claude Base": "RAG $-$ CB (Base)",
        "Claude+RAG Temp $-$ Claude Temp": "RAG $-$ CB (Temp)",
    }
    for d in deltas:
        if d["label"] not in rag_labels:
            continue
        delta_str, ci_str, excl = tex_delta(d["mean"], d["ci_lo"], d["ci_hi"])
        short = rag_labels[d["label"]]
        if excl:
            tex.append(f"\\textbf{{{short}}} & $\\boldsymbol{{{delta_str}}}$ & ${ci_str}$ \\\\")
        else:
            tex.append(f"{short} \\textit{{(ns)}} & ${delta_str}$ & ${ci_str}$ \\\\")
    with open(os.path.join(args.delta_dir, "rag_table_row.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    # ---- rag_paragraph.tex (§5 paragraph) ----
    headline_sig = sig(d_temp)
    para = []
    para.append("\\paragraph{A retrieval-augmented baseline (answer to R5).}")
    sentence_lead = (
        f"To quantify how much of the observed citation hallucination is a grounding "
        f"deficit rather than a generation behavior, we ran a minimal retrieval-augmented "
        f"baseline for Claude Sonnet on the same {len(topic_ids)} claims under the Baseline "
        f"and Temporal conditions, injecting the top-{5} Crossref records (by keyword, "
        f"year-windowed under Temporal) into the prompt while holding the system prompt, "
        f"output schema, temperature, and verification pipeline fixed."
    )
    para.append(sentence_lead)
    para.append(
        f"Grounding raises Claude's Baseline existence rate from "
        f"{cells['cb_base']['exists']['mean']:.3f} to {cells['rag_base']['exists']['mean']:.3f} "
        f"($\\Delta = {d_base['mean']:+.3f}$, 95\\% CI "
        f"$[{d_base['ci_lo']:.3f},\\,{d_base['ci_hi']:.3f}]$)."
    )
    if overshoots:
        temp_tex = (
            f"Under the Temporal condition\\,---\\,the steepest closed-book drop "
            f"({cb_base_e:.3f}$\\rightarrow${cb_temp_e:.3f})\\,---\\,retrieval more than fully "
            f"offsets the collapse: existence rises to {rag_temp_e:.3f} "
            f"($\\Delta = {d_temp['mean']:+.3f}$, 95\\% CI "
            f"$[{d_temp['ci_lo']:.3f},\\,{d_temp['ci_hi']:.3f}]$), exceeding even the "
            f"unconstrained closed-book Baseline, while the temporal-violation rate stays at "
            f"{cells['rag_temp']['temporal_viol']['mean']:.3f}."
        )
    else:
        temp_tex = (
            f"Under the Temporal condition\\,---\\,the steepest closed-book drop "
            f"({cb_base_e:.3f}$\\rightarrow${cb_temp_e:.3f})\\,---\\,retrieval moves existence "
            f"to {rag_temp_e:.3f} ($\\Delta = {d_temp['mean']:+.3f}$, 95\\% CI "
            f"$[{d_temp['ci_lo']:.3f},\\,{d_temp['ci_hi']:.3f}]$), recovering roughly "
            f"{frac_recovered*100:.0f}\\% of the closed-book Temporal collapse while keeping the "
            f"temporal-violation rate at {cells['rag_temp']['temporal_viol']['mean']:.3f}."
        )
    para.append(temp_tex)
    para.append(
        "These two rows are added to Table~\\ref{tab:delta}. Two caveats bound the "
        "interpretation: the injected candidates are by construction real Crossref records, so "
        "the gain partly reflects faithful reuse of supplied references rather than improved "
        f"unaided recall, and {cells['rag_base']['n_abstain']+cells['rag_temp']['n_abstain']} "
        "of the 288 RAG runs abstained from citing entirely (counted as zero existing). Even so, "
        "grounding eliminates most of the constraint-induced hallucination for this model, "
        "motivating the broader four-model, five-condition RAG study outlined in the "
        "now-shortened Future Work paragraph (Retrieval-augmented baselines)."
    )
    with open(os.path.join(args.delta_dir, "rag_paragraph.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(para) + "\n")

    # ---- SUMMARY.txt ----
    if overshoots and headline_sig:
        closes_clause = (
            f"RAG more than fully closes the Temporal gap (RAG-Temporal existence "
            f"{rag_temp_e:.3f} exceeds even closed-book Baseline {cb_base_e:.3f})"
        )
    elif headline_sig and frac_recovered >= 0.5:
        closes_clause = f"RAG substantially closes the Temporal gap (~{frac_recovered*100:.0f}% of the drop recovered)"
    elif headline_sig and frac_recovered > 0:
        closes_clause = f"RAG modestly closes the Temporal gap (~{frac_recovered*100:.0f}% of the drop recovered)"
    else:
        closes_clause = "RAG does not significantly close the Temporal gap"
    summary_line = (
        f"Claude+RAG vs. Claude closed-book: "
        f"Base Δ={d_base['mean']:+.3f} [{d_base['ci_lo']:.2f}, {d_base['ci_hi']:.2f}], "
        f"Temp Δ={d_temp['mean']:+.3f} [{d_temp['ci_lo']:.2f}, {d_temp['ci_hi']:.2f}]. "
        f"{closes_clause}."
    )
    with open(os.path.join(args.delta_dir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(summary_line + "\n")

    # ---- console echo ----
    print("== Cell rates ==")
    for label, key in row_order:
        c = cells[key]
        tv = fmt3(c["temporal_viol"]["mean"]) if c["temporal_viol"] else "n/a"
        print(f"  {label:26s} exist={fmt_ci(c['exists'])} fab={fmt_ci(c['fabricated'])} "
              f"unres={fmt_ci(c['unresolved'])} #cit={c['avg_num_citations']:.2f} tv={tv}")
    print("== Deltas ==")
    for d in deltas:
        print(f"  {d['label']:40s} {d['mean']:+.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] "
              f"{'(sig)' if sig(d) else '(ns)'}")
    print("\n" + summary_line)
    print(f"\nDeliverables written to: {args.delta_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
