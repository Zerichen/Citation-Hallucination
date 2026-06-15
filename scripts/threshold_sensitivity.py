#!/usr/bin/env python3
"""Threshold-sensitivity analysis (§5 of the DeLTA 2026 paper).

Re-labels every citation in ``out/verify/citations.jsonl`` under four
plausible perturbations of the (EXISTS_TH, AMBIG_TH) thresholds, then
recomputes the headline metrics and pairwise Δ's from Table 3.

The point: the expensive parts (generation + Crossref/S2 verification) are
already frozen in ``citations.jsonl`` as a per-citation ``confidence`` score.
Sensitivity analysis is just ``s ≥ τ`` arithmetic — no API calls, no LLM
re-runs.

Default perturbations (matching the paper):
    (orig) 0.85 / 0.60   — paper's chosen thresholds
    (a)    0.80 / 0.60   — more permissive Existing cut
    (b)    0.90 / 0.60   — more restrictive Existing cut
    (c)    0.85 / 0.65   — shrink the Unresolved band
    (d)    0.85 / 0.55   — widen the Unresolved band

Outputs:
    out/analysis/threshold_sensitivity.json
        Per-perturbation per-cell rates, model rankings, and pairwise Δ's
        with 95% paired-bootstrap CIs.

Usage:
    python scripts/threshold_sensitivity.py
    python scripts/threshold_sensitivity.py --input out/verify/citations.jsonl
    python scripts/threshold_sensitivity.py --boot 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Display name mappings (data files use raw IDs; paper uses friendly names)
COND_MAP = {
    "baseline": "Base",
    "temporal": "Temp",
    "survey":   "Surv",
    "privacy":  "N-D",
    "combo":    "Combo",
}
MODEL_MAP = {
    "claude-sonnet-4-5-20250929":              "Claude Sonnet",
    "gpt-4o":                                  "GPT-4o",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "LLaMA 3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct":               "Qwen 2.5-14B",
}
PROPRIETARY = {"Claude Sonnet", "GPT-4o"}
OPEN_WEIGHT = {"LLaMA 3.1-8B", "Qwen 2.5-14B"}
CONDITIONS  = ["Base", "Temp", "Surv", "N-D", "Combo"]
ALL_MODELS  = list(MODEL_MAP.values())

# Perturbations to test
PERTURBATIONS: Dict[str, Tuple[float, float]] = {
    "(orig) 0.85/0.60": (0.85, 0.60),
    "(a)    0.80/0.60": (0.80, 0.60),
    "(b)    0.90/0.60": (0.90, 0.60),
    "(c)    0.85/0.65": (0.85, 0.65),
    "(d)    0.85/0.55": (0.85, 0.55),
}


def label(score: float, exists_th: float, ambig_th: float) -> str:
    if score >= exists_th:
        return "Existing"
    if score >= ambig_th:
        return "Unresolved"
    return "Fabricated"


def load_citations(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "model":     MODEL_MAP[d["model"]],
                "condition": COND_MAP[d["condition"]],
                "score":     float(d.get("confidence") or 0.0),
                "topic_id":  d.get("topic_id"),
            })
    return rows


def cell_rate(rows, model, cond, exists_th, ambig_th, which="Existing") -> float | None:
    subset = [r for r in rows if r["model"] == model and r["condition"] == cond]
    if not subset:
        return None
    k = sum(1 for r in subset if label(r["score"], exists_th, ambig_th) == which)
    return k / len(subset)


def bootstrap_delta(
    rows, group_a_filter, group_b_filter, exists_th, ambig_th,
    n_resamples=1000, seed=42,
) -> Tuple[float, float, float]:
    """Paired cluster-bootstrap over topic_id of (mean exists in A) − (B)."""
    rng = random.Random(seed)
    sel = [r for r in rows if group_a_filter(r) or group_b_filter(r)]
    topics = sorted({r["topic_id"] for r in sel})
    by_topic = defaultdict(list)
    for r in sel:
        by_topic[r["topic_id"]].append(r)

    deltas = []
    for _ in range(n_resamples):
        sample_topics = [rng.choice(topics) for _ in topics]
        sample = [r for t in sample_topics for r in by_topic[t]]
        a = [r for r in sample if group_a_filter(r)]
        b = [r for r in sample if group_b_filter(r)]
        if not a or not b:
            continue
        rate_a = sum(1 for r in a if label(r["score"], exists_th, ambig_th) == "Existing") / len(a)
        rate_b = sum(1 for r in b if label(r["score"], exists_th, ambig_th) == "Existing") / len(b)
        deltas.append(rate_a - rate_b)

    deltas.sort()
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    mean = sum(deltas) / len(deltas)
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return mean, lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  default="out/verify/citations.jsonl",
                        help="Per-citation labels with confidence scores")
    parser.add_argument("--output", default="out/analysis/threshold_sensitivity.json")
    parser.add_argument("--boot",   type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] {args.input} not found. Run scripts/verify_runs.py first.", file=sys.stderr)
        sys.exit(1)

    rows = load_citations(args.input)
    print(f"Loaded {len(rows)} citations.\n")

    results = {"perturbations": {}, "input": args.input, "n_bootstrap": args.boot, "seed": args.seed}

    # 1. Per-cell Existing rates under each perturbation
    print("=" * 70)
    print("Existing-rate by (model, condition) under each perturbation")
    print("=" * 70)
    for tag, (ex_th, am_th) in PERTURBATIONS.items():
        cell_rates = {}
        for m in ALL_MODELS:
            for c in CONDITIONS:
                r = cell_rate(rows, m, c, ex_th, am_th)
                cell_rates[f"{m}|{c}"] = r
        results["perturbations"][tag] = {
            "exists_th": ex_th, "ambig_th": am_th,
            "cell_existing_rates": cell_rates,
        }

    # 2. Model rankings per condition (does the ordering hold?)
    print("\n" + "=" * 70)
    print("Model rankings on Existing rate")
    print("=" * 70)
    for cond in CONDITIONS:
        print(f"\n  {cond}:")
        for tag, (ex_th, am_th) in PERTURBATIONS.items():
            rates = {m: cell_rate(rows, m, cond, ex_th, am_th) for m in ALL_MODELS}
            ordered = sorted(rates.items(), key=lambda x: -x[1])
            ordering = " > ".join(m.split()[0] for m, _ in ordered)
            print(f"    {tag:<25} {ordering}")

    # 3. Key pairwise deltas under each perturbation
    print("\n" + "=" * 70)
    print("Key pairwise deltas (Existing rate) under each perturbation")
    print("=" * 70)
    key_deltas = []
    for c in CONDITIONS:
        key_deltas.append(("Prop vs Open", c,
                           lambda r, c_=c: r["model"] in PROPRIETARY and r["condition"] == c_,
                           lambda r, c_=c: r["model"] in OPEN_WEIGHT and r["condition"] == c_))
    for m in ALL_MODELS:
        for cond in ["Temp", "Surv", "N-D", "Combo"]:
            key_deltas.append((f"{cond}-Base ({m})", cond,
                               lambda r, m_=m, c_=cond: r["model"] == m_ and r["condition"] == c_,
                               lambda r, m_=m: r["model"] == m_ and r["condition"] == "Base"))

    delta_results = {}
    n_meaningful_orig = 0
    sign_flips = 0
    significance_changes = 0

    for tag, (ex_th, am_th) in PERTURBATIONS.items():
        print(f"\n  {tag}:")
        delta_results[tag] = {}
        for name, cond, a_filter, b_filter in key_deltas:
            mean, lo, hi = bootstrap_delta(rows, a_filter, b_filter, ex_th, am_th,
                                           n_resamples=args.boot, seed=args.seed)
            sig = "*" if (lo > 0 or hi < 0) else " "
            delta_results[tag][name] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "significant": sig == "*"}
            if tag.startswith("(orig)"):
                if sig == "*":
                    n_meaningful_orig += 1
            print(f"    {name:<35} Δ={mean:+.3f} [{lo:+.3f},{hi:+.3f}] {sig}")
        print()

    # 4. Cross-perturbation sign/significance stability
    if n_meaningful_orig > 0:
        for name, *_ in key_deltas:
            orig = delta_results["(orig) 0.85/0.60"].get(name)
            if not orig or not orig["significant"]:
                continue
            for tag in PERTURBATIONS:
                if tag.startswith("(orig)"):
                    continue
                perturbed = delta_results[tag].get(name)
                if not perturbed:
                    continue
                if (orig["mean"] > 0) != (perturbed["mean"] > 0):
                    sign_flips += 1
                if not perturbed["significant"]:
                    significance_changes += 1

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Meaningful Δ's at original thresholds:        {n_meaningful_orig} of {len(key_deltas)}")
    print(f"Sign flips across 4 perturbations:            {sign_flips}")
    print(f"Significance changes across 4 perturbations:  {significance_changes}")

    results["summary"] = {
        "n_meaningful_orig":      n_meaningful_orig,
        "sign_flips":             sign_flips,
        "significance_changes":   significance_changes,
        "n_perturbation_compare": (len(PERTURBATIONS) - 1) * n_meaningful_orig,
    }
    results["deltas"] = delta_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {args.output}")


if __name__ == "__main__":
    main()
