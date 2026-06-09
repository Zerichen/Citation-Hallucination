#!/usr/bin/env python3
"""
Compute domain-stratified citation verification metrics.

Joins citation-level verification results with the claim dataset to produce
existence, fabricated, and ambiguous rates grouped by domain (and optionally
by model). Outputs a CSV for reproducibility and prints a summary table.

Usage:
    python analysis/domain_metrics.py \
        --citations out/verify/citations.jsonl \
        --claims data/claims.csv \
        --out_csv out/analysis/domain_metrics.csv
"""

import argparse
import json
import numpy as np
import pandas as pd

# Map domain prefix to the six domain groups used in the paper
DOMAIN_GROUP_MAP = {
    "CS": "SE & CS",
    "HUM": "Humanities",
    "INT": "Interdisciplinary",
    "MH": "Medicine & Health",
    "NS": "Natural Sciences",
    "SS": "Social Sciences",
}


def read_citations_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def bootstrap_ci(x, n_boot=1000, alpha=0.05, rng=None):
    """Return (mean, ci_lo, ci_hi) via bootstrap resampling."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x)
    means = [rng.choice(x, size=n, replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(x.mean()), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(
        description="Compute domain-stratified citation verification metrics."
    )
    parser.add_argument(
        "--citations",
        default="out/verify/citations.jsonl",
        help="Path to citation-level verification results",
    )
    parser.add_argument(
        "--claims",
        default="data/claims.csv",
        help="Path to claims CSV with domain labels",
    )
    parser.add_argument(
        "--out_csv",
        default="out/analysis/domain_metrics.csv",
        help="Output CSV path",
    )
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load data
    cit = read_citations_jsonl(args.citations)
    claims = pd.read_csv(args.claims)

    # Map claim domains to domain groups
    claims["domain_group"] = claims["domain"].str.split("_").str[0].map(DOMAIN_GROUP_MAP)

    # topic_id in citations is like 't001', claim_id in claims is integer
    # Build lookup: 't001' -> domain_group for claim_id=1, etc.
    domain_lookup = {}
    for _, row in claims.iterrows():
        key = f"t{int(row['claim_id']):03d}"
        domain_lookup[key] = row["domain_group"]

    # Add domain group to citations
    cit["domain_group"] = cit["topic_id"].map(domain_lookup)
    unmapped = cit["domain_group"].isna().sum()
    if unmapped > 0:
        print(f"Warning: {unmapped} citations could not be mapped to a domain group")

    # Binary indicators for each label
    cit["is_existing"] = (cit["label"] == "EXISTS").astype(int)
    cit["is_fabricated"] = (cit["label"] == "FABRICATED").astype(int)
    cit["is_ambiguous"] = (cit["label"] == "AMBIGUOUS").astype(int)

    rng = np.random.default_rng(args.seed)
    rows = []

    # --- Domain group aggregated (across all models and conditions) ---
    for domain_group, g in cit.groupby("domain_group"):
        n_citations = len(g)
        n_claims = g["topic_id"].nunique()
        exist_mean, exist_lo, exist_hi = bootstrap_ci(g["is_existing"].values, args.n_boot, rng=rng)
        fab_mean, fab_lo, fab_hi = bootstrap_ci(g["is_fabricated"].values, args.n_boot, rng=rng)
        amb_mean, amb_lo, amb_hi = bootstrap_ci(g["is_ambiguous"].values, args.n_boot, rng=rng)
        rows.append({
            "domain_group": domain_group,
            "model": "ALL",
            "n_claims": n_claims,
            "n_citations": n_citations,
            "exists_rate": round(exist_mean, 3),
            "exists_ci": f"[{exist_lo:.3f}, {exist_hi:.3f}]",
            "fabricated_rate": round(fab_mean, 3),
            "fabricated_ci": f"[{fab_lo:.3f}, {fab_hi:.3f}]",
            "ambiguous_rate": round(amb_mean, 3),
            "ambiguous_ci": f"[{amb_lo:.3f}, {amb_hi:.3f}]",
        })

    # --- Domain group x model ---
    for (domain_group, model), g in cit.groupby(["domain_group", "model"]):
        n_citations = len(g)
        n_claims = g["topic_id"].nunique()
        exist_mean, exist_lo, exist_hi = bootstrap_ci(g["is_existing"].values, args.n_boot, rng=rng)
        fab_mean, fab_lo, fab_hi = bootstrap_ci(g["is_fabricated"].values, args.n_boot, rng=rng)
        amb_mean, amb_lo, amb_hi = bootstrap_ci(g["is_ambiguous"].values, args.n_boot, rng=rng)
        rows.append({
            "domain_group": domain_group,
            "model": model,
            "n_claims": n_claims,
            "n_citations": n_citations,
            "exists_rate": round(exist_mean, 3),
            "exists_ci": f"[{exist_lo:.3f}, {exist_hi:.3f}]",
            "fabricated_rate": round(fab_mean, 3),
            "fabricated_ci": f"[{fab_lo:.3f}, {fab_hi:.3f}]",
            "ambiguous_rate": round(amb_mean, 3),
            "ambiguous_ci": f"[{amb_lo:.3f}, {amb_hi:.3f}]",
        })

    out = pd.DataFrame(rows).sort_values(["domain_group", "model"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)

    # Print summary
    print(f"\nDomain-stratified metrics ({len(cit)} citations, {cit['topic_id'].nunique()} claims)")
    print("=" * 90)
    agg = out[out["model"] == "ALL"].copy()
    print(f"\n{'Domain Group':<22s} {'Claims':>6s} {'Citations':>9s} {'Exists':>8s} {'Fabricated':>10s} {'Ambiguous':>10s}")
    print("-" * 70)
    for _, r in agg.iterrows():
        print(f"{r['domain_group']:<22s} {r['n_claims']:>6d} {r['n_citations']:>9d} "
              f"{r['exists_rate']:>8.3f} {r['fabricated_rate']:>10.3f} {r['ambiguous_rate']:>10.3f}")

    print(f"\n\nDomain x Model breakdown:")
    print(f"{'Domain Group':<22s} {'Model':<25s} {'Citations':>9s} {'Exists':>8s} {'Fabricated':>10s}")
    print("-" * 80)
    detail = out[out["model"] != "ALL"].copy()
    for _, r in detail.iterrows():
        print(f"{r['domain_group']:<22s} {r['model']:<25s} {r['n_citations']:>9d} "
              f"{r['exists_rate']:>8.3f} {r['fabricated_rate']:>10.3f}")

    print(f"\n[OK] Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
