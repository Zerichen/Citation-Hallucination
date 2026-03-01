#!/usr/bin/env python3
"""
Compute BOTH citation-level and claim-level metrics from citations.jsonl.

Citation-level: fraction of individual citations that are Existing/Fabricated/Ambiguous.
Claim-level (strict): a claim is "Fully Verified" only if ALL its citations are Existing;
                       "Fabricated" if ANY citation is Fabricated; else "Ambiguous".

Usage:
    python analysis/compute_dual_metrics.py \
        --in_jsonl out/verify/citations.jsonl \
        --out_csv  out/verify/dual_metrics.csv \
        --out_tex  out/verify/dual_table.tex
"""

import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict


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
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x)
    means = [rng.choice(x, size=n, replace=True).mean() for _ in range(n_boot)]
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(x.mean()), float(lower), float(upper)


def claim_level_label(labels):
    """Strict claim-level aggregation."""
    labels = list(labels)
    if all(l == "EXISTS" for l in labels):
        return "EXISTS"
    if any(l == "FABRICATED" for l in labels):
        return "FABRICATED"
    return "AMBIGUOUS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="Path to citations.jsonl")
    parser.add_argument("--out_csv", required=True, help="Output summary CSV")
    parser.add_argument("--out_tex", default=None, help="Output LaTeX table fragment")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = read_citations_jsonl(args.in_jsonl)
    rng = np.random.default_rng(args.seed)

    # Ensure required columns
    for col in ["run_id", "model", "condition", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # ---- Citation-level metrics per (model, condition) ----
    # Group by run_id to get per-claim citation-level rates, then average across claims
    run_groups = df.groupby(["run_id", "model", "condition"])

    claim_records = []
    for (run_id, model, condition), g in run_groups:
        n_cit = len(g)
        labels = g["label"].tolist()
        n_exist = sum(1 for l in labels if l == "EXISTS")
        n_fab = sum(1 for l in labels if l == "FABRICATED")
        n_ambig = sum(1 for l in labels if l == "AMBIGUOUS")
        n_temporal = sum(1 for _, r in g.iterrows() if r.get("error_type") == "temporal")

        claim_records.append({
            "run_id": run_id,
            "model": model,
            "condition": condition,
            "n_citations": n_cit,
            # Per-claim citation-level rates
            "cit_exist_rate": n_exist / n_cit if n_cit > 0 else 0,
            "cit_fab_rate": n_fab / n_cit if n_cit > 0 else 0,
            "cit_ambig_rate": n_ambig / n_cit if n_cit > 0 else 0,
            "temporal_viol_rate": n_temporal / n_cit if n_cit > 0 else 0,
            # Strict claim-level label
            "claim_label": claim_level_label(labels),
        })

    claims_df = pd.DataFrame(claim_records)

    # Now aggregate across claims per (model, condition)
    rows = []
    for (model, condition), g in claims_df.groupby(["model", "condition"], dropna=False):
        n_claims = len(g)
        row = {"model": model, "condition": condition, "n_claims": n_claims}

        # Citation-level rates (mean of per-claim rates = overall citation-level rate)
        for metric in ["cit_exist_rate", "cit_fab_rate", "cit_ambig_rate", "temporal_viol_rate"]:
            mean, lo, hi = bootstrap_ci(g[metric].values, n_boot=args.n_boot, rng=rng)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi

        # Avg citations per claim
        mean_cit, lo_cit, hi_cit = bootstrap_ci(g["n_citations"].values, n_boot=args.n_boot, rng=rng)
        row["avg_citations"] = mean_cit

        # Claim-level strict rates
        claim_exist = (g["claim_label"] == "EXISTS").mean()
        claim_fab = (g["claim_label"] == "FABRICATED").mean()
        claim_ambig = (g["claim_label"] == "AMBIGUOUS").mean()

        # Bootstrap for claim-level existence
        claim_exist_arr = (g["claim_label"] == "EXISTS").astype(float).values
        ce_mean, ce_lo, ce_hi = bootstrap_ci(claim_exist_arr, n_boot=args.n_boot, rng=rng)

        row["claim_exist_rate"] = ce_mean
        row["claim_exist_ci_lo"] = ce_lo
        row["claim_exist_ci_hi"] = ce_hi
        row["claim_fab_rate"] = claim_fab
        row["claim_ambig_rate"] = claim_ambig

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["condition", "model"]).reset_index(drop=True)
    summary.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")

    # ---- Generate LaTeX table fragment ----
    if args.out_tex:
        cond_order = ["baseline", "temporal", "survey", "privacy", "combo"]
        cond_short = {"baseline": "Base", "temporal": "Temp", "survey": "Surv", "privacy": "Priv", "combo": "Combo"}

        lines = []
        prev_cond = None
        for _, r in summary.sort_values(
            ["condition", "model"],
            key=lambda col: col.map(lambda x: cond_order.index(x) if x in cond_order else 99) if col.name == "condition" else col
        ).iterrows():
            cond = r["condition"]
            if prev_cond is not None and cond != prev_cond:
                lines.append("\\midrule")
            prev_cond = cond

            def fmt_ci(mean, lo, hi):
                return f".{int(round(mean*1000)):03d}\\,{{\\scriptsize[.{int(round(lo*1000)):03d},.{int(round(hi*1000)):03d}]}}"

            cit_e = fmt_ci(r["cit_exist_rate_mean"], r["cit_exist_rate_ci_lo"], r["cit_exist_rate_ci_hi"])
            cit_f = fmt_ci(r["cit_fab_rate_mean"], r["cit_fab_rate_ci_lo"], r["cit_fab_rate_ci_hi"])
            cit_a = fmt_ci(r["cit_ambig_rate_mean"], r["cit_ambig_rate_ci_lo"], r["cit_ambig_rate_ci_hi"])
            clm_e = fmt_ci(r["claim_exist_rate"], r["claim_exist_ci_lo"], r["claim_exist_ci_hi"])
            tviol = f".{int(round(r['temporal_viol_rate_mean']*1000)):03d}"
            avgc = f"{r['avg_citations']:.2f}"

            model_tex = r["model"].replace("_", "\\_")
            lines.append(
                f"{model_tex:18s} & {cond_short.get(cond, cond):5s} & {int(r['n_claims'])} "
                f"& {cit_e} & {cit_f} & {cit_a} & {clm_e} & {tviol} & {avgc} \\\\"
            )

        with open(args.out_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[OK] Wrote: {args.out_tex}")


if __name__ == "__main__":
    main()
