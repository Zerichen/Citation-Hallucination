#!/usr/bin/env python3
import json
import argparse
import numpy as np
import pandas as pd

METRICS = [
    "fabricated_rate",
    "exists_rate",
    "ambiguous_rate",
    "temporal_violation_rate",
    "num_citations",
]

def read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
    return pd.DataFrame(rows)

def bootstrap_ci(x, n_boot=1000, alpha=0.05, rng=None):
    x = np.asarray(x, dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    means = []
    n = len(x)
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(x.mean()), float(lower), float(upper)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="Path to run_metrics.jsonl")
    parser.add_argument("--out_csv", required=True, help="Output summary CSV")
    parser.add_argument("--out_tex", required=True, help="Output LaTeX table")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = read_jsonl(args.in_jsonl)

    required = {"topic_id", "model", "condition"} | set(METRICS)
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns in jsonl: {missing}")

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    rng = np.random.default_rng(args.seed)

    rows = []
    for (model, condition), g in df.groupby(["model", "condition"], dropna=False):
        row = {"model": model, "condition": condition, "n_topics": len(g)}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(g[m].dropna().values, n_boot=args.n_boot, rng=rng)
            row[f"{m}_mean"] = mean
            row[f"{m}_ci_lo"] = lo
            row[f"{m}_ci_hi"] = hi
            row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["condition", "model"]).reset_index(drop=True)
    summary.to_csv(args.out_csv, index=False)

    # Build LaTeX table
    latex_cols = [
        "model", "condition", "n_topics",
        "fabricated_rate_mean", "fabricated_rate_ci",
        "exists_rate_mean", "exists_rate_ci",
        "ambiguous_rate_mean",
        "temporal_violation_rate_mean",
        "num_citations_mean",
    ]
    latex_df = summary[latex_cols].copy()
    latex_df.columns = [
        "Model", "Condition", "N",
        "Fabricated $\\downarrow$", "CI",
        "Exists $\\uparrow$", "CI",
        "Ambiguous",
        "Temporal Viol.",
        "\\#Citations",
    ]

    latex = latex_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Citation quality metrics with 95\\% bootstrap confidence intervals.",
        label="tab:citation_metrics",
    )
    with open(args.out_tex, "w") as f:
        f.write(latex)

    print(f"[OK] Wrote: {args.out_csv}")
    print(f"[OK] Wrote: {args.out_tex}")

if __name__ == "__main__":
    main()
