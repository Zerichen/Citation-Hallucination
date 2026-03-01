#!/usr/bin/env python3
"""
Export source data for Figure 2a: per-claim verification fraction boxplots.

Reads run_metrics.jsonl (one row per model × condition × claim) and outputs
a tidy CSV with columns:
    model, condition, topic_id, frac_existing

Each row represents one claim's verification fraction (f_i = #Existing / #Parsed).
The CSV is directly consumable by matplotlib for grouped boxplots.

Usage:
    python analysis/export_fig2a_data.py \
        --in_jsonl out/verify/run_metrics.jsonl \
        --out_csv  out/analysis/fig2a_frac_existing.csv
"""

import json
import argparse
import os


MODEL_DISPLAY = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet",
    "gpt-4o": "GPT-4o",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": "LLaMA 3.1-8B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen 2.5-14B",
}

CONDITION_ORDER = ["baseline", "temporal", "survey", "privacy", "combo"]
CONDITION_DISPLAY = {
    "baseline": "Baseline",
    "temporal": "Temporal",
    "survey": "Survey",
    "privacy": "Non-Disc.",
    "combo": "Combo",
}


def main():
    parser = argparse.ArgumentParser(
        description="Export per-claim verification fractions for Figure 2a."
    )
    parser.add_argument(
        "--in_jsonl",
        default="out/verify/run_metrics.jsonl",
        help="Path to run_metrics.jsonl",
    )
    parser.add_argument(
        "--out_csv",
        default="out/analysis/fig2a_frac_existing.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            model_raw = r.get("model", "")
            condition_raw = r.get("condition", "")
            topic_id = r.get("topic_id", "")
            exists_rate = r.get("exists_rate")

            if exists_rate is None:
                continue

            rows.append({
                "model": model_raw,
                "model_display": MODEL_DISPLAY.get(model_raw, model_raw),
                "condition": condition_raw,
                "condition_display": CONDITION_DISPLAY.get(condition_raw, condition_raw),
                "topic_id": topic_id,
                "frac_existing": exists_rate,
            })

    # Sort for deterministic output
    cond_rank = {c: i for i, c in enumerate(CONDITION_ORDER)}
    rows.sort(key=lambda r: (
        cond_rank.get(r["condition"], 99),
        r["model"],
        r["topic_id"],
    ))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    header = "model,model_display,condition,condition_display,topic_id,frac_existing"
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['model_display']},{r['condition']},"
                f"{r['condition_display']},{r['topic_id']},{r['frac_existing']:.6f}\n"
            )

    # Summary stats
    n_rows = len(rows)
    models = sorted(set(r["model_display"] for r in rows))
    conditions = sorted(set(r["condition_display"] for r in rows))
    print(f"[OK] Wrote {n_rows} rows to {args.out_csv}")
    print(f"     Models: {models}")
    print(f"     Conditions: {conditions}")
    print(f"     Claims per cell: {n_rows // max(len(models) * len(conditions), 1)}")


if __name__ == "__main__":
    main()
