#!/usr/bin/env python3
"""
Sample citations from the verification output for manual validation.

Produces a CSV with columns for the pipeline's automated label and a blank
column for the human annotator to fill in, enabling inter-rater agreement
analysis between the automated pipeline and human judgment.

Usage:
    python scripts/sample_for_manual_validation.py \
        --citations out/verify/citations.jsonl \
        --out manual_validation_100.csv \
        --n 100 \
        --seed 42
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from typing import Dict, List


def read_citations(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stratified_sample(
    citations: List[Dict],
    n: int,
    seed: int,
) -> List[Dict]:
    """
    Stratified sampling by (model, label) to ensure representation across
    all models and all three outcome categories (EXISTS, AMBIGUOUS, FABRICATED).

    Within each stratum, citations are shuffled and drawn proportionally.
    If a stratum is too small, all its citations are included and the
    remaining budget is redistributed to larger strata.
    """
    rng = random.Random(seed)

    # Group by (model, label)
    strata: Dict[tuple, List[Dict]] = defaultdict(list)
    for c in citations:
        key = (c.get("model", "unknown"), c.get("label", "UNKNOWN"))
        strata[key].append(c)

    # Shuffle within each stratum
    for key in strata:
        rng.shuffle(strata[key])

    # Proportional allocation with redistribution
    total = len(citations)
    allocation = {}
    for key, group in strata.items():
        allocation[key] = max(1, round(n * len(group) / total))

    # Redistribute if over-allocated or under-allocated
    sampled = {}
    remaining_budget = n
    remaining_keys = list(strata.keys())

    # First pass: handle strata smaller than their allocation
    for key in list(remaining_keys):
        alloc = allocation[key]
        available = len(strata[key])
        take = min(alloc, available)
        sampled[key] = strata[key][:take]
        remaining_budget -= take
        if take >= available:
            remaining_keys.remove(key)

    # Second pass: fill remaining budget from larger strata
    while remaining_budget > 0 and remaining_keys:
        per_key = max(1, remaining_budget // len(remaining_keys))
        made_progress = False
        for key in list(remaining_keys):
            already = len(sampled.get(key, []))
            available = len(strata[key])
            can_take = min(per_key, available - already, remaining_budget)
            if can_take > 0:
                sampled[key] = strata[key][:already + can_take]
                remaining_budget -= can_take
                made_progress = True
            if len(sampled[key]) >= available:
                remaining_keys.remove(key)
        if not made_progress:
            break

    # Flatten and shuffle final sample
    result = []
    for group in sampled.values():
        result.extend(group)
    rng.shuffle(result)
    return result[:n]


def format_row(idx: int, c: Dict) -> Dict:
    """Format a citation record into a CSV row for manual annotation."""
    parsed = c.get("parsed", {})
    authors_str = "; ".join(parsed.get("authors", []) or [])
    return {
        "idx": idx,
        "citation_id": c.get("citation_id", ""),
        "model": c.get("model", ""),
        "condition": c.get("condition", ""),
        "pipeline_label": c.get("label", ""),
        "pipeline_score": c.get("confidence", ""),
        "human_label": "",  # blank for annotator
        "agree": "",        # blank; filled after annotation
        "title": parsed.get("title", ""),
        "authors": authors_str,
        "year": parsed.get("year", ""),
        "venue": parsed.get("venue", ""),
        "reasoning": "",    # blank for annotator notes
    }


FIELDNAMES = [
    "idx",
    "citation_id",
    "model",
    "condition",
    "pipeline_label",
    "pipeline_score",
    "human_label",
    "agree",
    "title",
    "authors",
    "year",
    "venue",
    "reasoning",
]


def main():
    parser = argparse.ArgumentParser(
        description="Sample citations for manual validation."
    )
    parser.add_argument(
        "--citations",
        default="out/verify/citations.jsonl",
        help="Path to citation-level verification results (default: out/verify/citations.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="manual_validation_100.csv",
        help="Output CSV path (default: manual_validation_100.csv)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of citations to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    citations = read_citations(args.citations)
    print(f"Loaded {len(citations)} citations from {args.citations}")

    # Print stratum sizes
    strata_counts = defaultdict(int)
    for c in citations:
        key = (c.get("model", "unknown"), c.get("label", "UNKNOWN"))
        strata_counts[key] += 1
    print("\nStratum sizes (model, label):")
    for (model, label), count in sorted(strata_counts.items()):
        print(f"  {model:45s} {label:12s} {count:5d}")

    sample = stratified_sample(citations, args.n, args.seed)
    print(f"\nSampled {len(sample)} citations")

    # Print sample distribution
    sample_counts = defaultdict(int)
    for c in sample:
        key = (c.get("model", "unknown"), c.get("label", "UNKNOWN"))
        sample_counts[key] += 1
    print("\nSample distribution (model, label):")
    for (model, label), count in sorted(sample_counts.items()):
        print(f"  {model:45s} {label:12s} {count:5d}")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, c in enumerate(sample, start=1):
            writer.writerow(format_row(i, c))

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
