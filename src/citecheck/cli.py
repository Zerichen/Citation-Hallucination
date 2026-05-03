"""Command-line interface for citecheck.

Usage:
    citecheck verify <input.jsonl> [--output result.jsonl] [--cache cache/]
    citecheck info
    citecheck --version
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__


def _cmd_verify(args: argparse.Namespace) -> int:
    from .verify import verify_file

    if not os.path.exists(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    output = args.output
    results = verify_file(
        args.input,
        output_path=output,
        cache_dir=args.cache,
        mailto=args.mailto,
        s2_api_key=args.s2_key,
        k=args.k,
    )

    counts = {"EXISTS": 0, "AMBIGUOUS": 0, "FABRICATED": 0}
    for r in results:
        label = r.get("label", "FABRICATED")
        counts[label] = counts.get(label, 0) + 1

    if not output:
        for r in results:
            print(json.dumps(r, ensure_ascii=False))

    n = len(results)
    summary = (
        f"Verified {n} citation(s): "
        f"{counts.get('EXISTS', 0)} EXISTS, "
        f"{counts.get('AMBIGUOUS', 0)} AMBIGUOUS, "
        f"{counts.get('FABRICATED', 0)} FABRICATED"
    )
    if output:
        summary += f" -> {output}"
    print(summary, file=sys.stderr)
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    print(f"citecheck {__version__}")
    print("Defensive verification of LLM-generated bibliographic citations.")
    print("Sources: Crossref, Semantic Scholar.")
    print("Labels: EXISTS (>=0.85), AMBIGUOUS (>=0.60), FABRICATED (<0.60).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citecheck",
        description="Verify bibliographic citations against scholarly databases.",
    )
    parser.add_argument(
        "--version", action="version", version=f"citecheck {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", metavar="command")

    p_verify = sub.add_parser(
        "verify", help="Verify citations in a JSONL file"
    )
    p_verify.add_argument(
        "input",
        type=str,
        help='Input JSONL file (one record per line: {"title": ..., "authors": [...], "year": ..., "doi": ...})',
    )
    p_verify.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSONL results to this path (default: stdout)",
    )
    p_verify.add_argument(
        "--cache",
        type=str,
        default="cache",
        help="Directory for HTTP response cache (default: ./cache)",
    )
    p_verify.add_argument(
        "--mailto",
        type=str,
        default=None,
        help="Email for Crossref polite-pool requests (optional)",
    )
    p_verify.add_argument(
        "--s2-key",
        dest="s2_key",
        type=str,
        default=None,
        help="Semantic Scholar API key (optional)",
    )
    p_verify.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k candidates per source (default: 5)",
    )
    p_verify.set_defaults(func=_cmd_verify)

    p_info = sub.add_parser("info", help="Print package info")
    p_info.set_defaults(func=_cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
