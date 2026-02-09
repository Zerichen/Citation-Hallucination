import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.parser import parse_citations, build_citation_objects
from src.normalize import normalize_citation, normalize_title, normalize_venue
from src.clients import CrossrefClient, SemanticScholarClient
from src.match import best_match
from src.label import assign_label
from src.aggregate import aggregate_run
from src.schema import to_dict

def adapt_crossref_item(item: Dict) -> Dict:
    title = (item.get("title") or [""])[0]
    authors = []
    for a in item.get("author", []) or []:
        given = a.get("given", "") or ""
        family = a.get("family", "") or ""
        name = (given + " " + family).strip() if (given or family) else ""
        if name:
            authors.append(name)
    year = None
    issued = item.get("issued", {}).get("date-parts")
    if issued and issued[0] and issued[0][0]:
        year = int(issued[0][0])
    venue = item.get("container-title", [""])
    venue = venue[0] if venue else ""
    doi = item.get("DOI")
    return {
        "title": title,
        "norm_title": normalize_title(title),
        "authors": authors,
        "year": year,
        "venue": venue,
        "norm_venue": normalize_venue(venue),
        "doi": doi
    }

def adapt_s2_item(item: Dict) -> Dict:
    title = item.get("title", "") or ""
    authors = [a.get("name", "") for a in item.get("authors", []) or [] if a.get("name")]
    year = item.get("year")
    venue = item.get("venue", "") or ""
    ext = item.get("externalIds") or {}
    doi = ext.get("DOI")
    return {
        "title": title,
        "norm_title": normalize_title(title),
        "authors": authors,
        "year": year,
        "venue": venue,
        "norm_venue": normalize_venue(venue),
        "doi": doi
    }

def _verify_file(
    runs_path: str,
    out_dir: str,
    mailto: Optional[str],
    s2_key: Optional[str],
    k: int,
    *,
    citations_path_override: Optional[str] = None,
    metrics_path_override: Optional[str] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    citations_path = citations_path_override or os.path.join(out_dir, "citations.jsonl")
    metrics_path = metrics_path_override or os.path.join(out_dir, "run_metrics.jsonl")

    resume = os.getenv("RESUME", "").strip().lower() in {"1", "true", "yes"}
    completed_run_ids = set()
    if resume and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_run_ids.add(json.loads(line).get("run_id"))
                except Exception:
                    continue

    crossref = CrossrefClient(mailto=mailto, cache_dir="cache")
    s2 = SemanticScholarClient(api_key=s2_key, cache_dir="cache")

    # Load runs
    runs = []
    with open(runs_path, "r", encoding="utf-8") as f:
        for line in f:
            runs.append(json.loads(line))

    with open(citations_path, "a", encoding="utf-8") as cit_out, \
         open(metrics_path, "a", encoding="utf-8") as met_out:

        for run in tqdm(runs, desc=f"Verifying runs ({os.path.basename(runs_path)})"):
            run_id = run["run_id"]
            if resume and run_id in completed_run_ids:
                continue
            output_text = run.get("output", "")
            time_window = run.get("time_window")

            blocks = parse_citations(run_id, output_text)
            citations = build_citation_objects(run_id, blocks)
            citations = [normalize_citation(c) for c in citations]

            per_run_results = []

            for c in citations:
                candidates: List[Tuple[str, Dict]] = []
                doi_failed = False

                # 1) DOI lookup (Crossref) if provided
                if c.norm_doi:
                    try:
                        rec = crossref.lookup_doi(c.norm_doi)
                    except Exception as e:
                        print(f"[WARN] Crossref DOI lookup failed: {c.norm_doi} ({e})")
                        rec = None
                    if rec is None:
                        doi_failed = True
                    else:
                        candidates.append(("crossref", adapt_crossref_item(rec)))

                # 2) Title search if title exists (or DOI failed)
                if c.norm_title:
                    # Semantic Scholar
                    try:
                        s2_items = s2.search_title(c.title, limit=k) if c.title else []
                    except Exception as e:
                        print(f"[WARN] S2 title search failed: {c.title} ({e})")
                        s2_items = []
                    for it in s2_items:
                        candidates.append(("s2", adapt_s2_item(it)))
                    # Crossref
                    try:
                        cr_items = crossref.search_title(c.title, rows=k) if c.title else []
                    except Exception as e:
                        print(f"[WARN] Crossref title search failed: {c.title} ({e})")
                        cr_items = []
                    for it in cr_items:
                        candidates.append(("crossref", adapt_crossref_item(it)))

                bm = best_match(c, candidates)
                # keep top supporting matches (e.g., top 3 by score)
                supporting = []
                if candidates:
                    scored = []
                    for src, rec in candidates:
                        # reuse best_match scoring by calling best_match is expensive; keep it simple:
                        # approximate: just store those with same as best or take first few
                        # If you want: compute exact scores by importing score_record.
                        scored.append((src, rec))
                    # For simplicity, store empty here; you can extend to store top-N matches with scores.
                    supporting = []

                vr = assign_label(c, bm, supporting, time_window=time_window, doi_lookup_failed=doi_failed)
                vr_dict = to_dict(vr)
                vr_dict["run_id"] = run_id
                vr_dict["condition"] = run.get("condition")
                vr_dict["topic_id"] = run.get("topic_id")
                vr_dict["model"] = run.get("model")
                vr_dict["temperature"] = run.get("temperature")
                vr_dict["parsed"] = {
                    "title": c.title,
                    "authors": c.authors,
                    "venue": c.venue,
                    "year": c.year,
                    "doi": c.norm_doi
                }
                # Add canonical metadata if exists
                if vr.best_match:
                    vr_dict["canonical"] = {
                        "source": vr.best_match.source,
                        "title": vr.best_match.record.get("title"),
                        "authors": vr.best_match.record.get("authors"),
                        "venue": vr.best_match.record.get("venue"),
                        "year": vr.best_match.record.get("year"),
                        "doi": vr.best_match.record.get("doi"),
                        "score": vr.best_match.score
                    }
                else:
                    vr_dict["canonical"] = None

                cit_out.write(json.dumps(vr_dict, ensure_ascii=False) + "\n")
                per_run_results.append(vr_dict)

            # aggregate run metrics
            run_metrics = aggregate_run(per_run_results)
            # Preserve all keys from input run; keep aggregate fields if any collisions.
            run_metrics = {**run, **run_metrics}
            met_out.write(json.dumps(run_metrics, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="path to runs.jsonl")
    ap.add_argument("--out_dir", default="out/verify", help="output directory")
    ap.add_argument("--mailto", default=None, help="mailto for Crossref polite requests")
    ap.add_argument("--s2_key", default=None, help="Semantic Scholar API key (optional)")
    ap.add_argument("--k", type=int, default=5, help="top-k candidates per source")
    ap.add_argument("--clear", action="store_true", help="clear output jsonl files before writing")
    args = ap.parse_args()

    if args.runs:
        if args.clear:
            citations_path = os.path.join(args.out_dir, "citations.jsonl")
            metrics_path = os.path.join(args.out_dir, "run_metrics.jsonl")
            os.makedirs(args.out_dir, exist_ok=True)
            if os.path.exists(citations_path):
                os.remove(citations_path)
            if os.path.exists(metrics_path):
                os.remove(metrics_path)
        _verify_file(args.runs, args.out_dir, args.mailto, args.s2_key, args.k)
        return

    out_root = "out"
    runs_files = [
        os.path.join(out_root, name)
        for name in os.listdir(out_root)
        if name.endswith("_full_runs_result.jsonl")
    ]
    if not runs_files:
        raise SystemExit("No *_full_runs.jsonl files found in out/")

    citations_path = os.path.join(args.out_dir, "citations.jsonl")
    metrics_path = os.path.join(args.out_dir, "run_metrics.jsonl")
    if args.clear:
        os.makedirs(args.out_dir, exist_ok=True)
        if os.path.exists(citations_path):
            os.remove(citations_path)
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
    for runs_path in sorted(runs_files):
        stem = os.path.splitext(os.path.basename(runs_path))[0]
        out_dir = args.out_dir
        _verify_file(
            runs_path,
            out_dir,
            args.mailto,
            args.s2_key,
            args.k,
            citations_path_override=citations_path,
            metrics_path_override=metrics_path,
        )

if __name__ == "__main__":
    main()
