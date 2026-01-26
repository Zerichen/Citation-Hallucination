import argparse
import json
import os
from typing import Dict, List, Tuple
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="path to runs.jsonl")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--mailto", default=None, help="mailto for Crossref polite requests")
    ap.add_argument("--s2_key", default=None, help="Semantic Scholar API key (optional)")
    ap.add_argument("--k", type=int, default=5, help="top-k candidates per source")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    citations_path = os.path.join(args.out_dir, "citations.jsonl")
    metrics_path = os.path.join(args.out_dir, "run_metrics.jsonl")

    crossref = CrossrefClient(mailto=args.mailto, cache_dir="cache")
    s2 = SemanticScholarClient(api_key=args.s2_key, cache_dir="cache")

    # Load runs
    runs = []
    with open(args.runs, "r", encoding="utf-8") as f:
        for line in f:
            runs.append(json.loads(line))

    with open(citations_path, "a", encoding="utf-8") as cit_out, \
         open(metrics_path, "a", encoding="utf-8") as met_out:

        for run in tqdm(runs, desc="Verifying runs"):
            run_id = run["run_id"]
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
                    rec = crossref.lookup_doi(c.norm_doi)
                    if rec is None:
                        doi_failed = True
                    else:
                        candidates.append(("crossref", adapt_crossref_item(rec)))

                # 2) Title search if title exists (or DOI failed)
                if c.norm_title:
                    # Semantic Scholar
                    s2_items = s2.search_title(c.title, limit=args.k) if c.title else []
                    for it in s2_items:
                        candidates.append(("s2", adapt_s2_item(it)))
                    # Crossref
                    cr_items = crossref.search_title(c.title, rows=args.k) if c.title else []
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
            run_metrics.update({
                "run_id": run_id,
                "condition": run.get("condition"),
                "topic_id": run.get("topic_id"),
                "model": run.get("model"),
                "temperature": run.get("temperature")
            })
            met_out.write(json.dumps(run_metrics, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
