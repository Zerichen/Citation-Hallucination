"""High-level verification API.

`verify_citation` and `verify_file` wrap the parser/normalize/match/label
pipeline into a single callable. They accept structured citation records
(dicts with title/authors/venue/year/doi keys), so callers do not need to
preformat output text the way the LLM-prompted runs do.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .clients import CrossrefClient, SemanticScholarClient
from .label import assign_label
from .match import best_match
from .normalize import normalize_citation, normalize_title, normalize_venue
from .schema import Citation, MatchResult, VerificationResult, to_dict


def _adapt_crossref_item(item: Dict[str, Any]) -> Dict[str, Any]:
    title_list = item.get("title") or [""]
    title = title_list[0] if title_list else ""
    authors: List[str] = []
    for a in item.get("author", []) or []:
        given = a.get("given", "") or ""
        family = a.get("family", "") or ""
        name = (given + " " + family).strip() if (given or family) else ""
        if name:
            authors.append(name)
    year = None
    issued = (item.get("issued") or {}).get("date-parts")
    if issued and issued[0] and issued[0][0]:
        year = int(issued[0][0])
    venue_list = item.get("container-title") or [""]
    venue = venue_list[0] if venue_list else ""
    return {
        "title": title,
        "norm_title": normalize_title(title),
        "authors": authors,
        "year": year,
        "venue": venue,
        "norm_venue": normalize_venue(venue),
        "doi": item.get("DOI"),
    }


def _adapt_s2_item(item: Dict[str, Any]) -> Dict[str, Any]:
    title = item.get("title") or ""
    authors = [a.get("name", "") for a in item.get("authors", []) or [] if a.get("name")]
    year = item.get("year")
    venue = item.get("venue") or ""
    ext = item.get("externalIds") or {}
    return {
        "title": title,
        "norm_title": normalize_title(title),
        "authors": authors,
        "year": year,
        "venue": venue,
        "norm_venue": normalize_venue(venue),
        "doi": ext.get("DOI"),
    }


def _record_to_citation(record: Dict[str, Any], citation_id: str) -> Citation:
    title = record.get("title")
    authors = record.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.replace(";", ",").split(",") if a.strip()]
    raw = record.get("raw") or json.dumps(record, ensure_ascii=False)
    return Citation(
        citation_id=citation_id,
        run_id=record.get("run_id", "user"),
        index=int(record.get("index", 0) or 0),
        raw=raw,
        title=title,
        authors=list(authors) if authors else None,
        venue=record.get("venue") or record.get("journal") or record.get("conference"),
        year=int(record["year"]) if record.get("year") not in (None, "") else None,
        doi=record.get("doi"),
        url=record.get("url"),
    )


def verify_citation(
    record: Dict[str, Any],
    *,
    crossref: Optional[CrossrefClient] = None,
    s2: Optional[SemanticScholarClient] = None,
    cache_dir: str = "cache",
    mailto: Optional[str] = None,
    s2_api_key: Optional[str] = None,
    k: int = 5,
    citation_id: str = "user_c01",
    time_window: Optional[Dict[str, int]] = None,
) -> VerificationResult:
    """Verify a single citation record.

    `record` keys (all optional except at least one of title/doi):
        title, authors (list[str] or "A; B"), venue, year, doi, url
    """
    if crossref is None:
        crossref = CrossrefClient(mailto=mailto, cache_dir=cache_dir)
    if s2 is None:
        s2 = SemanticScholarClient(api_key=s2_api_key, cache_dir=cache_dir)

    c = _record_to_citation(record, citation_id)
    c = normalize_citation(c)

    candidates: List[Tuple[str, Dict[str, Any]]] = []
    doi_failed = False

    if c.norm_doi:
        try:
            rec = crossref.lookup_doi(c.norm_doi)
        except Exception:
            rec = None
        if rec is None:
            doi_failed = True
        else:
            candidates.append(("crossref", _adapt_crossref_item(rec)))

    if c.title:
        try:
            s2_items = s2.search_title(c.title, limit=k)
        except Exception:
            s2_items = []
        for it in s2_items:
            candidates.append(("s2", _adapt_s2_item(it)))
        try:
            cr_items = crossref.search_title(c.title, rows=k)
        except Exception:
            cr_items = []
        for it in cr_items:
            candidates.append(("crossref", _adapt_crossref_item(it)))

    bm = best_match(c, candidates)
    return assign_label(c, bm, [], time_window=time_window, doi_lookup_failed=doi_failed)


def _verification_to_record(
    record: Dict[str, Any], result: VerificationResult
) -> Dict[str, Any]:
    out = to_dict(result)
    out["input"] = record
    if result.best_match is not None:
        out["canonical"] = {
            "source": result.best_match.source,
            "title": result.best_match.record.get("title"),
            "authors": result.best_match.record.get("authors"),
            "venue": result.best_match.record.get("venue"),
            "year": result.best_match.record.get("year"),
            "doi": result.best_match.record.get("doi"),
            "score": result.best_match.score,
        }
    else:
        out["canonical"] = None
    return out


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {i} of {path}: {e}") from e


def verify_file(
    input_path: str,
    output_path: Optional[str] = None,
    cache_dir: str = "cache",
    *,
    mailto: Optional[str] = None,
    s2_api_key: Optional[str] = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Verify a JSONL file of citation records.

    Each input line is a JSON object with title/authors/venue/year/doi keys.
    Writes JSONL output (one verification record per input line) and returns
    the list of records.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    crossref = CrossrefClient(mailto=mailto, cache_dir=cache_dir)
    s2 = SemanticScholarClient(api_key=s2_api_key, cache_dir=cache_dir)

    records = list(_iter_jsonl(input_path))
    results: List[Dict[str, Any]] = []
    for i, record in enumerate(records, start=1):
        cid = record.get("citation_id") or f"user_c{i:03d}"
        time_window = record.get("time_window")
        vr = verify_citation(
            record,
            crossref=crossref,
            s2=s2,
            cache_dir=cache_dir,
            k=k,
            citation_id=cid,
            time_window=time_window,
        )
        out = _verification_to_record(record, vr)
        results.append(out)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results
