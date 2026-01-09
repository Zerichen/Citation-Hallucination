from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
from .schema import Citation, MatchResult

def title_similarity(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0

def year_match(c_year: Optional[int], r_year: Optional[int]) -> float:
    if not c_year or not r_year:
        return 0.0
    if c_year == r_year:
        return 1.0
    if abs(c_year - r_year) == 1:
        return 0.5
    return 0.0

def author_overlap(c_last: List[str], r_authors: List[str]) -> float:
    if not c_last or not r_authors:
        return 0.0
    r_last = set([a.lower().split(" ")[-1] for a in r_authors if a])
    inter = 0
    for ln in c_last:
        if ln in r_last:
            inter += 1
    return min(1.0, inter / max(1, min(2, len(c_last))))

def venue_similarity(c_venue: Optional[str], r_venue: Optional[str]) -> float:
    if not c_venue or not r_venue:
        return 0.0
    return fuzz.partial_ratio(c_venue, r_venue) / 100.0

def score_record(c: Citation, rec: Dict, source: str) -> float:
    """
    rec is canonicalized per source adapter below.
    """
    t_sim = title_similarity(c.norm_title, rec.get("norm_title"))
    y = year_match(c.year, rec.get("year"))
    a = author_overlap(c.norm_authors_last or [], rec.get("authors", []))
    v = venue_similarity(c.norm_venue, rec.get("norm_venue"))

    # weights: title 0.6, authors 0.2, year 0.15, venue 0.05
    return 0.6 * t_sim + 0.2 * a + 0.15 * y + 0.05 * v

def best_match(c: Citation, candidates: List[Tuple[str, Dict]]) -> Optional[MatchResult]:
    best = None
    for source, rec in candidates:
        s = score_record(c, rec, source)
        mr = MatchResult(source=source, score=s, record=rec)
        if best is None or mr.score > best.score:
            best = mr
    return best
