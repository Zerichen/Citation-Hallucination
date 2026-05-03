import re
from typing import List, Optional
from .schema import Citation

def normalize_citation(c: Citation) -> Citation:
    c.norm_title = normalize_title(c.title)
    c.norm_authors_last = normalize_authors_last(c.authors or [])
    c.norm_venue = normalize_venue(c.venue)
    c.norm_doi = normalize_doi(c.doi)
    return c

def normalize_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = title.lower()
    t = re.sub(r"[\[\]{}()\"'`]", " ", t)
    t = re.sub(r"[^a-z0-9\s:.-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_authors_last(authors: List[str]) -> List[str]:
    last = []
    for a in authors:
        a = re.sub(r"\s+", " ", a.strip())
        if not a:
            continue
        # Take last token as last name (robust enough)
        toks = a.split(" ")
        ln = toks[-1].lower()
        ln = re.sub(r"[^a-z-]", "", ln)
        if ln:
            last.append(ln)
    # de-dup while preserving order
    seen = set()
    out = []
    for ln in last:
        if ln not in seen:
            out.append(ln)
            seen.add(ln)
    return out

def normalize_venue(venue: Optional[str]) -> Optional[str]:
    if not venue:
        return None
    v = venue.lower().strip()
    v = re.sub(r"\s+", " ", v)
    # basic aliases
    v = v.replace("neurips", "nips").replace("neural information processing systems", "nips")
    v = v.replace("empirical methods in natural language processing", "emnlp")
    v = v.replace("association for computational linguistics", "acl")
    return v

def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip().lower()
    d = d.replace("https://doi.org/", "").replace("http://doi.org/", "")
    d = d.replace("doi:", "").strip()
    return d or None
