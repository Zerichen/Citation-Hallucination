from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

@dataclass
class Citation:
    citation_id: str
    run_id: str
    index: int
    raw: str

    title: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    # Normalized fields
    norm_title: Optional[str] = None
    norm_authors_last: Optional[List[str]] = None
    norm_venue: Optional[str] = None
    norm_doi: Optional[str] = None

@dataclass
class MatchResult:
    source: str  # "crossref" | "s2" | "openalex"
    score: float
    record: Dict[str, Any]

@dataclass
class VerificationResult:
    citation_id: str
    label: str  # EXISTS | AMBIGUOUS | FABRICATED
    confidence: float
    best_match: Optional[MatchResult]
    supporting_matches: List[MatchResult]
    error_type: Optional[str] = None  # fabricated | compositional | wrong-attribution | temporal
    notes: Optional[str] = None

def to_dict(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
