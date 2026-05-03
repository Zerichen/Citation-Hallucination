from typing import Dict, List, Optional
from .schema import Citation, VerificationResult, MatchResult

EXISTS_TH = 0.85
AMBIG_TH = 0.60

def temporal_violation(c: Citation, time_window: Optional[Dict]) -> bool:
    if not time_window or not c.year:
        return False
    sy = time_window.get("start_year")
    ey = time_window.get("end_year")
    if sy and c.year < sy:
        return True
    if ey and c.year > ey:
        return True
    return False

def assign_label(
    c: Citation,
    best: Optional[MatchResult],
    supporting: List[MatchResult],
    time_window: Optional[Dict] = None,
    doi_lookup_failed: bool = False
) -> VerificationResult:
    temp_v = temporal_violation(c, time_window)

    if best is None:
        return VerificationResult(
            citation_id=c.citation_id,
            label="FABRICATED",
            confidence=0.0,
            best_match=None,
            supporting_matches=supporting,
            error_type="temporal" if temp_v else "fabricated",
            notes="no candidates matched"
        )

    if best.score >= EXISTS_TH:
        label = "EXISTS"
    elif best.score >= AMBIG_TH:
        label = "AMBIGUOUS"
    else:
        label = "FABRICATED"

    et = None
    if temp_v:
        et = "temporal"
    elif label == "FABRICATED":
        et = "fabricated"

    return VerificationResult(
        citation_id=c.citation_id,
        label=label,
        confidence=best.score,
        best_match=best,
        supporting_matches=supporting,
        error_type=et,
        notes=None
    )
