from typing import Dict, List, Optional, Tuple
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
    # temporal violation is a separate flag/type; still may exist
    temp_v = temporal_violation(c, time_window)

    if best is None:
        # if DOI was provided but lookup failed, strong evidence of fabrication
        label = "FABRICATED" if (c.norm_doi and doi_lookup_failed) else "FABRICATED"
        et = "temporal" if temp_v else "fabricated"
        return VerificationResult(
            citation_id=c.citation_id,
            label=label,
            confidence=0.0,
            best_match=None,
            supporting_matches=supporting,
            error_type=et,
            notes="no candidates matched"
        )

    if best.score >= EXISTS_TH:
        label = "EXISTS"
        conf = best.score
    elif best.score >= AMBIG_TH:
        label = "AMBIGUOUS"
        conf = best.score
    else:
        label = "FABRICATED"
        conf = best.score

    # error typing beyond label
    et = None
    if temp_v:
        et = "temporal"
    elif label == "FABRICATED":
        et = "fabricated"
    elif label in ("EXISTS", "AMBIGUOUS"):
        # compositional/wrong-attribution heuristic: title matches but authors/year poor
        # if best.score is driven mostly by title but other metadata mismatch, you can tag compositional
        # Here: if score is moderate and year mismatch + author overlap low => compositional/wrong-attribution
        pass

    return VerificationResult(
        citation_id=c.citation_id,
        label=label,
        confidence=conf,
        best_match=best,
        supporting_matches=supporting,
        error_type=et,
        notes=None
    )
