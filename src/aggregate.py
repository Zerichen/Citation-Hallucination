from typing import Dict, List
import math

def aggregate_run(citation_results: List[Dict]) -> Dict:
    """
    citation_results: list of dicts with keys label, error_type, confidence
    """
    n = len(citation_results)
    if n == 0:
        return {
            "num_citations": 0,
            "exists_rate": math.nan,
            "fabricated_rate": math.nan,
            "ambiguous_rate": math.nan,
            "temporal_violation_rate": math.nan
        }
    counts = {"EXISTS": 0, "FABRICATED": 0, "AMBIGUOUS": 0}
    temporal = 0
    for r in citation_results:
        counts[r["label"]] += 1
        if r.get("error_type") == "temporal":
            temporal += 1
    return {
        "num_citations": n,
        "exists_rate": counts["EXISTS"] / n,
        "fabricated_rate": counts["FABRICATED"] / n,
        "ambiguous_rate": counts["AMBIGUOUS"] / n,
        "temporal_violation_rate": temporal / n
    }
