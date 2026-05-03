import re
from typing import List, Dict, Optional
from .schema import Citation

BLOCK_RE = re.compile(r"(?m)^\s*\[(\d+)\]\s*(.+?)(?=^\s*\[\d+\]|\Z)", re.S)
KV_RE = re.compile(r"(?im)^\s*(Title|Authors?|Venue|Journal|Conference|Year|DOI|URL)\s*:\s*(.+?)\s*$")

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def parse_citations(run_id: str, output_text: str) -> List[Dict]:
    """
    Returns list of dict blocks: {index:int, raw:str, fields:dict}
    """
    blocks = []
    m = list(BLOCK_RE.finditer(output_text))
    if m:
        for mm in m:
            idx = int(mm.group(1))
            raw_block = mm.group(0).strip()
            body = mm.group(2).strip()
            fields = {}
            for kvm in KV_RE.finditer(body):
                key = kvm.group(1).lower()
                val = _clean(kvm.group(2))
                fields[key] = val
            blocks.append({"index": idx, "raw": raw_block, "fields": fields})
        return blocks

    # Fallback: try to detect DOI-like patterns and treat each as a "citation"
    doi_pat = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
    dois = doi_pat.findall(output_text)
    for i, d in enumerate(dois[:20], start=1):
        blocks.append({"index": i, "raw": d, "fields": {"doi": d}})
    return blocks

def build_citation_objects(run_id: str, blocks: List[Dict]) -> List[Citation]:
    out: List[Citation] = []
    for j, b in enumerate(blocks, start=1):
        idx = b.get("index", j)
        fields = b.get("fields", {})
        c = Citation(
            citation_id=f"{run_id}_c{idx:02d}",
            run_id=run_id,
            index=idx,
            raw=b.get("raw", "")
        )
        # Map fields
        c.title = fields.get("title")
        authors = fields.get("author") or fields.get("authors")
        if authors:
            # split on comma/and
            parts = re.split(r"\s*(?:,|;| and )\s*", authors.strip(), flags=re.I)
            c.authors = [p.strip() for p in parts if p.strip()]
        c.year = _parse_year(fields.get("year"))
        c.venue = fields.get("venue") or fields.get("journal") or fields.get("conference")
        c.doi = fields.get("doi")
        c.url = fields.get("url")
        out.append(c)
    return out

def _parse_year(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None
