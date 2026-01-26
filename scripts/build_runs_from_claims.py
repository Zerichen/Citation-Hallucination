import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.prompts import render_prompt

OUT_PATH = "data/claude_sonnet_3.5.jsonl"
CLAIMS_PATH = "data/claims.csv"

MODEL = os.getenv("OPENAI_MODEL", "claude-3-5-sonnet-2024")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

CONDITIONS: List[str] = ["baseline", "temporal", "survey", "privacy", "combo"]
FIXED_CITES = {
    "baseline": 5,
    "temporal": 5,
    "privacy": 5,
    "survey": 8,
    "combo": 8,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required_field(row: Dict[str, str], key: str, row_idx: int) -> str:
    val = (row.get(key) or "").strip()
    if not val:
        raise ValueError(f"Missing required field '{key}' at row {row_idx}")
    return val


def _optional_field(row: Dict[str, str], key: str) -> Optional[str]:
    val = (row.get(key) or "").strip()
    return val or None


def _parse_int(value: Optional[str], key: str, row_idx: int, default: Optional[int] = None) -> int:
    if value is None or value == "":
        if default is None:
            raise ValueError(f"Missing required field '{key}' at row {row_idx}")
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Invalid int for '{key}' at row {row_idx}: {value}")


def _topic_id(row_idx: int, claim_id: Optional[str]) -> str:
    if not claim_id:
        return f"t{row_idx - 1:03d}"
    cid = claim_id.strip()
    if cid.isdigit():
        return f"t{int(cid):03d}"
    if cid.startswith("t") and cid[1:].isdigit():
        return cid
    return f"t_{cid}"


def _question_with_domain(question: str, domain: Optional[str]) -> str:
    q = question.strip()
    if domain:
        q = f"{q}\n\nDomain: {domain}"
    return q


def main() -> int:
    if not os.path.exists(CLAIMS_PATH):
        print(f"[ERROR] missing {CLAIMS_PATH}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    runs: List[Dict] = []
    with open(CLAIMS_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("[ERROR] claims.csv has no header", file=sys.stderr)
            return 1
        for i, row in enumerate(reader, start=2):
            question = _required_field(row, "question", i)
            domain = _optional_field(row, "domain")
            seed_anchors = _optional_field(row, "seed_anchors")
            temporal_start_year = _optional_field(row, "temporal_start_year")
            temporal_end_year = _optional_field(row, "temporal_end_year")
            _parse_int(_optional_field(row, "n_papers"), "n_papers", i, default=10)
            claim_id = _optional_field(row, "claim_id")

            topic_id = _topic_id(i, claim_id)
            q = _question_with_domain(question, domain)

            for cond in CONDITIONS:
                n_papers = FIXED_CITES[cond]
                start_year = None
                end_year = None
                if cond == "temporal":
                    start_year = _parse_int(temporal_start_year, "temporal_start_year", i)
                    end_year = _parse_int(temporal_end_year, "temporal_end_year", i)
                prompt = render_prompt(
                    cond,
                    question=q,
                    n_papers=n_papers,
                    start_year=start_year,
                    end_year=end_year,
                    seed_anchors=seed_anchors,
                )
                run = {
                    "run_id": str(uuid.uuid4()),
                    "timestamp": utc_now_iso(),
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "condition": cond,
                    "topic_id": topic_id,
                    "question": q,
                    "prompt": prompt,
                    "output": "",
                }
                if cond == "temporal":
                    run["time_window"] = {"start_year": start_year, "end_year": end_year}
                runs.append(run)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(runs)} runs to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
