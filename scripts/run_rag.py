#!/usr/bin/env python3
"""Retrieval-augmented (RAG) generation for the DeLTA 2026 camera-ready.

Minimal RAG baseline requested by Reviewer R5: take the SAME 144 claims used in
the closed-book study, run Claude Sonnet 4.5 under the Baseline and Temporal
conditions only, but inject top-5 Crossref candidate references into the prompt
before the citation-format block. Everything else (system prompt, output format,
citation count, temperature 0, max_tokens 2048) is identical to closed-book.

The 144 topic_ids, their question text, and the Temporal year windows are read
back from the existing closed-book artifacts so the comparison is apples-to-apples
(no re-sampling). Seed anchors (used to build the retrieval query) come from
data/claims.csv.

Outputs:
    out/rag/claude-sonnet-4-5-20250929_rag_runs.jsonl   (one row per run)
    out/rag/run_config.json                             (reproducibility metadata)
    cache/crossref_rag/queries.jsonl                    (cached Crossref responses)

The output rows are a superset of the closed-book `out/<model>_full_runs.jsonl`
schema (same core keys + a `retrieval` block), so they flow through the existing
verification pipeline unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from citecheck.clients import JsonlCache  # noqa: E402
from citecheck.prompts import render_prompt  # noqa: E402

# ---- Fixed experiment configuration (matches closed-book) --------------------
MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.0
MAX_TOKENS = 2048
SYSTEM_PROMPT = "You are a research assistant."
CONDITIONS = ["baseline", "temporal"]
FIXED_CITES = {"baseline": 5, "temporal": 5}
TOP_K = 5
DEFAULT_MAILTO = "zerichenn@gmail.com"

CLAIMS_PATH = os.path.join(ROOT_DIR, "data", "claims.csv")
CLOSED_BOOK_METRICS = os.path.join(ROOT_DIR, "out", "verify", "run_metrics.jsonl")
OUT_DIR = os.path.join(ROOT_DIR, "out", "rag")
OUT_PATH = os.path.join(OUT_DIR, f"{MODEL}_rag_runs.jsonl")
CONFIG_PATH = os.path.join(OUT_DIR, "run_config.json")
RAG_CACHE_DIR = os.path.join(ROOT_DIR, "cache", "crossref_rag")

CROSSREF_BASE = "https://api.crossref.org/works"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Topic loading -----------------------------------------------------------
def _topic_id(row_idx: int, claim_id: Optional[str]) -> str:
    """Replicates build_runs_from_claims._topic_id so the mapping matches."""
    if not claim_id:
        return f"t{row_idx - 1:03d}"
    cid = claim_id.strip()
    if cid.isdigit():
        return f"t{int(cid):03d}"
    if cid.startswith("t") and cid[1:].isdigit():
        return cid
    return f"t_{cid}"


def load_seed_anchors() -> Dict[str, str]:
    anchors: Dict[str, str] = {}
    with open(CLAIMS_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):
            cid = (row.get("claim_id") or "").strip() or None
            tid = _topic_id(i, cid)
            anchors[tid] = (row.get("seed_anchors") or "").strip()
    return anchors


def load_closed_book_topics() -> Dict[str, Dict[str, Any]]:
    """Return {topic_id: {question, time_window}} for the 144 closed-book claims.

    The question is taken from the Claude baseline run; the time_window from the
    Claude temporal run. Both reference the same topic, so the question is
    identical across conditions.
    """
    topics: Dict[str, Dict[str, Any]] = {}
    with open(CLOSED_BOOK_METRICS, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("model") != MODEL:
                continue
            cond = o.get("condition")
            if cond not in ("baseline", "temporal"):
                continue
            tid = o["topic_id"]
            entry = topics.setdefault(tid, {})
            if cond == "baseline":
                entry["question"] = o["question"]
            if cond == "temporal":
                entry.setdefault("question", o["question"])
                entry["time_window"] = o.get("time_window")
    return topics


# ---- Retrieval-query construction --------------------------------------------
_ANCHOR_PREFIX_RE = re.compile(r"(?i)^\s*anchor\s*\d*\s*:\s*")


def build_query(seed_anchors: str, question: str) -> str:
    """Build a Crossref keyword query from hand-curated seed anchors.

    Seed anchors look like:
        Anchor1: a; b; c; Justification: <prose>.
        Anchor2: d; e; Justification: <prose>.
    We keep the keyword tokens (the part before "Justification:"), drop the
    Anchor/Justification scaffolding, de-duplicate while preserving order, and
    join into one free-text query. Falls back to the question keywords when no
    anchors are available.
    """
    terms: List[str] = []
    seen = set()

    def _add(tok: str) -> None:
        tok = tok.strip()
        if not tok:
            return
        key = tok.lower()
        if key in seen:
            return
        seen.add(key)
        terms.append(tok)

    if seed_anchors and seed_anchors.strip():
        for raw_line in seed_anchors.splitlines():
            line = _ANCHOR_PREFIX_RE.sub("", raw_line.strip())
            # Drop the justification tail.
            line = re.split(r"(?i)\bjustification\s*:", line)[0]
            for tok in line.split(";"):
                _add(tok)

    if not terms:
        # Fallback: keywords from the question (strip the "Domain:" suffix line).
        q = question.split("\n\nDomain:")[0]
        q = re.sub(r"[^A-Za-z0-9\s]", " ", q)
        words = [w for w in q.split() if len(w) > 3]
        for w in words[:12]:
            _add(w)

    return " ".join(terms).strip()


# ---- Crossref retriever ------------------------------------------------------
def _extract_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    title_list = item.get("title") or [""]
    title = (title_list[0] if title_list else "") or ""
    last_names: List[str] = []
    for a in (item.get("author") or [])[:3]:
        fam = (a.get("family") or "").strip()
        if fam:
            last_names.append(fam)
        elif a.get("name"):
            last_names.append(a["name"].strip())
    venue_list = item.get("container-title") or [""]
    venue = (venue_list[0] if venue_list else "") or ""
    year = None
    issued = (item.get("issued") or {}).get("date-parts")
    if issued and issued[0] and issued[0][0]:
        try:
            year = int(issued[0][0])
        except (TypeError, ValueError):
            year = None
    return {
        "title": title.strip(),
        "authors": last_names,
        "venue": venue.strip(),
        "year": year,
        "doi": item.get("DOI"),
    }


class CrossrefRagRetriever:
    """Top-k Crossref keyword retriever with on-disk caching.

    Caches the raw Crossref `message.items` list keyed by the full request
    (query + rows + year filter) so retrieval is reproducible offline. Candidate
    extraction is deterministic from the cached items.
    """

    def __init__(self, mailto: str, cache_dir: str = RAG_CACHE_DIR, sleep_s: float = 0.2):
        self.mailto = mailto
        self.sleep_s = sleep_s
        self.cache = JsonlCache(os.path.join(cache_dir, "queries.jsonl"))

    @staticmethod
    def _cache_key(query: str, rows: int, year_filter: Optional[str]) -> str:
        return json.dumps(
            {"q": query, "rows": rows, "filter": year_filter or ""},
            ensure_ascii=False,
            sort_keys=True,
        )

    def retrieve(
        self,
        query: str,
        rows: int = TOP_K,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Dict[str, Any]:
        year_filter = None
        if start_year is not None and end_year is not None:
            year_filter = f"from-pub-date:{start_year},until-pub-date:{end_year}"

        key = self._cache_key(query, rows, year_filter)
        cached = self.cache.get(key)
        cache_hit = cached is not None
        if cached is None:
            params = {"query": query, "rows": rows, "mailto": self.mailto}
            if year_filter:
                params["filter"] = year_filter
            items: List[Dict[str, Any]] = []
            try:
                r = requests.get(CROSSREF_BASE, params=params, timeout=30)
                time.sleep(self.sleep_s)
                if r.status_code == 200:
                    items = r.json().get("message", {}).get("items", []) or []
                else:
                    print(f"[WARN] Crossref {r.status_code} for query={query!r} filter={year_filter}")
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Crossref request failed for query={query!r}: {e!r}")
                items = []
            self.cache.set(key, items)
            cached = items

        candidates = [_extract_candidate(it) for it in cached[:rows]]
        return {
            "query": query,
            "filter": year_filter,
            "candidates": candidates,
            "num_candidates": len(candidates),
            "cache_hit": cache_hit,
            "source": "crossref",
        }


# ---- Claude generation -------------------------------------------------------
def make_claude_client():
    import anthropic

    api_key = (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
    auth_token = (os.getenv("CLAUDE_AUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()
    base_url = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com")
    if api_key:
        return anthropic.Anthropic(api_key=api_key, base_url=base_url)
    if auth_token:
        return anthropic.Anthropic(auth_token=auth_token, base_url=base_url)
    raise RuntimeError(
        "Claude API key not set. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY (or auth token)."
    )


def call_claude(client, prompt: str, max_retries: int = 6) -> str:
    backoff = 1.0
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for block in getattr(resp, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    parts.append(getattr(block, "text", ""))
            return "".join(parts).strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[WARN] Claude call failed: {e!r}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(20.0, backoff * 2)
    raise RuntimeError(f"Claude call failed after retries: {last_err}")


# ---- Orchestration -----------------------------------------------------------
def load_existing_runs(path: str) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("run_id"):
                    existing[o["run_id"]] = o
    return existing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailto", default=DEFAULT_MAILTO, help="Crossref polite mailto")
    ap.add_argument("--max-runs", type=int, default=None, help="cap number of runs (debug)")
    ap.add_argument(
        "--skip-generation",
        action="store_true",
        help="only run + cache Crossref retrieval, do not call the model",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip runs whose run_id already has a non-empty output",
    )
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RAG_CACHE_DIR, exist_ok=True)

    seed_anchors = load_seed_anchors()
    topics = load_closed_book_topics()
    topic_ids = sorted(topics.keys())
    if len(topic_ids) != 144:
        print(f"[ERROR] expected 144 closed-book topics, found {len(topic_ids)}", file=sys.stderr)
        return 1
    missing_anchor = [t for t in topic_ids if t not in seed_anchors]
    if missing_anchor:
        print(f"[ERROR] topics missing from claims.csv: {missing_anchor}", file=sys.stderr)
        return 1

    retriever = CrossrefRagRetriever(mailto=args.mailto)
    client = None if args.skip_generation else make_claude_client()

    existing = load_existing_runs(OUT_PATH) if args.resume else {}

    # Build the work list: (condition, topic_id) in stable order.
    work = []
    for cond in CONDITIONS:
        for tid in topic_ids:
            work.append((cond, tid))
    if args.max_runs is not None:
        work = work[: args.max_runs]

    started_at = utc_now_iso()
    config = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "conditions": CONDITIONS,
        "top_k": TOP_K,
        "n_papers": FIXED_CITES,
        "retriever": "crossref",
        "crossref_base": CROSSREF_BASE,
        "mailto": args.mailto,
        "n_topics": len(topic_ids),
        "n_runs_planned": len(work),
        "started_at": started_at,
        "closed_book_source": os.path.relpath(CLOSED_BOOK_METRICS, ROOT_DIR),
        "claims_source": os.path.relpath(CLAIMS_PATH, ROOT_DIR),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    out_fp = open(OUT_PATH, "a", encoding="utf-8")
    generated = 0
    skipped = 0
    try:
        for cond, tid in tqdm(work, desc="RAG runs", unit="run"):
            run_id = f"rag-{cond}-{tid}"
            if args.resume and existing.get(run_id, {}).get("output"):
                skipped += 1
                continue

            entry = topics[tid]
            question = entry["question"]
            tw = entry.get("time_window") if cond == "temporal" else None
            start_year = tw["start_year"] if tw else None
            end_year = tw["end_year"] if tw else None

            query = build_query(seed_anchors[tid], question)
            ret = retriever.retrieve(
                query, rows=TOP_K, start_year=start_year, end_year=end_year
            )
            ret["retrieved_at"] = utc_now_iso()

            prompt = render_prompt(
                cond,
                question=question,
                n_papers=FIXED_CITES[cond],
                start_year=start_year,
                end_year=end_year,
                seed_anchors=seed_anchors[tid] or None,
                retrieved_candidates=ret["candidates"],
            )

            if args.skip_generation:
                continue

            output = call_claude(client, prompt)
            run = {
                "run_id": run_id,
                "timestamp": utc_now_iso(),
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "condition": cond,
                "topic_id": tid,
                "question": question,
                "prompt": prompt,
                "output": output,
                "retrieval": ret,
            }
            if tw:
                run["time_window"] = tw
            out_fp.write(json.dumps(run, ensure_ascii=False) + "\n")
            out_fp.flush()
            generated += 1
    finally:
        out_fp.close()

    print(
        f"Done. condition×topic={len(work)} generated={generated} skipped={skipped} "
        f"skip_generation={args.skip_generation}"
    )
    print(f"Runs -> {os.path.relpath(OUT_PATH, ROOT_DIR)}")
    print(f"Config -> {os.path.relpath(CONFIG_PATH, ROOT_DIR)}")
    print(f"Retrieval cache -> {os.path.relpath(os.path.join(RAG_CACHE_DIR, 'queries.jsonl'), ROOT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
