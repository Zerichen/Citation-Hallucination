"""Smoke tests: package imports, label thresholds, end-to-end verify path.

External HTTP is monkey-patched in the verify test so the suite runs offline.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


def test_package_imports_and_version():
    import citecheck

    assert isinstance(citecheck.__version__, str)
    assert citecheck.__version__  # non-empty
    # Public surface re-exported from __init__
    assert callable(citecheck.parse_citations)
    assert callable(citecheck.build_citation_objects)
    assert callable(citecheck.normalize_citation)
    assert callable(citecheck.best_match)
    assert callable(citecheck.assign_label)
    assert callable(citecheck.verify_citation)
    assert callable(citecheck.verify_file)


def test_assign_label_thresholds():
    from citecheck.label import EXISTS_TH, AMBIG_TH, assign_label
    from citecheck.schema import Citation, MatchResult

    c = Citation(citation_id="t_c1", run_id="t", index=1, raw="x", title="x")

    high = MatchResult(source="crossref", score=EXISTS_TH + 0.05, record={})
    mid = MatchResult(source="crossref", score=AMBIG_TH + 0.05, record={})
    low = MatchResult(source="crossref", score=AMBIG_TH - 0.10, record={})

    assert assign_label(c, high, []).label == "EXISTS"
    assert assign_label(c, mid, []).label == "AMBIGUOUS"
    assert assign_label(c, low, []).label == "FABRICATED"
    # No candidates at all → FABRICATED with confidence 0
    none_result = assign_label(c, None, [])
    assert none_result.label == "FABRICATED"
    assert none_result.confidence == 0.0


def test_cli_version_runs():
    from citecheck.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_verify_file_with_mocked_clients(tmp_path, monkeypatch):
    """End-to-end: parse a real-looking citation, mock the API clients,
    and confirm the EXISTS label is assigned."""
    from citecheck import verify

    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    cache_dir = tmp_path / "cache"
    input_path.write_text(
        '{"title": "Attention Is All You Need", '
        '"authors": ["Ashish Vaswani", "Noam Shazeer"], '
        '"venue": "NeurIPS", "year": 2017, "doi": "10.5555/3295222.3295349"}\n',
        encoding="utf-8",
    )

    crossref_record = {
        "title": "Attention Is All You Need",
        "norm_title": "attention is all you need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "year": 2017,
        "venue": "NeurIPS",
        "norm_venue": "nips",
        "doi": "10.5555/3295222.3295349",
    }

    class StubCrossref:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def lookup_doi(self, doi: str) -> Dict[str, Any]:
            return {
                "title": [crossref_record["title"]],
                "author": [
                    {"given": "Ashish", "family": "Vaswani"},
                    {"given": "Noam", "family": "Shazeer"},
                ],
                "issued": {"date-parts": [[2017]]},
                "container-title": ["NeurIPS"],
                "DOI": crossref_record["doi"],
            }

        def search_title(self, title: str, rows: int = 5) -> List[Dict[str, Any]]:
            return []

    class StubS2:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def search_title(self, title: str, limit: int = 5) -> List[Dict[str, Any]]:
            return [
                {
                    "title": "Attention Is All You Need",
                    "authors": [
                        {"name": "Ashish Vaswani"},
                        {"name": "Noam Shazeer"},
                    ],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "externalIds": {"DOI": crossref_record["doi"]},
                }
            ]

    monkeypatch.setattr(verify, "CrossrefClient", StubCrossref)
    monkeypatch.setattr(verify, "SemanticScholarClient", StubS2)

    results = verify.verify_file(
        str(input_path),
        output_path=str(output_path),
        cache_dir=str(cache_dir),
    )
    assert len(results) == 1
    assert results[0]["label"] == "EXISTS"
    assert results[0]["confidence"] >= 0.85
    assert output_path.exists()


def test_parse_citations_block_format():
    from citecheck.parser import parse_citations, build_citation_objects

    text = (
        "Some paragraph...\n\n"
        "[1]\n"
        "Title: Example Paper\n"
        "Authors: Doe, J.; Smith, A.\n"
        "Venue: ACL\n"
        "Year: 2020\n"
        "DOI: 10.1234/example\n"
        "URL: N/A\n"
    )
    blocks = parse_citations("r1", text)
    assert len(blocks) == 1
    cits = build_citation_objects("r1", blocks)
    assert cits[0].title == "Example Paper"
    assert cits[0].year == 2020
    assert cits[0].doi == "10.1234/example"
