"""citecheck — defensive verification of LLM-generated bibliographic citations."""

__version__ = "0.1.0"

from .parser import parse_citations, build_citation_objects
from .normalize import normalize_citation
from .match import best_match
from .label import assign_label
from .verify import verify_citation, verify_file

__all__ = [
    "__version__",
    "parse_citations",
    "build_citation_objects",
    "normalize_citation",
    "best_match",
    "assign_label",
    "verify_citation",
    "verify_file",
]
