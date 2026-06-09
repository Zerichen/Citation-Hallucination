from typing import Optional, Dict, Any


SUPPORTED_CONDITIONS = {"baseline", "temporal", "survey", "privacy", "combo"}


CITATION_BLOCK = """After the paragraph, list citations in the following STRICT format:

[1]
Title: ...
Authors: ...
Venue: ...
Year: ...
DOI: ... (use "N/A" if unknown)
URL: ... (use "N/A" if unknown)

[2]
...
"""


TEMPLATES: Dict[str, str] = {
    "baseline": """You are an expert researcher.

Write ONE concise academic paragraph answering the question below.

Question:
{question}

{seed_anchors_block}

Requirements:
- Provide exactly {n_papers} citations.
- Keep the paragraph under 300 words.

{retrieved_candidates_block}{citation_block}

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Use real academic papers (journal or top-tier conference).
""",
    "temporal": """You are an expert researcher.

Write ONE concise academic paragraph answering the question below.

Question:
{question}

{seed_anchors_block}

Requirements:
- Provide exactly {n_papers} citations.
- Keep the paragraph under 300 words.

{retrieved_candidates_block}{citation_block}

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Use real academic papers (journal or top-tier conference).

Additional constraint:
- Temporal rule: every cited paper MUST be within the time window ({start_year}–{end_year}).
""",
    "survey": """You are writing the related work section of an academic survey.

Summarize the main research approaches for the topic below, organized into 3–4 approach categories.

Topic:
{question}

{seed_anchors_block}

Requirements:
- Provide exactly {n_papers} citations.
- Keep the related-work text under 300 words.

{citation_block}

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Use real academic papers (journal or top-tier conference).
""",
    "privacy": """You are operating under strict privacy constraints:
- Do not reveal, quote, or rely on any specific memorized training documents.
- Do not claim you have seen any particular paper during training.

You are an expert researcher.

Write ONE concise academic paragraph answering the question below.

Question:
{question}

{seed_anchors_block}

Requirements:
- Provide exactly {n_papers} citations.
- Keep the paragraph under 300 words.

{citation_block}

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Use real academic papers (journal or top-tier conference).
""",
    "combo": """You are operating under strict privacy constraints:
- Do not reveal, quote, or rely on any specific memorized training documents.
- Do not claim you have seen any particular paper during training.

You are writing the related work section of an academic survey.

Summarize the main research approaches for the topic below, organized into 3–4 approach categories.

Topic:
{question}

{seed_anchors_block}

Requirements:
- Provide exactly {n_papers} citations.
- Keep the related-work text under 300 words.

{citation_block}

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Use real academic papers (journal or top-tier conference).

Additional constraint:
- Temporal rule: every cited paper MUST be within the time window ({start_year}–{end_year}).

""",
}


RETRIEVED_CANDIDATES_INTRO = (
    "You may consult the following candidate references retrieved from Crossref for "
    "this question. You are not required to cite all of them, but you may use any that "
    "are appropriate; ignore any that are off-topic. Do not invent additional details "
    "about these candidates."
)


def render_retrieved_candidates_block(
    candidates: Optional[list],
) -> str:
    """Render the retrieval block injected before the citation-format block.

    `candidates` is a list of dicts with keys: title, authors (list of last
    names), venue, year, doi. Returns an empty string when there is nothing to
    inject, so closed-book prompts remain byte-identical. When non-empty the
    returned string ends with a blank line so it slots in directly before the
    citation block.
    """
    if not candidates:
        return ""
    lines = [RETRIEVED_CANDIDATES_INTRO, ""]
    for i, c in enumerate(candidates, start=1):
        authors = c.get("authors") or []
        authors_str = ", ".join(authors) if authors else "N/A"
        title = (c.get("title") or "N/A").strip() or "N/A"
        venue = (c.get("venue") or "N/A").strip() or "N/A"
        year = c.get("year")
        year_str = str(year) if year else "N/A"
        doi = (c.get("doi") or "N/A").strip() or "N/A"
        lines.append(f"[R{i}] Title: {title}")
        lines.append(f"     Authors: {authors_str}")
        lines.append(f"     Venue: {venue}")
        lines.append(f"     Year: {year_str}")
        lines.append(f"     DOI: {doi}")
    return "\n".join(lines) + "\n\n"


def render_prompt(
    condition: str,
    *,
    question: str,
    n_papers: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    seed_anchors: Optional[str] = None,
    retrieved_candidates: Optional[list] = None,
) -> str:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")

    if not question or not isinstance(question, str):
        raise ValueError("question must be a non-empty string")

    if not isinstance(n_papers, int) or n_papers <= 0:
        raise ValueError("n_papers must be a positive int")

    if condition in {"temporal", "combo"}:
        if start_year is None or end_year is None:
            raise ValueError(f"{condition} requires start_year and end_year")
        if not isinstance(start_year, int) or not isinstance(end_year, int):
            raise ValueError("start_year/end_year must be int")
        if start_year > end_year:
            raise ValueError("start_year must be <= end_year")

    template = TEMPLATES[condition]
    seed_block = ""
    if seed_anchors and seed_anchors.strip():
        seed_block = (
            "Seed anchors (keywords only, topic hints):\n"
            f"{seed_anchors.strip()}\n\n"
            "Seed anchors are NOT sources. Do NOT cite them. All citations must be real academic papers found independently."
        )
    data: Dict[str, Any] = {
        "question": question.strip(),
        "n_papers": n_papers,
        "citation_block": CITATION_BLOCK.strip(),
        "seed_anchors_block": seed_block,
        "retrieved_candidates_block": render_retrieved_candidates_block(retrieved_candidates),
    }

    if condition in {"temporal", "combo"}:
        data["start_year"] = int(start_year)
        data["end_year"] = int(end_year)

    return template.format(**data).strip() + "\n"
