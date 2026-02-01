# src/prompts.py

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

{citation_block}

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

{citation_block}

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

""",
}


def render_prompt(
    condition: str,
    *,
    question: str,
    n_papers: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    seed_anchors: Optional[str] = None,
) -> str:
    if condition not in SUPPORTED_CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")

    if not question or not isinstance(question, str):
        raise ValueError("question must be a non-empty string")

    if not isinstance(n_papers, int) or n_papers <= 0:
        raise ValueError("n_papers must be a positive int")

    if condition == "temporal":
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
    }

    if condition == "temporal":
        data["start_year"] = int(start_year)
        data["end_year"] = int(end_year)

    return template.format(**data).strip() + "\n"
