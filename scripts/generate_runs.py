import os
import json
import uuid
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from tqdm import tqdm
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------

OUT_PATH = "data/runs.jsonl"

MODEL = os.getenv("OPENAI_MODEL", "gpt-5")  # or "gpt-4o"
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

# Responses API recommended by OpenAI docs
# https://platform.openai.com/docs/api-reference/responses
client = OpenAI()

# If you want temporal cutoff stressor:
TEMPORAL_WINDOW = {"start_year": 2023, "end_year": 2025}

# ----------------------------
# Prompt templates
# ----------------------------

BASELINE_TMPL = """You are an expert researcher.

Write ONE concise academic paragraph answering the question below.

Question:
{question}

Requirements:
- Provide exactly {n_cites} citations.
- Output citations in the following STRICT format so they can be verified:

[1]
Title: ...
Authors: ...
Venue: ...
Year: ...
DOI: ... (use "N/A" if unknown)
URL: ... (use "N/A" if unknown)

[2]
...

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Keep the paragraph under {max_words} words.
- Citations must be academic papers (journal or top-tier conference).
"""

SURVEY_TMPL = """You are writing the related work section of an academic survey.

Summarize the main research approaches for the topic below, organized into 3–4 approach categories.

Topic:
{question}

Requirements:
- Provide exactly {n_cites} citations.
- Output citations in the following STRICT format so they can be verified:

[1]
Title: ...
Authors: ...
Venue: ...
Year: ...
DOI: ... (use "N/A" if unknown)
URL: ... (use "N/A" if unknown)

[2]
...

Rules:
- Do not invent DOIs. If unknown, write "N/A".
- Keep the related-work text under {max_words} words.
- Citations must be academic papers (journal or top-tier conference).
"""

PRIVACY_PREAMBLE = """You are operating under strict privacy constraints:
- Do not reveal, quote, or rely on any specific memorized training documents.
- Do not claim you have seen any particular paper during training.
"""

TEMPORAL_APPENDIX = """Additional constraint:
- Focus specifically on research published between 2023 and 2025.
"""

# ----------------------------
# Conditions
# ----------------------------

@dataclass
class Condition:
    name: str
    template: str
    n_cites: int
    max_words: int
    add_privacy: bool = False
    add_temporal: bool = False

CONDITIONS: List[Condition] = [
    Condition(name="baseline", template=BASELINE_TMPL, n_cites=5, max_words=150),
    Condition(name="temporal", template=BASELINE_TMPL, n_cites=5, max_words=150, add_temporal=True),
    Condition(name="survey", template=SURVEY_TMPL, n_cites=8, max_words=220),
    Condition(name="privacy", template=BASELINE_TMPL, n_cites=5, max_words=150, add_privacy=True),
    Condition(name="combo", template=SURVEY_TMPL, n_cites=8, max_words=220, add_privacy=True, add_temporal=True),
]

# ----------------------------
# Topic list (replace with your own loader)
# ----------------------------

TOPICS: List[Dict[str, str]] = [
    {"topic_id": "t001", "question": "What are the main approaches to continual learning in deep neural networks?"},
    {"topic_id": "t002", "question": "What are the dominant paradigms in federated learning for healthcare?"},
    {"topic_id": "t003", "question": "What methods are used to detect and mitigate citation hallucination in LLM outputs?"},
]

# ----------------------------
# Helpers
# ----------------------------

def build_prompt(question: str, cond: Condition) -> str:
    prompt = cond.template.format(question=question, n_cites=cond.n_cites, max_words=cond.max_words)
    if cond.add_privacy:
        prompt = PRIVACY_PREAMBLE.strip() + "\n\n" + prompt
    if cond.add_temporal:
        prompt = prompt + "\n\n" + TEMPORAL_APPENDIX.strip()
    return prompt.strip()

def call_openai(prompt: str, model: str, temperature: float, max_retries: int = 5) -> str:
    """
    Calls OpenAI Responses API and returns plain text.
    Retries on transient errors.
    """
    backoff = 1.0
    last_err: Optional[Exception] = None

    for _ in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
            )
            # output_text is in docs examples
            return resp.output_text
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff = min(20.0, backoff * 2)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_jsonl_line(fp, obj: Dict[str, Any]):
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def validate_jsonl(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            json.loads(line)  # will raise if invalid

# ----------------------------
# Main
# ----------------------------
print("generate_runs.py started")
def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

if __name__ == "__main__":
    main()

