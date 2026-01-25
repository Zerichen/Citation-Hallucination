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

OUT_PATH = "out/qwen2_5_14b_runs_result.jsonl"
TOPICS_PATH = "data/qwen2_5_14b_runs.jsonl"

MODEL = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-14B-Instruct")  # 你也可以改成 gpt-5
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

MAX_TOKENS = 2048

# Initialize OpenAI client (uses OPENAI_API_KEY)
other_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY", ""), base_url="https://api.siliconflow.cn/v1",
)

llama_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY", ""), base_url="https://api.together.xyz/v1",
)

# ----------------------------
# Prompt templates
# ----------------------------

# 为了可解析，强制 Key:Value 引用格式
CITATION_FORMAT = """Output citations in the following STRICT format:

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
- Use real academic papers (journal or top-tier conference).
"""

BASELINE_TMPL = """You are an expert researcher.

Write ONE concise academic paragraph answering the question below.

Question:
{question}

Requirements:
- Provide exactly {n_cites} citations.
- Keep the paragraph under {max_words} words.

{citation_format}
"""

SURVEY_TMPL = """You are writing the related work section of an academic survey.

Summarize the main research approaches for the topic below, organized into 3–4 approach categories.

Topic:
{question}

Requirements:
- Provide exactly {n_cites} citations.
- Keep the related-work text under {max_words} words.

{citation_format}
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
# Topics (replace with file loader if you want)
# ----------------------------

TOPICS: List[Dict[str, str]] = [
    {"topic_id": "t001", "question": "What are the main approaches to continual learning in deep neural networks?"},
    {"topic_id": "t002", "question": "What are the dominant paradigms in federated learning for healthcare?"},
    {"topic_id": "t003", "question": "What methods are used to detect and mitigate citation hallucination in LLM outputs?"},
]

# ----------------------------
# Helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def call_openai(prompt: str, model: str, temperature: float, max_retries: int = 6) -> str:
    backoff = 1.0
    last_err: Optional[Exception] = None

    for _ in range(max_retries):
        if "llama" in model:
            client = llama_client
        else:
            client = other_client

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. "
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            # 打印一次错误，避免“无输出卡住”
            print(f"[WARN] OpenAI call failed: {repr(e)}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff = min(20.0, backoff * 2)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")

def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

def validate_jsonl(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Invalid JSON at line {i}: {repr(e)}")

# ----------------------------
# Main
# ----------------------------

def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # read from TOPICS_PATH
    results = []
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # remove trailing newline
            if not line:
                continue  # skip empty lines
            single_run = json.loads(line)  # parse JSON
            model = single_run["model"]
            temp = single_run["temperature"]
            prompt = single_run["prompt"]
            output = call_openai(prompt, model, temp)
            run = {
                "run_id": str(uuid.uuid4()),
                "timestamp": utc_now_iso(),
                "model": model,
                "temperature": temp,
                "condition": single_run["condition"],
                "topic_id": single_run["topic_id"],
                "question": single_run["question"],
                "prompt": prompt,
                "output": output,
            }
            results.append(run)

            with open(OUT_PATH, "a", encoding="utf-8") as f:
                write_jsonl_line(f, run)

    validate_jsonl(OUT_PATH)
    print("Done. JSONL validated OK.")

if __name__ == "__main__":
    main()
