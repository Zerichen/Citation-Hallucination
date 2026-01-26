import os
import json
import uuid
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from tqdm import tqdm
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------

OUT_PATH = "out/gpt-4o_runs_result.jsonl"
TOPICS_PATH = "data/gpt-4o.jsonl"
LOG_PATH = os.getenv("RUNS_LOG_PATH", "out/generate_runs.log")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # 你也可以改成 gpt-5
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

MAX_TOKENS = 2048

# Initialize OpenAI-compatible clients
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

CLAUDE_API_URL = os.getenv("CLAUDE_API_URL", "https://api.anthropic.com/v1/messages")
CLAUDE_API_VERSION = os.getenv("CLAUDE_API_VERSION", "2023-06-01")

@dataclass
class _ClaudeMessage:
    content: str

@dataclass
class _ClaudeChoice:
    message: _ClaudeMessage

@dataclass
class _ClaudeResponse:
    choices: List[_ClaudeChoice]

class _ClaudeChatCompletions:
    def __init__(self, api_url: str, api_version: str, api_key: str) -> None:
        self._api_url = api_url
        self._api_version = api_version
        self._api_key = api_key

    def create(self, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> _ClaudeResponse:
        if not messages:
            raise ValueError("Claude messages cannot be empty.")
        system_text = ""
        user_text = ""
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_text = msg.get("content", "")
            elif role == "user":
                user_text = msg.get("content", "")

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_text,
            "messages": [{"role": "user", "content": user_text}],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._api_url,
            data=body,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self._api_version,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude API HTTP {e.code}: {err_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Claude API request failed: {e}") from e

        content = data.get("content", [])
        if isinstance(content, list):
            text_parts = [blk.get("text", "") for blk in content if blk.get("type") == "text"]
            text = "".join(text_parts).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            text = ""

        return _ClaudeResponse(choices=[_ClaudeChoice(message=_ClaudeMessage(content=text))])

class _ClaudeClient:
    def __init__(self, api_url: str, api_version: str, api_key: str) -> None:
        self.chat = type("Chat", (), {"completions": _ClaudeChatCompletions(api_url, api_version, api_key)})()

claude_client = _ClaudeClient(
    api_url=CLAUDE_API_URL,
    api_version=CLAUDE_API_VERSION,
    api_key=os.getenv("CLAUDE_API_KEY", ""),
)

qwen_client = OpenAI(
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
        try:
            if model.startswith(("gpt-", "o1", "o3", "o4")):
                client = openai_client
            elif model.startswith("claude-"):
                client = claude_client
            elif "llama" in model:
                client = llama_client
            else:
                client = qwen_client

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
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Starting runs: topics=%s output=%s", TOPICS_PATH, OUT_PATH)

    # read from TOPICS_PATH
    results = []
    with open(TOPICS_PATH, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    progress = tqdm(lines, desc="Generating runs", unit="run")
    for line in progress:
            line = line.strip()  # remove trailing newline
            if not line:
                continue  # skip empty lines
            single_run = json.loads(line)  # parse JSON
            model = single_run["model"]
            temp = single_run["temperature"]
            prompt = single_run["prompt"]
            progress.set_postfix(
                topic_id=single_run.get("topic_id"),
                condition=single_run.get("condition"),
                model=model,
            )
            logging.info(
                "Running topic_id=%s condition=%s model=%s temp=%s",
                single_run.get("topic_id"),
                single_run.get("condition"),
                model,
                temp,
            )
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
            logging.info(
                "Wrote run_id=%s topic_id=%s output_chars=%d",
                run["run_id"],
                run["topic_id"],
                len(output or ""),
            )

    validate_jsonl(OUT_PATH)
    logging.info("Done. JSONL validated OK.")

if __name__ == "__main__":
    main()
