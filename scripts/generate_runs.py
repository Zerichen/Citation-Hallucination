import os
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Any

from tqdm import tqdm
from openai import OpenAI
# import anthropic

# ----------------------------
# Config
# ----------------------------

OUT_PATH = "out/llama3_8b_full_runs_combo_only.jsonl"
TOPICS_PATH = "data/llama3_8b_full_runs.jsonl"

GPT4O_OUT_PATH = "out/gpt-4o_full_runs.jsonl"
GPT4O_TOPICS_PATH = "data/gpt-4o_full_runs.jsonl"

MAX_TOKENS = 2048

# Initialize OpenAI-compatible clients
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "https://api.anthropic.com")
# _claude_api_key = (os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or "").strip()
# _claude_auth_token = (os.getenv("CLAUDE_AUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()

# if _claude_api_key:
#     claude_client = anthropic.Anthropic(
#         api_key=_claude_api_key,
#         base_url=CLAUDE_BASE_URL,
#     )
# elif _claude_auth_token:
#     claude_client = anthropic.Anthropic(
#         auth_token=_claude_auth_token,
#         base_url=CLAUDE_BASE_URL,
#     )
# else:
#     claude_client = None

qwen_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY", ""), base_url="https://api.siliconflow.cn/v1",
)

llama_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY", ""), base_url="https://api.together.xyz/v1",
)

# ----------------------------
# Helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _client_for_model(model: str):
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return openai_client
    if model.startswith("claude-"):
        return claude_client
    if "llama" in model:
        return llama_client
    return qwen_client

def call_openai(prompt: str, model: str, temperature: float, max_retries: int = 6) -> str:
    backoff = 1.0
    last_err: Optional[Exception] = None

    for _ in range(max_retries):
        try:
            if model.startswith("claude-"):
                if claude_client is None:
                    raise RuntimeError(
                        "Claude API key not set. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY (or auth token)."
                    )
                resp = claude_client.messages.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                    system="You are a research assistant.",
                    messages=[{"role": "user", "content": prompt}],
                )
                text_parts = []
                for block in getattr(resp, "content", []) or []:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(getattr(block, "text", ""))
                return "".join(text_parts).strip()

            client = _client_for_model(model)
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
            print(f"[WARN] Model call failed ({model}): {repr(e)}. Retrying in {backoff:.1f}s...")
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

def _process_file(topics_path: str, out_path: str) -> None:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    resume = os.getenv("RESUME", "").strip().lower() in {"1", "true", "yes"}
    done_count = 0
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            done_count = sum(1 for _ in f)

    with open(topics_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if resume and done_count:
        print(f"[INFO] RESUME enabled: skipping first {done_count} runs already in {out_path}")
        if done_count >= len(lines):
            print("[INFO] All runs already completed. Nothing to do.")
            return

    progress = tqdm(lines[done_count:], desc=f"Generating runs ({os.path.basename(out_path)})", unit="run")
    for line in progress:
        line = line.strip()  # remove trailing newline
        if not line:
            continue  # skip empty lines
        single_run = json.loads(line)  # parse JSON
        if single_run["condition"] != 'combo':
            continue
        model = single_run["model"]
        temp = single_run["temperature"]
        prompt = single_run["prompt"]
        progress.set_postfix(
            topic_id=single_run.get("topic_id"),
            condition=single_run.get("condition"),
            model=model,
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
        with open(out_path, "a", encoding="utf-8") as f:
            write_jsonl_line(f, run)
    validate_jsonl(out_path)
    print(f"Done. JSONL validated OK: {out_path}")


def main():
    _process_file(TOPICS_PATH, OUT_PATH)
    # _process_file(GPT4O_TOPICS_PATH, GPT4O_OUT_PATH)

if __name__ == "__main__":
    main()
