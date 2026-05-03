import json
import os
import time
from typing import Any, Dict, List, Optional
import requests

class JsonlCache:
    def __init__(self, path: str):
        self.path = path
        self._mem: Dict[str, Any] = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self._mem[obj["key"]] = obj["value"]
                    except Exception:
                        continue

    def get(self, key: str):
        return self._mem.get(key)

    def set(self, key: str, value: Any):
        if key in self._mem:
            return
        self._mem[key] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

class CrossrefClient:
    BASE = "https://api.crossref.org"

    def __init__(self, mailto: Optional[str], cache_dir: str = "cache", sleep_s: float = 0.2):
        self.mailto = mailto
        self.sleep_s = sleep_s
        self.cache_doi = JsonlCache(os.path.join(cache_dir, "crossref_doi.jsonl"))
        self.cache_title = JsonlCache(os.path.join(cache_dir, "crossref_title.jsonl"))

    def lookup_doi(self, doi: str) -> Optional[Dict]:
        key = f"doi::{doi}"
        cached = self.cache_doi.get(key)
        if cached is not None:
            return cached
        params = {}
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{self.BASE}/works/{requests.utils.quote(doi)}"
        r = requests.get(url, params=params, timeout=20)
        time.sleep(self.sleep_s)
        if r.status_code != 200:
            self.cache_doi.set(key, None)
            return None
        msg = r.json().get("message")
        self.cache_doi.set(key, msg)
        return msg

    def search_title(self, title: str, rows: int = 5) -> List[Dict]:
        key = f"title::{title}::rows={rows}"
        cached = self.cache_title.get(key)
        if cached is not None:
            return cached
        params = {"query.title": title, "rows": rows}
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"{self.BASE}/works"
        r = requests.get(url, params=params, timeout=20)
        time.sleep(self.sleep_s)
        if r.status_code != 200:
            self.cache_title.set(key, [])
            return []
        items = r.json().get("message", {}).get("items", [])
        self.cache_title.set(key, items)
        return items

class SemanticScholarClient:
    BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache", sleep_s: float = 0.2):
        self.api_key = api_key
        self.sleep_s = sleep_s
        self.cache_title = JsonlCache(os.path.join(cache_dir, "s2_title.jsonl"))

    def search_title(self, title: str, limit: int = 5) -> List[Dict]:
        key = f"title::{title}::limit={limit}"
        cached = self.cache_title.get(key)
        if cached is not None:
            return cached
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        params = {
            "query": title,
            "limit": limit,
            "fields": "title,authors,year,venue,externalIds,publicationTypes"
        }
        url = f"{self.BASE}/paper/search"
        r = requests.get(url, params=params, headers=headers, timeout=20)
        time.sleep(self.sleep_s)
        if r.status_code != 200:
            self.cache_title.set(key, [])
            return []
        data = r.json().get("data", [])
        self.cache_title.set(key, data)
        return data
