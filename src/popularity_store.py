"""Popularity store: likes/downloads separated from reasoning metrics.

Why this file exists:
- Likes/Downloads are *volatile* HF API metadata (change hourly).
- KL Divergence / Refusals are *expensive* parsed reasoning metrics
  (README + reproduce.json fetch + regex, 20 workers).
- Mixing them in one cache key (`cached_get_model_data(model_id, likes, downloads)`)
  churns the expensive cache every time someone likes a model.

Well-grounded approach: yes — keep them separate.
- Reasoning cache: keyed by model_id only, long TTL (1h+).
- Popularity: short TTL snapshot + refreshable JSON file
  (`data/popularity.json`) that can be rewritten with minimal impact
  (no re-parse of READMEs, just a cheap HfApi list).

File format (data/popularity.json):
{
  "updated_at": "2026-09-14T12:00:00+00:00",
  "models": {
    "<model_id>": {"likes": 12, "downloads": 345, "url": "https://..."},
    ...
  }
}
Missing file -> treated as empty map (graceful, Docker-safe).
"""

import json
import os
from datetime import datetime, timezone

# Resolve repo-root/data/popularity.json from either:
#   /app/src/popularity_store.py -> /app/data/popularity.json (Docker)
#   <root>/src/popularity_store.py -> <root>/data/popularity.json (local)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(os.path.dirname(_THIS_DIR), "data", "popularity.json"),
    os.path.join(os.getcwd(), "data", "popularity.json"),
    os.path.join("/app", "data", "popularity.json"),
]


def popularity_path() -> str:
    for p in _CANDIDATES:
        d = os.path.dirname(p)
        if os.path.isdir(d):
            return p
    return _CANDIDATES[0]


def load_popularity() -> dict:
    """Return {model_id: {likes, downloads, url}}. Empty dict if missing/corrupt."""
    path = popularity_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", {}) if isinstance(data, dict) else {}
        return models if isinstance(models, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_popularity(models_map: dict) -> str:
    """Persist {model_id: {likes, downloads}} snapshot. Returns path written."""
    path = popularity_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": models_map,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def snapshot_from_metadata(metadata: list) -> dict:
    """Build file-ready map from cheap HfApi metadata list."""
    out = {}
    for d in metadata or []:
        mid = d.get("Model ID")
        if not mid:
            continue
        try:
            likes = int(d.get("Likes", 0) or 0)
        except (TypeError, ValueError):
            likes = 0
        try:
            downloads = int(d.get("Downloads", 0) or 0)
        except (TypeError, ValueError):
            downloads = 0
        out[mid] = {"likes": likes, "downloads": downloads, "url": d.get("URL")}
    return out


def merge_popularity(rows: list, metadata: list, file_map: dict | None = None) -> list:
    """Enrich parsed reasoning rows with likes/downloads.

    Precedence per model: file snapshot > live metadata > row's own values.
    This lets you refresh popularity (rewrite one JSON) without touching
    the expensive KL/refusal cache.
    """
    if file_map is None:
        file_map = load_popularity()
    meta_map = {d.get("Model ID"): d for d in (metadata or [])}
    for r in rows or []:
        mid = r.get("Model ID")
        file_hit = (file_map or {}).get(mid, {}) if mid else {}
        meta_hit = meta_map.get(mid, {}) if mid else {}
        if isinstance(file_hit, dict) and ("likes" in file_hit or "downloads" in file_hit):
            try:
                r["Likes"] = int(file_hit.get("likes", r.get("Likes", 0)) or 0)
            except (TypeError, ValueError):
                pass
            try:
                r["Downloads"] = int(file_hit.get("downloads", r.get("Downloads", 0)) or 0)
            except (TypeError, ValueError):
                pass
        else:
            if mid in meta_map:
                try:
                    r["Likes"] = int(meta_hit.get("Likes", r.get("Likes", 0)) or 0)
                except (TypeError, ValueError):
                    pass
                try:
                    r["Downloads"] = int(meta_hit.get("Downloads", r.get("Downloads", 0)) or 0)
                except (TypeError, ValueError):
                    pass
    return rows
