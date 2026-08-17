"""
Garmin Connect client for weight data, with local JSON caching.

Weight route is: Garmin Index scale -> Garmin Connect -> (Google Health Connect,
on-device only). Garmin Connect is the only cloud source for weigh-ins after the
Fitbit Web API stopped receiving them on 2026-07-22.

Auth: run `garmin_login.py` once in a terminal to create the token store
(.garmin_tokens). After that this module logs in silently from tokens.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

TOKENSTORE = os.path.join(os.path.dirname(__file__), ".garmin_tokens")
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".garmin_cache")
WEIGHT_CACHE_PATH = os.path.join(CACHE_DIR, "weight.json")

# Earliest date worth asking Garmin about (predates the scale; harmless).
EARLIEST_DATE = "2021-09-01"
# Re-fetch this many days before the last cached entry so late-synced
# weigh-ins are picked up.
RESYNC_OVERLAP_DAYS = 7
# Garmin range endpoints misbehave on very long ranges; chunk requests.
MAX_WINDOW_DAYS = 365
# Skip API calls if the cache was written recently (matches fitbit_client).
CACHE_MAX_AGE_SECONDS = 86400

_api = None


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def is_configured() -> bool:
    """True once the user has run garmin_login.py successfully."""
    return os.path.isdir(TOKENSTORE) and bool(os.listdir(TOKENSTORE))


def is_cache_fresh() -> bool:
    """True if the weight cache was written within the last 24 hours."""
    if not os.path.exists(WEIGHT_CACHE_PATH):
        return False
    age = datetime.now().timestamp() - os.path.getmtime(WEIGHT_CACHE_PATH)
    return age < CACHE_MAX_AGE_SECONDS


def _get_api():
    """Return a logged-in Garmin API object (cached per process)."""
    global _api
    if _api is not None:
        return _api
    from garminconnect import Garmin

    if not is_configured():
        raise RuntimeError(
            "Garmin tokens not found. Run `.venv/bin/python garmin_login.py` "
            "in a terminal once to log in."
        )
    api = Garmin()
    api.login(tokenstore=TOKENSTORE)
    # Persist refreshed tokens so the next process starts from fresh ones.
    try:
        api.client.dump(TOKENSTORE)
    except Exception:
        pass
    _api = api
    return _api


def _load_cache() -> List[Dict[str, Any]]:
    if not os.path.exists(WEIGHT_CACHE_PATH):
        return []
    with open(WEIGHT_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(records: List[Dict[str, Any]]) -> None:
    _ensure_cache_dir()
    with open(WEIGHT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f)


def _grams_to_kg(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    # Garmin reports weight in grams; tolerate already-kg values defensively.
    return round(value / 1000.0, 2) if value > 500 else round(float(value), 2)


def _parse_weigh_ins(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a get_weigh_ins() response to [{date, weight, bmi, fat}]."""
    records: List[Dict[str, Any]] = []

    summaries = payload.get("dailyWeightSummaries") or []
    for day in summaries:
        day_date = day.get("summaryDate")
        metrics = day.get("allWeightMetrics") or []
        best = day.get("latestWeight") or (metrics[0] if metrics else None)
        if not day_date or not best:
            continue
        records.append(
            {
                "date": day_date,
                "weight": _grams_to_kg(best.get("weight")),
                "bmi": best.get("bmi"),
                "fat": best.get("bodyFat"),
                "source": best.get("sourceType") or "GARMIN",
            }
        )

    # Older/alternate response shape.
    for entry in payload.get("dateWeightList") or []:
        day_date = entry.get("calendarDate") or entry.get("date")
        if not day_date or entry.get("weight") is None:
            continue
        records.append(
            {
                "date": str(day_date)[:10],
                "weight": _grams_to_kg(entry.get("weight")),
                "bmi": entry.get("bmi"),
                "fat": entry.get("bodyFat"),
                "source": entry.get("sourceType") or "GARMIN",
            }
        )

    return [r for r in records if r["weight"]]


def fetch_weight(force_full: bool = False, progress_cb=None) -> pd.DataFrame:
    """
    Fetch Garmin weigh-ins. Incremental: only asks Garmin for dates after the
    last cached entry (minus a small overlap). force_full re-pulls everything.
    """
    cached = [] if force_full else _load_cache()
    if cached and not force_full and is_cache_fresh():
        return _records_to_df(cached)
    if cached:
        last = max(r["date"] for r in cached)
        fetch_start = date.fromisoformat(last) - timedelta(days=RESYNC_OVERLAP_DAYS)
    else:
        fetch_start = date.fromisoformat(EARLIEST_DATE)

    today = date.today()
    if fetch_start > today:
        return _records_to_df(cached)

    api = _get_api()
    new_records: List[Dict[str, Any]] = []
    window_start = fetch_start
    total_days = (today - fetch_start).days + 1
    while window_start <= today:
        if progress_cb:
            done = (window_start - fetch_start).days
            progress_cb(min(done / max(total_days, 1), 1.0))
        window_end = min(window_start + timedelta(days=MAX_WINDOW_DAYS), today)
        payload = api.get_weigh_ins(window_start.isoformat(), window_end.isoformat())
        new_records.extend(_parse_weigh_ins(payload or {}))
        window_start = window_end + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(all_records)
    if progress_cb:
        progress_cb(1.0)
    return _records_to_df(all_records)


def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "Weight", "BMI", "Fat", "Source"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(
        columns={"weight": "Weight", "bmi": "BMI", "fat": "Fat", "source": "Source"}
    ).drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def load_cached_dataframe() -> pd.DataFrame:
    """Cached Garmin weight as a DataFrame without hitting the API."""
    return _records_to_df(_load_cache())


def load_merged_weight(fetch: bool = False) -> pd.DataFrame:
    """
    Single weight history from both sources:
    - Garmin Connect (scale weigh-ins, the live source going forward)
    - the archived Fitbit history (includes manually logged Fitbit entries
      that never existed in Garmin)
    On dates present in both, Garmin wins. Adds a Source column.
    """
    import fitbit_client

    garmin_df = fetch_weight() if fetch else load_cached_dataframe()
    fitbit_df = fitbit_client.load_cached_dataframe("weight")

    frames = []
    if fitbit_df is not None and not fitbit_df.empty:
        f = fitbit_df.copy()
        f["Source"] = "FITBIT"
        frames.append(f[["Date", "Weight", "BMI", "Fat", "Source"]])
    if not garmin_df.empty:
        g = garmin_df.copy()
        g["Source"] = g.get("Source", "GARMIN").fillna("GARMIN")
        frames.append(g[["Date", "Weight", "BMI", "Fat", "Source"]])

    if not frames:
        return pd.DataFrame(columns=["Date", "Weight", "BMI", "Fat", "Source"])

    merged = pd.concat(frames, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"]).dt.normalize()
    # keep="last" prefers Garmin because the Garmin frame is appended last
    merged = (
        merged.sort_values(["Date"])
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    return merged
