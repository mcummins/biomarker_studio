"""
Hevy API client with lightweight local JSON caching.

Stores fetched workouts locally so the dashboard can open quickly without
hammering the Hevy API on every page visit.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

HEVY_API_BASE = "https://api.hevyapp.com"
REQUEST_TIMEOUT = (10, 30)
MAX_PAGE_SIZE = 10
CACHE_TTL = timedelta(hours=6)
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hevy_cache")
CACHE_PATH = os.path.join(CACHE_DIR, "workouts.json")
KEY_FILENAMES = ("hevy_api_key.txt",)


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_cache_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _cache_payload() -> Optional[Dict[str, Any]]:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache_payload(workouts: List[Dict[str, Any]]) -> None:
    _ensure_cache_dir()
    payload = {
        "fetched_at": _utc_now().isoformat(),
        "workouts": workouts,
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def get_api_key_path() -> Optional[str]:
    base_dir = os.path.dirname(__file__)
    for filename in KEY_FILENAMES:
        candidate = os.path.join(base_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_api_key() -> Optional[str]:
    path = get_api_key_path()
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        value = f.read().strip()
    return value or None


def has_api_key() -> bool:
    return bool(load_api_key())


def cache_last_updated() -> Optional[datetime]:
    payload = _cache_payload()
    if not payload:
        return None
    return _parse_cache_timestamp(payload.get("fetched_at"))


def is_cache_fresh() -> bool:
    fetched_at = cache_last_updated()
    if not fetched_at:
        return False
    return (_utc_now() - fetched_at) <= CACHE_TTL


def clear_cache() -> None:
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)


def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    api_key = load_api_key()
    if not api_key:
        raise ValueError("No Hevy API key found. Add `hevy_api_key.txt` to the project root.")

    response = requests.get(
        f"{HEVY_API_BASE}{path}",
        params=params,
        headers={"api-key": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _fetch_all_workouts_from_api() -> List[Dict[str, Any]]:
    workouts: List[Dict[str, Any]] = []
    page = 1
    page_count = 1

    while page <= page_count:
        payload = _api_get("/v1/workouts", params={"page": page, "pageSize": MAX_PAGE_SIZE})
        page_count = int(payload.get("page_count") or 1)
        workouts.extend(payload.get("workouts", []))
        page += 1

    return workouts


def get_workouts(force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cached = _cache_payload()
    cached_workouts = cached.get("workouts", []) if cached else []
    cached_at = _parse_cache_timestamp(cached.get("fetched_at")) if cached else None

    if cached_workouts and not force_refresh and is_cache_fresh():
        return cached_workouts, {
            "source": "cache",
            "fetched_at": cached_at,
            "warning": None,
        }

    try:
        workouts = _fetch_all_workouts_from_api()
        _save_cache_payload(workouts)
        return workouts, {
            "source": "live",
            "fetched_at": cache_last_updated(),
            "warning": None,
        }
    except Exception as exc:
        if cached_workouts:
            return cached_workouts, {
                "source": "cache",
                "fetched_at": cached_at,
                "warning": f"Live Hevy sync failed, showing cached data instead ({exc}).",
            }
        raise


def build_working_set_history(workouts: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for workout in workouts:
        workout_id = workout.get("id")
        workout_title = workout.get("title") or "Untitled workout"
        start_time = workout.get("start_time") or workout.get("created_at") or workout.get("end_time")

        for exercise in workout.get("exercises", []) or []:
            exercise_title = exercise.get("title") or "Untitled exercise"
            exercise_key = exercise.get("exercise_template_id") or exercise_title

            for set_entry in exercise.get("sets", []) or []:
                set_type = (set_entry.get("type") or "normal").strip().lower()
                if set_type == "warmup":
                    continue

                weight_kg = _to_float(set_entry.get("weight_kg"))
                reps = _to_float(set_entry.get("reps"))
                if weight_kg is None or reps is None or weight_kg <= 0 or reps <= 0:
                    continue

                estimated_1rm_kg = weight_kg * (1 + reps / 30.0)
                rows.append(
                    {
                        "workout_id": workout_id,
                        "workout_title": workout_title,
                        "workout_start_time": start_time,
                        "exercise_key": exercise_key,
                        "exercise_template_id": exercise.get("exercise_template_id"),
                        "exercise_title": exercise_title,
                        "exercise_index": exercise.get("index"),
                        "set_index": set_entry.get("index"),
                        "set_type": set_type,
                        "weight_kg": weight_kg,
                        "reps": reps,
                        "estimated_1rm_kg": estimated_1rm_kg,
                    }
                )

    columns = [
        "workout_id",
        "workout_title",
        "workout_start_time",
        "exercise_key",
        "exercise_template_id",
        "exercise_title",
        "exercise_index",
        "set_index",
        "set_type",
        "weight_kg",
        "reps",
        "estimated_1rm_kg",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["workout_start_time"] = pd.to_datetime(df["workout_start_time"], utc=True, errors="coerce")
    df["workout_start_time"] = df["workout_start_time"].dt.tz_convert(None)
    df["workout_date"] = df["workout_start_time"].dt.normalize()
    return df.sort_values(["workout_start_time", "exercise_title", "set_index"]).reset_index(drop=True)


def summarize_session_best(history_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "exercise_key",
        "exercise_title",
        "workout_id",
        "workout_title",
        "workout_start_time",
        "workout_date",
        "estimated_1rm_kg",
        "best_weight_kg",
        "best_reps",
        "working_set_count",
    ]
    if history_df.empty:
        return pd.DataFrame(columns=columns)

    ordered = history_df.sort_values(["exercise_key", "workout_start_time", "estimated_1rm_kg"])
    best_rows = ordered.groupby(["exercise_key", "workout_id"], as_index=False).tail(1)
    working_set_counts = (
        history_df.groupby(["exercise_key", "workout_id"])
        .size()
        .rename("working_set_count")
        .reset_index()
    )

    session_best = best_rows.merge(
        working_set_counts,
        on=["exercise_key", "workout_id"],
        how="left",
    )
    session_best = session_best.rename(columns={"reps": "best_reps", "weight_kg": "best_weight_kg"})

    return session_best[columns].sort_values("workout_start_time").reset_index(drop=True)


def summarize_top_exercises(session_best_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    columns = [
        "exercise_key",
        "exercise_title",
        "session_count",
        "total_working_sets",
        "best_1rm_kg",
        "latest_1rm_kg",
        "first_logged",
        "last_logged",
    ]
    if session_best_df.empty:
        return pd.DataFrame(columns=columns)

    ordered = session_best_df.sort_values("workout_start_time")
    summary = (
        ordered.groupby(["exercise_key", "exercise_title"], as_index=False)
        .agg(
            session_count=("workout_id", "nunique"),
            total_working_sets=("working_set_count", "sum"),
            best_1rm_kg=("estimated_1rm_kg", "max"),
            first_logged=("workout_start_time", "min"),
            last_logged=("workout_start_time", "max"),
        )
    )

    latest = (
        ordered.groupby(["exercise_key", "exercise_title"], as_index=False)
        .tail(1)[["exercise_key", "exercise_title", "estimated_1rm_kg"]]
        .rename(columns={"estimated_1rm_kg": "latest_1rm_kg"})
    )
    summary = summary.merge(latest, on=["exercise_key", "exercise_title"], how="left")

    summary = summary.sort_values(
        ["session_count", "total_working_sets", "last_logged"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return summary.head(top_n)[columns]
