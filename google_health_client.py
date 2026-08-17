"""
Google Health API client (health.googleapis.com/v4) with local JSON caching.

Replacement for fitbit_client.py: the Fitbit Web API sunsets in September 2026
and Fitbit/Pixel device data moves to the Google Health API. Weight is NOT
served here — the Garmin scale never reaches Google's cloud (see
garmin_client.py).

Cache files in .google_health_cache/ use the SAME record schemas as
.fitbit_cache/ so DataFrames are drop-in compatible with the app.

Auth: run `google_health_login.py` once after creating an OAuth client in a
Google Cloud project (see that file's docstring). Tokens are stored in
google_health_config.json.

NOTE on parsing: exact dataPoint payload shapes are extracted defensively
(candidate key lists) because Google's docs don't publish full examples for
every type. compare_fitbit_google.py validates real responses against the
Fitbit archive before the app switches over.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

API_BASE = "https://health.googleapis.com/v4"
TOKEN_URL = "https://oauth2.googleapis.com/token"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
REQUEST_TIMEOUT = (10, 30)

SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
]

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "google_health_config.json")
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".google_health_cache")
CACHE_MAX_AGE_SECONDS = 86400

# Earliest data in the Fitbit archive; full backfills start here.
EARLIEST_DATE = "2021-09-11"


# ---------------------------------------------------------------------------
# Config / auth
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def is_configured() -> bool:
    return bool(load_config().get("client_id"))


def has_valid_token() -> bool:
    config = load_config()
    if config.get("refresh_token"):
        return True
    if config.get("access_token"):
        return datetime.now().timestamp() < config.get("expires_at", 0)
    return False


def build_auth_url(client_id: str, redirect_uri: str) -> str:
    from urllib.parse import urlencode

    return AUTH_URL + "?" + urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            # Per Google Health API docs: do NOT pass include_granted_scopes.
        }
    )


def exchange_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    config = load_config()
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": config["client_id"],
            "client_secret": config.get("client_secret", ""),
            "redirect_uri": redirect_uri,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    tokens = resp.json()
    config["access_token"] = tokens["access_token"]
    config["refresh_token"] = tokens.get("refresh_token", config.get("refresh_token"))
    config["expires_at"] = datetime.now().timestamp() + tokens.get("expires_in", 3600) - 60
    save_config(config)
    return config


def _get_access_token() -> str:
    config = load_config()
    if config.get("access_token") and datetime.now().timestamp() < config.get("expires_at", 0):
        return config["access_token"]
    if not config.get("refresh_token"):
        raise RuntimeError(
            "Google Health API is not authorized yet. Run "
            "`.venv/bin/python google_health_login.py` once."
        )
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": config["refresh_token"],
            "client_id": config["client_id"],
            "client_secret": config.get("client_secret", ""),
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    tokens = resp.json()
    config["access_token"] = tokens["access_token"]
    config["expires_at"] = datetime.now().timestamp() + tokens.get("expires_in", 3600) - 60
    save_config(config)
    return config["access_token"]


def _api_request(method: str, path: str, params: Optional[Dict] = None,
                 body: Optional[Dict] = None) -> Dict[str, Any]:
    token = _get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    for attempt in range(3):
        resp = requests.request(
            method, API_BASE + path, headers=headers, params=params, json=body,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 429:
            time.sleep(min(int(resp.headers.get("Retry-After", 5)), 60))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def _list_data_points(data_type: str, filter_expr: str,
                      reconcile: bool = True) -> List[Dict[str, Any]]:
    """List (optionally reconciled) dataPoints for a type, following pages."""
    suffix = ":reconcile" if reconcile else ""
    path = f"/users/me/dataTypes/{data_type}/dataPoints{suffix}"
    points: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {"filter": filter_expr}
    while True:
        payload = _api_request("GET", path, params=params)
        points.extend(payload.get("dataPoints", []))
        next_token = payload.get("nextPageToken")
        if not next_token:
            return points
        params = {"filter": filter_expr, "pageToken": next_token}


def _daily_rollup(data_type: str, start: date, end: date) -> List[Dict[str, Any]]:
    """dailyRollUp for an interval type over [start, end]."""
    body = {
        "range": {
            "start": {
                "date": {"year": start.year, "month": start.month, "day": start.day},
                "time": {"hours": 0, "minutes": 0, "seconds": 0, "nanos": 0},
            },
            "end": {
                "date": {"year": end.year, "month": end.month, "day": end.day},
                "time": {"hours": 23, "minutes": 59, "seconds": 59, "nanos": 0},
            },
        },
        "windowSizeDays": 1,
    }
    payload = _api_request(
        "POST", f"/users/me/dataTypes/{data_type}/dataPoints:dailyRollUp", body=body
    )
    return payload.get("rollupDataPoints", [])


# ---------------------------------------------------------------------------
# Defensive value extraction
# ---------------------------------------------------------------------------

def _civil_date(obj: Dict[str, Any]) -> Optional[str]:
    """Extract YYYY-MM-DD from a rollup/civil-time style point."""
    for key in ("civilStartTime", "civilEndTime"):
        block = obj.get(key)
        if isinstance(block, dict):
            d = block.get("date") or {}
            if d.get("year"):
                return f"{d['year']:04d}-{d.get('month', 1):02d}-{d.get('day', 1):02d}"
    for key in ("startTime", "sampleTime", "physicalTime", "civilTime"):
        v = obj.get(key)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    return None


def _first_number(obj: Any, skip_keys=("year", "month", "day", "hours", "minutes",
                                       "seconds", "nanos")) -> Optional[float]:
    """Depth-first search for the first numeric leaf in a payload value."""
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip_keys:
                continue
            found = _first_number(v, skip_keys)
            if found is not None:
                return found
    if isinstance(obj, list):
        for v in obj:
            found = _first_number(v, skip_keys)
            if found is not None:
                return found
    return None


def _extract_metric_value(point: Dict[str, Any], candidates: List[str]) -> Optional[float]:
    """Pull a numeric value from a dataPoint, trying candidate field names."""
    for path_keys in candidates:
        obj: Any = point
        ok = True
        for key in path_keys.split("."):
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                ok = False
                break
        if ok:
            val = _first_number(obj)
            if val is not None:
                return val
    # Last resort: any numeric leaf outside time fields.
    stripped = {k: v for k, v in point.items()
                if k not in ("startTime", "endTime", "sampleTime", "civilStartTime",
                             "civilEndTime", "dataSource", "id", "name")}
    return _first_number(stripped)


# ---------------------------------------------------------------------------
# Cache plumbing (mirrors fitbit_client)
# ---------------------------------------------------------------------------

def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(metric: str) -> str:
    return os.path.join(CACHE_DIR, f"{metric}.json")


def is_cache_fresh(metric: str) -> bool:
    path = _cache_path(metric)
    if not os.path.exists(path):
        return False
    return datetime.now().timestamp() - os.path.getmtime(path) < CACHE_MAX_AGE_SECONDS


def _load_cache(metric: str) -> List[Dict]:
    path = _cache_path(metric)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_cache(metric: str, records: List[Dict]) -> None:
    _ensure_cache_dir()
    with open(_cache_path(metric), "w") as f:
        json.dump(records, f)


def _resolve_fetch_start(metric: str, cached: List[Dict], start_date: Optional[str],
                         force_full: bool) -> Optional[date]:
    if not force_full and is_cache_fresh(metric) and cached:
        return None
    if cached and not force_full:
        last = max(r["date"] for r in cached if r.get("date"))
        return date.fromisoformat(last) + timedelta(days=1)
    if start_date:
        return date.fromisoformat(start_date)
    return date.fromisoformat(EARLIEST_DATE)


def _merge_and_save(metric: str, cached: List[Dict], new_records: List[Dict]) -> List[Dict]:
    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    return all_records


def _windows(start: date, end: date, days: int):
    window_start = start
    while window_start <= end:
        window_end = min(window_start + timedelta(days=days - 1), end)
        yield window_start, window_end
        window_start = window_end + timedelta(days=1)


# Candidate filter shapes per data-type kind. Google documents
# `steps.interval.civil_start_time` (Interval) and
# `body_fat.sample_time.physical_time` (Sample); the Daily/Session shapes are
# probed at runtime and the first accepted template is remembered per type.
_FILTER_TEMPLATES = [
    '{t}.interval.civil_start_time >= "{s}T00:00:00" AND '
    '{t}.interval.civil_start_time <= "{e}T23:59:59"',
    '{t}.civil_date >= "{s}" AND {t}.civil_date <= "{e}"',
    '{t}.sample_time.physical_time >= "{s}T00:00:00Z" AND '
    '{t}.sample_time.physical_time <= "{e}T23:59:59Z"',
    '{t}.session.civil_start_time >= "{s}T00:00:00" AND '
    '{t}.session.civil_start_time <= "{e}T23:59:59"',
]

_FILTER_CHOICE_PATH = os.path.join(CACHE_DIR, "filter_templates.json")


def _load_filter_choices() -> Dict[str, int]:
    if os.path.exists(_FILTER_CHOICE_PATH):
        with open(_FILTER_CHOICE_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_filter_choice(data_type: str, index: int) -> None:
    _ensure_cache_dir()
    choices = _load_filter_choices()
    choices[data_type] = index
    with open(_FILTER_CHOICE_PATH, "w") as f:
        json.dump(choices, f)


def _list_points_for_range(data_type: str, snake: str, start: date,
                           end: date) -> List[Dict[str, Any]]:
    """List dataPoints in a civil-date range, probing filter templates until
    one is accepted (HTTP 400 = wrong shape for this type's kind)."""
    choices = _load_filter_choices()
    order = list(range(len(_FILTER_TEMPLATES)))
    if data_type in choices:
        order.insert(0, order.pop(order.index(choices[data_type])))
    last_error: Optional[Exception] = None
    for idx in order:
        expr = _FILTER_TEMPLATES[idx].format(
            t=snake, s=start.isoformat(), e=end.isoformat()
        )
        try:
            points = _list_data_points(data_type, expr)
            if choices.get(data_type) != idx:
                _save_filter_choice(data_type, idx)
            return points
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                last_error = e
                continue
            raise
    raise RuntimeError(
        f"No accepted filter shape for {data_type}; last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Metric fetchers — same record schemas as fitbit_client caches
# ---------------------------------------------------------------------------

# rollup window limits per docs: 14 days for total-calories/active-minutes/
# heart-rate familes, 90 days otherwise.
SHORT_WINDOW_TYPES = {"total-calories", "active-minutes", "heart-rate",
                      "calories-in-heart-rate-zone"}


def _rollup_series(data_type: str, start: date, end: date,
                   progress_cb=None, progress_range=(0.0, 1.0)) -> Dict[str, float]:
    """Daily values for an interval type via dailyRollUp, windowed."""
    values: Dict[str, float] = {}
    window_days = 14 if data_type in SHORT_WINDOW_TYPES else 90
    spans = list(_windows(start, end, window_days))
    for i, (w_start, w_end) in enumerate(spans):
        if progress_cb:
            lo, hi = progress_range
            progress_cb(lo + (hi - lo) * i / max(len(spans), 1))
        for point in _daily_rollup(data_type, w_start, w_end):
            d = _civil_date(point)
            if not d:
                continue
            val = _extract_metric_value(point, [
                data_type.replace("-", "_"),
                "value",
                "count",
            ])
            if val is not None:
                values[d] = val
    return values


def fetch_activity(start_date: Optional[str] = None, force_full: bool = False,
                   progress_cb=None) -> pd.DataFrame:
    """
    Daily activity via dailyRollUp: steps, distance, calories, active minutes.
    NOTE: Fitbit's four intensity buckets (sedentary/lightly/fairly/very) and
    three heart zones (fatBurn/cardio/peak) do not map 1:1 onto Google Health
    data types; fields without a Google equivalent stay None so the gap is
    visible rather than silently zero-filled.
    """
    metric = "activity"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _activity_records_to_df(cached)
    today = date.today()
    if fetch_start > today:
        return _activity_records_to_df(cached)

    steps = _rollup_series("steps", fetch_start, today, progress_cb, (0.0, 0.3))
    distance = _rollup_series("distance", fetch_start, today, progress_cb, (0.3, 0.5))
    calories = _rollup_series("total-calories", fetch_start, today, progress_cb, (0.5, 0.8))
    active_min = _rollup_series("active-minutes", fetch_start, today, progress_cb, (0.8, 0.9))
    azm = _rollup_series("active-zone-minutes", fetch_start, today, progress_cb, (0.9, 1.0))

    all_dates = sorted(set(steps) | set(distance) | set(calories) | set(active_min) | set(azm))
    new_records = []
    for d in all_dates:
        new_records.append({
            "date": d,
            "steps": steps.get(d),
            "calories": calories.get(d),
            "distance": distance.get(d),
            "activeMinutes": active_min.get(d),
            "zoneMinutes": azm.get(d),
            # No 1:1 Google equivalents (kept for schema compatibility):
            "minutesFairlyActive": None,
            "minutesVeryActive": None,
            "minutesSedentary": None,
            "minutesLightlyActive": None,
            "minutesFatBurn": None,
            "minutesCardio": None,
            "minutesPeak": None,
        })
    all_records = _merge_and_save(metric, cached, new_records)
    if progress_cb:
        progress_cb(1.0)
    return _activity_records_to_df(all_records)


def _activity_records_to_df(records: List[Dict]) -> pd.DataFrame:
    cols = ["Date", "Steps", "Calories", "Distance", "ActiveMinutes", "ZoneMinutes",
            "MinutesFairlyActive", "MinutesVeryActive", "MinutesSedentary",
            "MinutesLightlyActive", "MinutesFatBurn", "MinutesCardio", "MinutesPeak"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={
        "steps": "Steps", "calories": "Calories", "distance": "Distance",
        "activeMinutes": "ActiveMinutes", "zoneMinutes": "ZoneMinutes",
    })
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_rhr(start_date: Optional[str] = None, force_full: bool = False,
              progress_cb=None) -> pd.DataFrame:
    """Daily resting heart rate via daily-resting-heart-rate list."""
    metric = "rhr"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _simple_records_to_df(cached, "rhr", "RHR")
    today = date.today()
    if fetch_start > today:
        return _simple_records_to_df(cached, "rhr", "RHR")

    new_records = []
    spans = list(_windows(fetch_start, today, 90))
    for i, (w_start, w_end) in enumerate(spans):
        if progress_cb:
            progress_cb(i / max(len(spans), 1))
        points = _list_points_for_range(
            "daily-resting-heart-rate", "daily_resting_heart_rate", w_start, w_end
        )
        for point in points:
            d = _civil_date(point)
            val = _extract_metric_value(point, [
                "dailyRestingHeartRate", "restingHeartRate", "value", "bpm",
            ])
            if d and val is not None:
                new_records.append({"date": d, "rhr": val})
    all_records = _merge_and_save(metric, cached, new_records)
    if progress_cb:
        progress_cb(1.0)
    return _simple_records_to_df(all_records, "rhr", "RHR")


def fetch_hrv(start_date: Optional[str] = None, force_full: bool = False,
              progress_cb=None) -> pd.DataFrame:
    """Daily HRV (RMSSD) via daily-heart-rate-variability list."""
    metric = "hrv"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _hrv_records_to_df(cached)
    today = date.today()
    if fetch_start > today:
        return _hrv_records_to_df(cached)

    new_records = []
    spans = list(_windows(fetch_start, today, 90))
    for i, (w_start, w_end) in enumerate(spans):
        if progress_cb:
            progress_cb(i / max(len(spans), 1))
        points = _list_points_for_range(
            "daily-heart-rate-variability", "daily_heart_rate_variability", w_start, w_end
        )
        for point in points:
            d = _civil_date(point)
            val = _extract_metric_value(point, [
                "dailyHeartRateVariability", "rmssd", "dailyRmssd", "value",
            ])
            if d and val is not None:
                new_records.append({"date": d, "rmssd": val,
                                    "coverage": None, "hf": None, "lf": None})
    all_records = _merge_and_save(metric, cached, new_records)
    if progress_cb:
        progress_cb(1.0)
    return _hrv_records_to_df(all_records)


def _hrv_records_to_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "RMSSD", "Coverage", "HF", "LF"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"rmssd": "RMSSD", "coverage": "Coverage",
                            "hf": "HF", "lf": "LF"})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_breathing_rate(start_date: Optional[str] = None, force_full: bool = False,
                         progress_cb=None) -> pd.DataFrame:
    """Daily respiratory rate via daily-respiratory-rate list."""
    metric = "breathing_rate"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _simple_records_to_df(cached, "breathing_rate", "BreathingRate")
    today = date.today()
    if fetch_start > today:
        return _simple_records_to_df(cached, "breathing_rate", "BreathingRate")

    new_records = []
    spans = list(_windows(fetch_start, today, 90))
    for i, (w_start, w_end) in enumerate(spans):
        if progress_cb:
            progress_cb(i / max(len(spans), 1))
        points = _list_points_for_range(
            "daily-respiratory-rate", "daily_respiratory_rate", w_start, w_end
        )
        for point in points:
            d = _civil_date(point)
            val = _extract_metric_value(point, [
                "dailyRespiratoryRate", "respiratoryRate", "breathingRate", "value",
            ])
            if d and val is not None:
                new_records.append({"date": d, "breathing_rate": val})
    all_records = _merge_and_save(metric, cached, new_records)
    if progress_cb:
        progress_cb(1.0)
    return _simple_records_to_df(all_records, "breathing_rate", "BreathingRate")


def _simple_records_to_df(records: List[Dict], field: str, column: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", column])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={field: column})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def _minutes(value: Any) -> Optional[float]:
    """Convert a duration payload (seconds string like '3600s', millis, or
    {seconds: N}) to minutes."""
    if value is None:
        return None
    if isinstance(value, dict):
        secs = value.get("seconds")
        if secs is not None:
            return float(secs) / 60.0
        value = _first_number(value)
    if isinstance(value, str):
        if value.endswith("s"):
            try:
                return float(value[:-1]) / 60.0
            except ValueError:
                return None
        try:
            value = float(value)
        except ValueError:
            return None
    if isinstance(value, (int, float)):
        # Heuristic: > 24h in minutes means it's probably seconds or millis.
        v = float(value)
        if v > 86400:
            return v / 60000.0
        if v > 1440:
            return v / 60.0
        return v
    return None


def fetch_sleep(start_date: Optional[str] = None, force_full: bool = False,
                progress_cb=None) -> pd.DataFrame:
    """
    Sleep sessions (main sleep per day) with stage minutes. Sessions are
    paged (cap 25/page). Raw first-session payload is saved once to
    .google_health_cache/sleep_raw_sample.json to make schema drift visible.
    """
    metric = "sleep"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _sleep_records_to_df(cached)
    today = date.today()
    if fetch_start > today:
        return _sleep_records_to_df(cached)

    new_records = []
    spans = list(_windows(fetch_start, today, 90))
    saved_sample = os.path.exists(os.path.join(CACHE_DIR, "sleep_raw_sample.json"))
    for i, (w_start, w_end) in enumerate(spans):
        if progress_cb:
            progress_cb(i / max(len(spans), 1))
        sessions = _list_points_for_range("sleep", "sleep", w_start, w_end)
        for session in sessions:
            if not saved_sample:
                _ensure_cache_dir()
                with open(os.path.join(CACHE_DIR, "sleep_raw_sample.json"), "w") as f:
                    json.dump(session, f, indent=2)
                saved_sample = True
            record = _parse_sleep_session(session)
            if record:
                new_records.append(record)

    # Keep the longest session per date (main sleep).
    best_by_date: Dict[str, Dict] = {}
    for r in new_records:
        cur = best_by_date.get(r["date"])
        if cur is None or (r.get("duration_minutes") or 0) > (cur.get("duration_minutes") or 0):
            best_by_date[r["date"]] = r
    all_records = _merge_and_save(metric, cached, list(best_by_date.values()))
    if progress_cb:
        progress_cb(1.0)
    return _sleep_records_to_df(all_records)


_SLEEP_STAGE_KEYS = {
    "rem_minutes": ("rem", "remSleep", "REM"),
    "deep_minutes": ("deep", "deepSleep", "DEEP"),
    "light_minutes": ("light", "lightSleep", "LIGHT"),
    "wake_minutes": ("wake", "awake", "AWAKE", "WAKE"),
}


def _parse_sleep_session(session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    body = session.get("sleep") or session
    d = _civil_date(session) or _civil_date(body)
    if not d:
        return None
    record: Dict[str, Any] = {
        "date": d,
        "duration_minutes": _minutes(
            body.get("duration") or body.get("totalSleepDuration")
            or body.get("timeAsleep")
        ),
        "efficiency": _first_number(body.get("efficiency")),
        "score": _first_number(body.get("score") or body.get("sleepScore")),
    }
    # Stage summaries: look for a dict of stage -> duration.
    summary = (body.get("stageSummary") or body.get("summary")
               or body.get("levels") or {})
    for field, keys in _SLEEP_STAGE_KEYS.items():
        val = None
        for key in keys:
            if isinstance(summary, dict) and key in summary:
                val = _minutes(summary[key])
                break
            if key in body:
                val = _minutes(body[key])
                break
        record[field] = val
    if record["duration_minutes"] is None:
        stage_sum = sum(v for k, v in record.items()
                        if k.endswith("_minutes") and isinstance(v, (int, float)))
        record["duration_minutes"] = stage_sum or None
    return record


def _sleep_records_to_df(records: List[Dict]) -> pd.DataFrame:
    cols = ["Date", "DurationMinutes", "Efficiency", "Score",
            "REM", "Deep", "Light", "Wake"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={
        "duration_minutes": "DurationMinutes",
        "efficiency": "Efficiency",
        "score": "Score",
        "rem_minutes": "REM",
        "deep_minutes": "Deep",
        "light_minutes": "Light",
        "wake_minutes": "Wake",
    })
    df = df.drop(columns=["date"], errors="ignore")
    df["DurationHours"] = df["DurationMinutes"] / 60
    return df.sort_values("Date").reset_index(drop=True)


def load_cached_dataframe(metric: str) -> pd.DataFrame:
    records = _load_cache(metric)
    if metric == "activity":
        return _activity_records_to_df(records)
    if metric == "hrv":
        return _hrv_records_to_df(records)
    if metric == "rhr":
        return _simple_records_to_df(records, "rhr", "RHR")
    if metric == "breathing_rate":
        return _simple_records_to_df(records, "breathing_rate", "BreathingRate")
    if metric == "sleep":
        return _sleep_records_to_df(records)
    raise ValueError(f"Unknown metric: {metric}")
