"""
Fitbit API client with local JSON caching.

Uses OAuth2 Authorization Code Grant with PKCE for authentication.
Caches data locally so we only fetch new data since the last sync.
"""

import os
import json
import hashlib
import secrets
import base64
import urllib.parse
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Any

import pandas as pd
import requests

# Fitbit API endpoints
FITBIT_AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_API_BASE = "https://api.fitbit.com"
REQUEST_TIMEOUT = (10, 30)
MAX_HTTP_RETRIES = 3
MAX_RATE_LIMIT_WAIT_SECONDS = 300

# Local cache directory
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".fitbit_cache")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "fitbit_config.json")


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load Fitbit OAuth config from disk."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_config(config: Dict[str, Any]):
    """Save Fitbit OAuth config to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def is_configured() -> bool:
    """Check if Fitbit client ID is configured."""
    config = load_config()
    return bool(config.get("client_id"))


def has_valid_token() -> bool:
    """Check if we have a non-expired access token (or a refresh token)."""
    config = load_config()
    if config.get("refresh_token"):
        return True
    if config.get("access_token"):
        expires_at = config.get("expires_at", 0)
        return datetime.now().timestamp() < expires_at
    return False


def generate_pkce_pair() -> tuple:
    """Generate PKCE code_verifier and code_challenge."""
    code_verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def get_auth_url(client_id: str, redirect_uri: str = "http://localhost:8501") -> tuple:
    """
    Build the Fitbit OAuth2 authorization URL with PKCE.
    Returns (url, code_verifier) — store the verifier for the token exchange.
    """
    code_verifier, code_challenge = generate_pkce_pair()

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "profile weight heartrate respiratory_rate sleep activity",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    url = f"{FITBIT_AUTH_URL}?{urllib.parse.urlencode(params)}"
    return url, code_verifier


def _post_fitbit_token(data: Dict[str, Any]) -> requests.Response:
    return requests.post(
        FITBIT_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=REQUEST_TIMEOUT,
    )


def exchange_code_for_token(
    client_id: str,
    auth_code: str,
    code_verifier: str,
    redirect_uri: str = "http://localhost:8501",
) -> Dict[str, Any]:
    """Exchange authorization code for access + refresh tokens."""
    resp = _post_fitbit_token(
        {
            "client_id": client_id,
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
    )
    resp.raise_for_status()
    token_data = resp.json()

    # Persist tokens
    config = load_config()
    config["client_id"] = client_id
    config["access_token"] = token_data["access_token"]
    config["refresh_token"] = token_data["refresh_token"]
    config["expires_at"] = datetime.now().timestamp() + token_data.get("expires_in", 28800)
    config["user_id"] = token_data.get("user_id", "")
    save_config(config)

    return token_data


def refresh_access_token() -> str:
    """Use refresh token to get a new access token."""
    config = load_config()
    refresh_token = config.get("refresh_token")
    client_id = config.get("client_id")
    if not refresh_token or not client_id:
        raise ValueError("No refresh token available. Please re-authorize.")

    resp = _post_fitbit_token(
        {
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    )
    if resp.status_code == 400:
        error_data = resp.json() if resp.text else {}
        error_type = ""
        for err in error_data.get("errors", []):
            error_type = err.get("errorType", "")
        if error_type == "invalid_grant":
            raise ValueError(
                "Refresh token has been revoked. Please re-authorize on the Fitbit Config page."
            )
    resp.raise_for_status()
    token_data = resp.json()

    config["access_token"] = token_data["access_token"]
    config["refresh_token"] = token_data.get("refresh_token", refresh_token)
    config["expires_at"] = datetime.now().timestamp() + token_data.get("expires_in", 28800)
    save_config(config)

    return token_data["access_token"]


def _get_access_token() -> str:
    """Get a valid access token, refreshing if needed."""
    config = load_config()
    expires_at = config.get("expires_at", 0)
    if datetime.now().timestamp() >= expires_at:
        return refresh_access_token()
    return config["access_token"]


# Module-level callback for rate-limit notifications (set by the UI layer)
_rate_limit_cb = None


def set_rate_limit_callback(cb):
    """Set a callback to be called when rate-limited. cb(seconds_to_wait)."""
    global _rate_limit_cb
    _rate_limit_cb = cb


def _api_get(path: str, params: Optional[Dict] = None) -> Dict:
    """Make an authenticated GET request to the Fitbit API.
    Automatically waits and retries on rate limits, transient network failures,
    and 5xx server errors, while keeping wait times bounded.
    """
    import time

    token = _get_access_token()
    retries = 0
    rate_limit_retries = 0
    auth_retry_done = False
    while True:
        try:
            resp = requests.get(
                f"{FITBIT_API_BASE}{path}",
                params=params,
                headers={"Authorization": f"Bearer {token}"},
                timeout=REQUEST_TIMEOUT,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if retries >= MAX_HTTP_RETRIES:
                raise requests.exceptions.RequestException(
                    f"Fitbit request failed after {MAX_HTTP_RETRIES} retries: {exc}"
                ) from exc
            retries += 1
            wait = min(5 * retries, 30)
            if _rate_limit_cb:
                _rate_limit_cb(wait)
            time.sleep(wait)
            continue

        if resp.status_code == 401:
            if auth_retry_done:
                resp.raise_for_status()
            token = refresh_access_token()
            auth_retry_done = True
            continue
        if resp.status_code == 429:
            if rate_limit_retries >= MAX_HTTP_RETRIES:
                resp.raise_for_status()
            retry_after = int(resp.headers.get("Retry-After", 60))
            retry_after = max(1, min(retry_after, MAX_RATE_LIMIT_WAIT_SECONDS))
            if _rate_limit_cb:
                _rate_limit_cb(retry_after)
            time.sleep(retry_after)
            rate_limit_retries += 1
            continue
        if resp.status_code >= 500 and retries < MAX_HTTP_RETRIES:
            retries += 1
            wait = min(5 * retries, 30)
            if _rate_limit_cb:
                _rate_limit_cb(wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()


def _get_member_since() -> date:
    """Get the user's Fitbit member-since date from their profile."""
    try:
        data = _api_get("/1/user/-/profile.json")
        member_since = data.get("user", {}).get("memberSince", "")
        if member_since:
            return datetime.strptime(member_since, "%Y-%m-%d").date()
    except Exception:
        pass
    # Fallback: 5 years back if profile call fails
    return date.today() - timedelta(days=5 * 365)


# ---------------------------------------------------------------------------
# Cached data fetchers — each metric has its own cache file.
# Cache files store a list of daily records as JSON.
# On fetch, we only request data from (last_cached_date + 1) to today.
# ---------------------------------------------------------------------------

CACHE_MAX_AGE_SECONDS = 86400  # Skip API calls if cache is less than 24 hours old


def _cache_path(metric: str) -> str:
    return os.path.join(CACHE_DIR, f"{metric}.json")


def _cache_is_fresh(metric: str) -> bool:
    """Return True if the cache file exists and was written recently."""
    path = _cache_path(metric)
    if not os.path.exists(path):
        return False
    age = datetime.now().timestamp() - os.path.getmtime(path)
    return age < CACHE_MAX_AGE_SECONDS


def _load_cache(metric: str) -> List[Dict]:
    path = _cache_path(metric)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_cache(metric: str, records: List[Dict]):
    _ensure_cache_dir()
    with open(_cache_path(metric), "w") as f:
        json.dump(records, f)


def _last_cached_date(metric: str) -> Optional[date]:
    records = _load_cache(metric)
    if not records:
        return None
    dates = [r["date"] for r in records if r.get("date")]
    if not dates:
        return None
    return max(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)


def _resolve_fetch_start(metric, cached, start_date, force_full):
    """Common logic to determine fetch start date.
    Returns None if cache is fresh and no fetch is needed.
    """
    if not force_full and _cache_is_fresh(metric) and cached:
        return None
    last = _last_cached_date(metric) if cached else None
    if last and not force_full:
        return last + timedelta(days=1)
    fetch_start = _get_member_since()
    if start_date:
        fetch_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    return fetch_start


def fetch_weight(start_date: Optional[str] = None, force_full: bool = False,
                 progress_cb=None) -> pd.DataFrame:
    """
    Fetch weight data. Uses cache; only pulls new data since last sync.
    Fitbit weight time series endpoint: max 31-day windows.
    """
    metric = "weight"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _weight_records_to_df(cached)

    today = date.today()
    if fetch_start > today:
        return _weight_records_to_df(cached)

    # Fetch in 30-day windows (API limit for weight log)
    new_records = []
    total_days = (today - fetch_start).days + 1
    window_start = fetch_start
    while window_start <= today:
        if progress_cb:
            done = (window_start - fetch_start).days
            progress_cb(min(done / max(total_days, 1), 1.0))
        window_end = min(window_start + timedelta(days=30), today)
        data = _api_get(
            f"/1/user/-/body/log/weight/date/"
            f"{window_start.isoformat()}/{window_end.isoformat()}.json"
        )
        for entry in data.get("weight", []):
            new_records.append({
                "date": entry["date"],
                "weight": entry.get("weight"),
                "bmi": entry.get("bmi"),
                "fat": entry.get("fat"),
            })
        window_start = window_end + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _weight_records_to_df(all_records)


def _weight_records_to_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "Weight", "BMI", "Fat"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"weight": "Weight", "bmi": "BMI", "fat": "Fat"})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_hrv(start_date: Optional[str] = None, force_full: bool = False,
              progress_cb=None) -> pd.DataFrame:
    """
    Fetch HRV data. Fitbit HRV endpoint returns daily RMSSD.
    Endpoint: GET /1/user/-/hrv/date/{start}/{end}.json  (max 30 days)
    """
    metric = "hrv"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _hrv_records_to_df(cached)

    today = date.today()
    if fetch_start > today:
        return _hrv_records_to_df(cached)

    new_records = []
    total_days = (today - fetch_start).days + 1
    window_start = fetch_start
    while window_start <= today:
        if progress_cb:
            progress_cb(min((window_start - fetch_start).days / max(total_days, 1), 1.0))
        window_end = min(window_start + timedelta(days=30), today)
        data = _api_get(
            f"/1/user/-/hrv/date/"
            f"{window_start.isoformat()}/{window_end.isoformat()}.json"
        )
        for entry in data.get("hrv", []):
            val = entry.get("value", {})
            new_records.append({
                "date": entry["dateTime"],
                "rmssd": val.get("dailyRmssd"),
                "coverage": val.get("coverage"),
                "hf": val.get("hf"),
                "lf": val.get("lf"),
            })
        window_start = window_end + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _hrv_records_to_df(all_records)


def _hrv_records_to_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "RMSSD", "Coverage", "HF", "LF"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"rmssd": "RMSSD", "coverage": "Coverage", "hf": "HF", "lf": "LF"})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_rhr(start_date: Optional[str] = None, force_full: bool = False,
              progress_cb=None) -> pd.DataFrame:
    """
    Fetch Resting Heart Rate from the heart rate time series.
    Endpoint: GET /1/user/-/activities/heart/date/{start}/{end}.json  (max 90 days)
    """
    metric = "rhr"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _rhr_records_to_df(cached)

    today = date.today()
    if fetch_start > today:
        return _rhr_records_to_df(cached)

    new_records = []
    total_days = (today - fetch_start).days + 1
    window_start = fetch_start
    while window_start <= today:
        if progress_cb:
            progress_cb(min((window_start - fetch_start).days / max(total_days, 1), 1.0))
        window_end = min(window_start + timedelta(days=89), today)
        data = _api_get(
            f"/1/user/-/activities/heart/date/"
            f"{window_start.isoformat()}/{window_end.isoformat()}.json"
        )
        for entry in data.get("activities-heart", []):
            rhr = entry.get("value", {}).get("restingHeartRate")
            if rhr is not None:
                new_records.append({
                    "date": entry["dateTime"],
                    "rhr": rhr,
                })
        window_start = window_end + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _rhr_records_to_df(all_records)


def _rhr_records_to_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "RHR"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"rhr": "RHR"})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_breathing_rate(start_date: Optional[str] = None, force_full: bool = False,
                         progress_cb=None) -> pd.DataFrame:
    """
    Fetch breathing rate data.
    Endpoint: GET /1/user/-/br/date/{start}/{end}.json  (max 30 days)
    """
    metric = "breathing_rate"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _breathing_rate_records_to_df(cached)

    today = date.today()
    if fetch_start > today:
        return _breathing_rate_records_to_df(cached)

    new_records = []
    total_days = (today - fetch_start).days + 1
    window_start = fetch_start
    while window_start <= today:
        if progress_cb:
            progress_cb(min((window_start - fetch_start).days / max(total_days, 1), 1.0))
        window_end = min(window_start + timedelta(days=30), today)
        data = _api_get(
            f"/1/user/-/br/date/"
            f"{window_start.isoformat()}/{window_end.isoformat()}.json"
        )
        for entry in data.get("br", []):
            val = entry.get("value", {})
            br_val = val.get("breathingRate")
            if br_val is not None:
                new_records.append({
                    "date": entry["dateTime"],
                    "breathing_rate": br_val,
                })
        window_start = window_end + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _breathing_rate_records_to_df(all_records)


def _breathing_rate_records_to_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "BreathingRate"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"breathing_rate": "BreathingRate"})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_sleep(start_date: Optional[str] = None, force_full: bool = False,
                progress_cb=None) -> pd.DataFrame:
    """
    Fetch sleep data.
    Endpoint: GET /1.2/user/-/sleep/date/{start}/{end}.json  (max 100 days)
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
    total_days = (today - fetch_start).days + 1
    window_start = fetch_start
    while window_start <= today:
        if progress_cb:
            progress_cb(min((window_start - fetch_start).days / max(total_days, 1), 1.0))
        window_end = min(window_start + timedelta(days=100), today)
        data = _api_get(
            f"/1.2/user/-/sleep/date/"
            f"{window_start.isoformat()}/{window_end.isoformat()}.json"
        )
        for entry in data.get("sleep", []):
            if not entry.get("isMainSleep", False):
                continue
            summary = entry.get("levels", {}).get("summary", {})
            record = {
                "date": entry["dateOfSleep"],
                "duration_minutes": entry.get("duration", 0) / 60000,
                "efficiency": entry.get("efficiency"),
                "score": None,
            }
            if "rem" in summary:
                record["rem_minutes"] = summary["rem"].get("minutes", 0)
                record["deep_minutes"] = summary["deep"].get("minutes", 0)
                record["light_minutes"] = summary["light"].get("minutes", 0)
                record["wake_minutes"] = summary["wake"].get("minutes", 0)
            else:
                record["rem_minutes"] = None
                record["deep_minutes"] = summary.get("deep", {}).get("minutes")
                record["light_minutes"] = summary.get("light", {}).get("minutes")
                record["wake_minutes"] = summary.get("awake", {}).get("minutes")
            new_records.append(record)
        window_start = window_end + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _sleep_records_to_df(all_records)


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
    # Convert duration to hours for display convenience
    df["DurationHours"] = df["DurationMinutes"] / 60
    return df.sort_values("Date").reset_index(drop=True)


def fetch_sleep_score(start_date: Optional[str] = None, force_full: bool = False,
                      progress_cb=None) -> pd.DataFrame:
    """
    Fetch sleep score from the sleep log list endpoint.
    Endpoint: GET /1.2/user/-/sleep/list.json?sort=asc&offset=0&limit=100
    """
    metric = "sleep_score"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _sleep_score_records_to_df(cached)

    today = date.today()
    if fetch_start > today:
        return _sleep_score_records_to_df(cached)

    new_records = []
    total_days = max((today - fetch_start).days, 1)
    after_date = fetch_start
    while after_date <= today:
        if progress_cb:
            progress_cb(min((after_date - fetch_start).days / total_days, 1.0))
        data = _api_get(
            "/1.2/user/-/sleep/list.json",
            params={"afterDate": after_date.isoformat(), "sort": "asc", "offset": 0, "limit": 100},
        )
        entries = data.get("sleep", [])
        if not entries:
            break
        for entry in entries:
            if not entry.get("isMainSleep", False):
                continue
            score = entry.get("sleepScore")
            if score is not None:
                new_records.append({
                    "date": entry["dateOfSleep"],
                    "sleep_score": score,
                })
        last_entry_date = datetime.strptime(entries[-1]["dateOfSleep"], "%Y-%m-%d").date()
        if last_entry_date >= today or last_entry_date <= after_date:
            break
        after_date = last_entry_date + timedelta(days=1)

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _sleep_score_records_to_df(all_records)


def _sleep_score_records_to_df(records: List[Dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["Date", "SleepScore"])
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"sleep_score": "SleepScore"})
    df = df.drop(columns=["date"], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_activity(start_date: Optional[str] = None, force_full: bool = False,
                   progress_cb=None) -> pd.DataFrame:
    """
    Fetch daily activity summary data using time-series endpoints.
    Activity time-series supports up to 1095-day windows.
    """
    metric = "activity"
    cached = [] if force_full else _load_cache(metric)
    fetch_start = _resolve_fetch_start(metric, cached, start_date, force_full)
    if fetch_start is None:
        return _activity_records_to_df(cached)

    today = date.today()
    if fetch_start > today:
        return _activity_records_to_df(cached)

    resources = {
        "steps": "activities/steps",
        "calories": "activities/calories",
        "distance": "activities/distance",
        "minutesFairlyActive": "activities/minutesFairlyActive",
        "minutesVeryActive": "activities/minutesVeryActive",
        "minutesSedentary": "activities/minutesSedentary",
        "minutesLightlyActive": "activities/minutesLightlyActive",
    }

    combined = {}
    resource_list = list(resources.items())
    for res_idx, (field, resource) in enumerate(resource_list):
        window_start = fetch_start
        while window_start <= today:
            if progress_cb:
                # Progress across resources and windows
                res_progress = res_idx / len(resource_list)
                total_days = max((today - fetch_start).days, 1)
                window_progress = (window_start - fetch_start).days / total_days / len(resource_list)
                progress_cb(min(res_progress + window_progress, 1.0))
            window_end = min(window_start + timedelta(days=89), today)
            data = _api_get(
                f"/1/user/-/{resource}/date/"
                f"{window_start.isoformat()}/{window_end.isoformat()}.json"
            )
            response_key = f"activities-{resource.split('/')[-1]}"
            for entry in data.get(response_key, []):
                d = entry["dateTime"]
                if d not in combined:
                    combined[d] = {"date": d}
                try:
                    combined[d][field] = float(entry["value"])
                except (ValueError, TypeError):
                    combined[d][field] = 0
            window_start = window_end + timedelta(days=1)

    new_records = list(combined.values())

    by_date = {r["date"]: r for r in cached}
    for r in new_records:
        by_date[r["date"]] = r
    all_records = sorted(by_date.values(), key=lambda r: r["date"])
    _save_cache(metric, all_records)
    if progress_cb:
        progress_cb(1.0)

    return _activity_records_to_df(all_records)


def _activity_records_to_df(records: List[Dict]) -> pd.DataFrame:
    cols = ["Date", "Steps", "Calories", "Distance",
            "MinutesFairlyActive", "MinutesVeryActive",
            "MinutesLightlyActive", "MinutesSedentary"]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={
        "steps": "Steps",
        "calories": "Calories",
        "distance": "Distance",
        "minutesFairlyActive": "MinutesFairlyActive",
        "minutesVeryActive": "MinutesVeryActive",
        "minutesLightlyActive": "MinutesLightlyActive",
        "minutesSedentary": "MinutesSedentary",
    })
    df = df.drop(columns=["date"], errors="ignore")
    # Active Zone Minutes = fairly + very active
    if "MinutesFairlyActive" in df.columns and "MinutesVeryActive" in df.columns:
        df["ZoneMinutes"] = df["MinutesFairlyActive"] + df["MinutesVeryActive"]
    return df.sort_values("Date").reset_index(drop=True)


def load_cached_dataframe(metric: str) -> pd.DataFrame:
    """Return cached metric data as a dataframe, or an empty one if unavailable."""
    records = _load_cache(metric)
    converters = {
        "weight": _weight_records_to_df,
        "hrv": _hrv_records_to_df,
        "rhr": _rhr_records_to_df,
        "breathing_rate": _breathing_rate_records_to_df,
        "sleep": _sleep_records_to_df,
        "sleep_score": _sleep_score_records_to_df,
        "activity": _activity_records_to_df,
    }
    if metric not in converters:
        raise ValueError(f"Unsupported metric: {metric}")
    return converters[metric](records)


def clear_cache():
    """Remove all cached Fitbit data."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)


def disconnect():
    """Remove stored tokens and config."""
    if os.path.exists(CONFIG_PATH):
        os.remove(CONFIG_PATH)
    clear_cache()
