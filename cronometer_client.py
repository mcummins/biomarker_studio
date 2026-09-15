"""Cronometer nutrition data.

Cronometer has no API, so the daily-summary export is saved by hand into
``cronometer_data/`` every so often. Every ``*.csv`` in that folder is read
and merged by date (the most recently modified file wins on overlap), so a
new export can simply be dropped alongside the old ones.

Only days with an energy entry count as logged; Cronometer's export lists
every calendar day, blank when nothing was recorded.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cronometer_data")

# Export column -> tidy column. Everything else in the export is kept in
# load_daily_summary(full=True) for the audit table.
COLUMN_MAP = {
    "Energy (kcal)": "Calories",
    "Protein (g)": "Protein",
    "Carbs (g)": "Carbs",
    "Net Carbs (g)": "NetCarbs",
    "Fat (g)": "Fat",
    "Fiber (g)": "Fiber",
    "Alcohol (g)": "Alcohol",
    "Sugars (g)": "Sugars",
    "Saturated (g)": "SaturatedFat",
    "Sodium (mg)": "Sodium",
    "Completed": "Completed",
}
TIDY_COLUMNS = ["Date"] + [c for c in COLUMN_MAP.values()]

# Atwater factors, kcal per gram, for the macro-split chart.
KCAL_PER_GRAM = {"Protein": 4.0, "Carbs": 4.0, "Fat": 9.0, "Alcohol": 7.0}


def list_exports() -> List[Tuple[str, float]]:
    """(path, mtime) for every CSV export, oldest first."""
    if not os.path.isdir(DATA_DIR):
        return []
    files = [
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.lower().endswith(".csv")
    ]
    return sorted(((path, os.path.getmtime(path)) for path in files), key=lambda item: item[1])


def signature() -> Tuple[Tuple[str, float], ...]:
    """Cache key: changes whenever an export is added or replaced."""
    return tuple((os.path.basename(path), mtime) for path, mtime in list_exports())


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=TIDY_COLUMNS)


def load_daily_summary(full: bool = False) -> pd.DataFrame:
    """One row per logged day, sorted by date.

    ``full=True`` keeps every export column (with the tidy names applied
    where they exist) instead of the tidy subset.
    """
    frames = []
    for path, _mtime in list_exports():
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "Date" not in frame.columns:
            continue
        frames.append(frame)
    if not frames:
        return _empty()
    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])
    # Later exports (appended last) win when the same day appears twice.
    data = data.drop_duplicates(subset="Date", keep="last")
    data = data.rename(columns=COLUMN_MAP)
    if "Calories" not in data.columns:
        return _empty()
    data["Calories"] = pd.to_numeric(data["Calories"], errors="coerce")
    data = data.dropna(subset=["Calories"])
    for column in COLUMN_MAP.values():
        if column == "Completed":
            continue
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = float("nan")
    if "Completed" in data.columns:
        data["Completed"] = data["Completed"].astype(str).str.strip().str.lower().eq("true")
    else:
        data["Completed"] = False
    data = data.sort_values("Date").reset_index(drop=True)
    if full:
        return data
    return data[TIDY_COLUMNS].copy()


def latest_logged_date() -> Optional[pd.Timestamp]:
    data = load_daily_summary()
    if data.empty:
        return None
    return pd.Timestamp(data["Date"].max())


def macro_kcal(data: pd.DataFrame) -> pd.DataFrame:
    """Calories contributed by each macro (and alcohol) per day."""
    out = pd.DataFrame({"Date": data["Date"]})
    for macro, factor in KCAL_PER_GRAM.items():
        grams = pd.to_numeric(data.get(macro), errors="coerce") if macro in data.columns else float("nan")
        out[macro] = grams * factor
    return out
