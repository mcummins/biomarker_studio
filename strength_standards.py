"""
Static strength-standard helpers backed by a local JSON snapshot.

The data comes from the project's Google Sheet so the app can use the
standards without hitting a network dependency at runtime.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

STANDARD_LEVELS = ("Noob", "Beginner", "Intermediate", "Advanced", "Elite")
DEFAULT_GENDER = "Male"
DATA_PATH = os.path.join(os.path.dirname(__file__), "strength_standards.json")


@lru_cache(maxsize=1)
def _load_payload() -> Dict:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def has_data() -> bool:
    return os.path.exists(DATA_PATH)


def available_exercises() -> List[str]:
    if not has_data():
        return []
    payload = _load_payload()
    exercises = payload.get("exercises", {})
    return sorted(exercises.keys())


def _interpolate_rows(rows: Sequence[Dict], bodyweight_kg: float) -> Optional[Dict[str, float]]:
    if not rows:
        return None

    ordered = sorted(rows, key=lambda row: float(row["bodyweight_kg"]))
    if bodyweight_kg <= float(ordered[0]["bodyweight_kg"]):
        return {level: float(ordered[0][level]) for level in STANDARD_LEVELS}
    if bodyweight_kg >= float(ordered[-1]["bodyweight_kg"]):
        return {level: float(ordered[-1][level]) for level in STANDARD_LEVELS}

    for lower_row, upper_row in zip(ordered, ordered[1:]):
        lower_bw = float(lower_row["bodyweight_kg"])
        upper_bw = float(upper_row["bodyweight_kg"])
        if lower_bw <= bodyweight_kg <= upper_bw:
            if upper_bw == lower_bw:
                return {level: float(lower_row[level]) for level in STANDARD_LEVELS}
            ratio = (bodyweight_kg - lower_bw) / (upper_bw - lower_bw)
            return {
                level: float(lower_row[level]) + (float(upper_row[level]) - float(lower_row[level])) * ratio
                for level in STANDARD_LEVELS
            }

    return None


def get_thresholds(
    exercise_titles: Sequence[str],
    bodyweight_kg: float,
    gender: str = DEFAULT_GENDER,
) -> Optional[Dict[str, float]]:
    if not has_data() or not exercise_titles:
        return None

    payload = _load_payload()
    exercises = payload.get("exercises", {})
    per_exercise_thresholds: List[Dict[str, float]] = []

    for title in exercise_titles:
        exercise_payload = exercises.get(title)
        if not exercise_payload:
            continue
        gender_rows = exercise_payload.get(gender)
        if not gender_rows:
            continue
        interpolated = _interpolate_rows(gender_rows, bodyweight_kg)
        if interpolated:
            per_exercise_thresholds.append(interpolated)

    if not per_exercise_thresholds:
        return None

    # The merged bicep-curl view combines dumbbell and barbell entries, so we
    # average the standards from each contributing exercise to keep the category
    # bands aligned with the merged series.
    return {
        level: sum(item[level] for item in per_exercise_thresholds) / len(per_exercise_thresholds)
        for level in STANDARD_LEVELS
    }


def get_category_bands(thresholds: Optional[Dict[str, float]]) -> List[Dict[str, Optional[float]]]:
    if thresholds is None:
        return []

    bands: List[Dict[str, Optional[float]]] = []
    lower_bound = 0.0
    for index, level in enumerate(STANDARD_LEVELS):
        upper_bound = float(thresholds[level])
        bands.append(
            {
                "category": level,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )
        lower_bound = upper_bound

    bands.append(
        {
            "category": "Elite+",
            "lower_bound": float(thresholds["Elite"]),
            "upper_bound": None,
        }
    )
    return bands


def classify_1rm(estimated_1rm_kg: Optional[float], thresholds: Optional[Dict[str, float]]) -> Optional[Dict[str, object]]:
    if estimated_1rm_kg is None or thresholds is None:
        return None

    value = float(estimated_1rm_kg)
    bands = get_category_bands(thresholds)
    for index, band in enumerate(bands):
        upper_bound = band["upper_bound"]
        if upper_bound is None or value < upper_bound:
            next_category = bands[index + 1]["category"] if index + 1 < len(bands) else None
            return {
                "category": band["category"],
                "next_category": next_category,
                "kg_to_next": (upper_bound - value) if upper_bound is not None else None,
                "lower_bound": band["lower_bound"],
                "upper_bound": upper_bound,
            }

    return {
        "category": "Elite+",
        "next_category": None,
        "kg_to_next": None,
        "lower_bound": float(thresholds["Elite"]),
        "upper_bound": None,
    }
