
import os
import io
import json
import html
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px

import fitbit_client
import garmin_client
import hevy_client
import strength_standards

# -----------------------------
# Config
# -----------------------------
EXCLUDE_SHEETS = {"All Data", "Optimal Ranges", "Centiles", "Graphs", "Labs and notes", "NN Metabolic Scorecard", "Dexa"}
DEFAULT_GROUP_SHEETS = []  # will be filled dynamically
TIME_PRESETS = ["30 days", "90 days", "1 year", "All time", "Custom"]
LAST_PLOTLY_ZOOM_EVENT_KEY = "_last_plotly_zoom_event_id"
LIFT_SERIES_COLOR = "#7EB8DA"
STRENGTH_STANDARDS_GENDER = "Male"
STRENGTH_STANDARD_BAND_COLORS = {
    "Noob": "rgba(253, 249, 240, 0.24)",
    "Beginner": "rgba(241, 229, 196, 0.30)",
    "Intermediate": "rgba(210, 237, 191, 0.20)",
    "Advanced": "rgba(126, 200, 171, 0.24)",
    "Elite": "rgba(150, 206, 255, 0.20)",
    "Elite+": "rgba(169, 145, 255, 0.24)",
}
CENTILE_COLUMNS = [
    ("top_1", 1.00, "Top 1%"),
    ("top_5", 0.92, "Top 5%"),
    ("top_10", 0.84, "Top 10%"),
    ("top_20", 0.76, "Top 20%"),
    ("top_30", 0.68, "Top 30%"),
    ("average", 0.50, "Average"),
    ("bottom_30", 0.32, "Bottom 30%"),
    ("bottom_20", 0.24, "Bottom 20%"),
    ("bottom_10", 0.16, "Bottom 10%"),
    ("bottom_5", 0.08, "Bottom 5%"),
    ("bottom_1", 0.00, "Bottom 1%"),
]
CENTILE_METRICS_CATEGORY = "Centile metrics"

# -----------------------------
# US-typical unit conversions
# -----------------------------
# Maps a biomarker (matched on its lower-cased name, the same test_key used
# elsewhere) to its US-customary unit and the affine transform that takes the
# SI/metric value stored in the sheet to that unit:  us = si * factor + offset.
# The All Data, Optimal Ranges and Centiles tabs all store a given marker in the
# same SI unit, so the same transform is applied to measurements, reference
# bounds and centile boundaries alike — keeping derived status, z-scores and
# centile shading unchanged in meaning (all are invariant under a positive
# affine transform). Markers already reported in US-typical units (mg/dL
# apolipoproteins, ng/dL T3, mcg/dL DHEA-S/T4, ng/mL C-peptide/GH, mg/L hs-CRP,
# nmol/L SHBG/Lp(a), …) and assay-specific titres without a standard conversion
# are intentionally omitted and left untouched.
US_UNIT_CONVERSIONS = {
    # Glucose (mmol/L -> mg/dL)
    "fasting glucose": {"unit": "mg/dL", "factor": 18.0182},
    # HbA1c (IFCC mmol/mol -> NGSP/DCCT %) — affine, NGSP master equation
    "hba1c": {"unit": "%", "factor": 0.09148, "offset": 2.152},
    # Insulin (pmol/L -> μIU/mL)
    "fasting insulin": {"unit": "μIU/mL", "factor": 0.14399},
    # Cholesterol family (mmol/L -> mg/dL)
    "total cholesterol": {"unit": "mg/dL", "factor": 38.67},
    "hdl cholesterol": {"unit": "mg/dL", "factor": 38.67},
    "ldl cholesterol": {"unit": "mg/dL", "factor": 38.67},
    "vldl cholesterol": {"unit": "mg/dL", "factor": 38.67},
    "non-hdl cholesterol": {"unit": "mg/dL", "factor": 38.67},
    "direct ldl": {"unit": "mg/dL", "factor": 38.67},
    # Triglycerides (mmol/L -> mg/dL)
    "triglycerides": {"unit": "mg/dL", "factor": 88.57},
    # Renal / nitrogen
    "creatinine": {"unit": "mg/dL", "factor": 0.011312},     # umol/L -> mg/dL
    "bun": {"unit": "mg/dL", "factor": 2.801},               # mmol/L -> mg/dL (as N)
    "urea": {"unit": "mg/dL", "factor": 6.006},              # mmol/L -> mg/dL (urea molecule)
    "uric acid": {"unit": "mg/dL", "factor": 0.016812},      # umol/L -> mg/dL
    "urate": {"unit": "mg/dL", "factor": 0.016812},
    # Bilirubin (umol/L -> mg/dL)
    "total bilirubin": {"unit": "mg/dL", "factor": 0.058467},
    "direct bilirubin": {"unit": "mg/dL", "factor": 0.058467},
    "indirect bilirubin": {"unit": "mg/dL", "factor": 0.058467},
    # Minerals (mmol/L -> mg/dL)
    "calcium": {"unit": "mg/dL", "factor": 4.008},
    "magnesium": {"unit": "mg/dL", "factor": 2.4305},
    "phosphate": {"unit": "mg/dL", "factor": 3.0974},
    # Iron studies (umol/L -> μg/dL)
    "iron": {"unit": "μg/dL", "factor": 5.587},
    "total iron binding capacity (tibc)": {"unit": "μg/dL", "factor": 5.587},
    # Vitamins
    "vitamin d": {"unit": "ng/mL", "factor": 0.40064},       # nmol/L -> ng/mL
    "vitamin b12": {"unit": "pg/mL", "factor": 1.0},         # ng/L  -> pg/mL (identical)
    "folate (vitamin b9)": {"unit": "ng/mL", "factor": 1.0}, # ug/L  -> ng/mL (identical)
    # Hormones
    "total testosterone": {"unit": "ng/dL", "factor": 28.842},          # nmol/L -> ng/dL
    "free testosterone": {"unit": "pg/mL", "factor": 288.42},           # nmol/L -> pg/mL
    "oestradiol": {"unit": "pg/mL", "factor": 0.27240},                 # pmol/L -> pg/mL
    "free t4": {"unit": "ng/dL", "factor": 0.07769},                    # pmol/L -> ng/dL
    "free tri-iodothyronine (ft3)": {"unit": "pg/mL", "factor": 0.6510},# pmol/L -> pg/mL
    "parathyroid hormone (pth)": {"unit": "pg/mL", "factor": 9.434},    # pmol/L -> pg/mL
    "prolactin": {"unit": "ng/mL", "factor": 0.04717},                  # mIU/L  -> ng/mL
    # Proteins reported in g/dL in the US (g/L -> g/dL)
    "albumin": {"unit": "g/dL", "factor": 0.1},
    "total protein": {"unit": "g/dL", "factor": 0.1},
    "haemoglobin": {"unit": "g/dL", "factor": 0.1},
    "calculated globulin": {"unit": "g/dL", "factor": 0.1},
    # Proteins reported in mg/dL in the US (g/L -> mg/dL)
    "immunoglobulin a (iga)": {"unit": "mg/dL", "factor": 100.0},
    "immunoglobulin g (igg)": {"unit": "mg/dL", "factor": 100.0},
    "immunoglobulin m (igm)": {"unit": "mg/dL", "factor": 100.0},
    "complement component 3 (c3)": {"unit": "mg/dL", "factor": 100.0},
    "complement component 4 (c4)": {"unit": "mg/dL", "factor": 100.0},
    "transferrin": {"unit": "mg/dL", "factor": 100.0},
    # Haematocrit (ratio -> %)
    "haematocrit": {"unit": "%", "factor": 100.0},
    # Markers reported in ng/mL in the US (numerically identical to ug/L / μg/l)
    "ferritin": {"unit": "ng/mL", "factor": 1.0},
    "leptin": {"unit": "ng/mL", "factor": 1.0},
    "total prostate specific antigen (tpsa)": {"unit": "ng/mL", "factor": 1.0},
    # Albumin/Creatinine ratio (mg/mmol -> mg/g creatinine)
    "microalbumin / creatinine ratio": {"unit": "mg/g creatinine", "factor": 8.8402},
    # Monovalent electrolytes (mmol/L -> mEq/L, numerically identical)
    "sodium": {"unit": "mEq/L", "factor": 1.0},
    "potassium": {"unit": "mEq/L", "factor": 1.0},
    "chloride": {"unit": "mEq/L", "factor": 1.0},
    "bicarbonate": {"unit": "mEq/L", "factor": 1.0},
    # Thyroid / pituitary titres (numerically identical relabels)
    "tsh": {"unit": "μIU/mL", "factor": 1.0},                          # mIU/L -> μIU/mL
    "follicle stimulating hormone": {"unit": "mIU/mL", "factor": 1.0}, # IU/L  -> mIU/mL
    "luteinising hormone": {"unit": "mIU/mL", "factor": 1.0},
}

# Warm-theme gradient anchors for centile shading: 0 = worst centile, 1 = best.
CENTILE_GRADIENT = [
    (0.00, "#E07A5F"),
    (0.20, "#F1B07B"),
    (0.40, "#F2CC8F"),
    (0.60, "#BED8C7"),
    (0.80, "#81B29A"),
    (1.00, "#4F8F73"),
]
DEXA_CENTILE_TEST_MAP = {
    "Tissue %Fat": "Fat Percentage",
}
LIFTS_PAGE_CONFIG = [
    {"label": "Bench", "source_titles": ["Bench Press (Barbell)"]},
    {"label": "Squat", "source_titles": ["Squat (Barbell)", "Zercher Squat"]},
    {"label": "Deadlift", "source_titles": ["Deadlift (Barbell)", "Deadlift (Trap Bar)"]},
    {"label": "Overhead Press", "source_titles": ["Overhead Press (Barbell)"]},
    {"label": "Bicep Curl", "source_titles": ["Bicep Curl (Dumbbell)", "Bicep Curl (Barbell)"]},
]
PLOTLY_ZOOM_SYNC = components.declare_component(
    "plotly_zoom_sync",
    path=str(Path(__file__).parent / "streamlit_components" / "plotly_zoom_sync"),
)

# -----------------------------
# Utility functions
# -----------------------------
def parse_spreadsheet_id(url_or_id: str) -> str:
    """
    Accepts a Google Sheets URL or a plain spreadsheet ID and returns the ID.
    """
    if "docs.google.com" in url_or_id:
        # URL format: https://docs.google.com/spreadsheets/d/<id>/edit...
        parts = url_or_id.split("/")
        if "spreadsheets" in parts and "d" in parts:
            try:
                i = parts.index("d")
                return parts[i+1]
            except Exception:
                pass
    return url_or_id.strip()

def detect_date_cols(columns: List) -> List:
    date_cols = []
    for c in columns:
        # Skip obvious non-dates
        if isinstance(c, str) and c.lower() in ("test","unit","notes","instalab values","lab","full details"):
            continue
        try:
            pd.to_datetime([c])
            if isinstance(c, str) and any(w in c.lower() for w in ["note","instalab"]):
                continue
            date_cols.append(c)
        except Exception:
            pass
    return date_cols

def parse_value(raw) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    if pd.isna(raw):
        return None, None, None
    s = str(raw).strip()
    # Handle "<" or ">" prefixed values like "<9.00" or "<15"
    m = re.match(r'^([<>]=?)\s*([0-9]*\.?[0-9]+)$', s)
    if m:
        q = m.group(1)
        try:
            v = float(m.group(2))
        except:
            v = None
        return v, q, s
    # Handle numbers with commas
    s2 = s.replace(',', '')
    try:
        v = float(s2)
        return v, None, s
    except:
        # Handle ranges like "4-9"
        m2 = re.match(r'^([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)$', s2)
        if m2:
            v = (float(m2.group(1)) + float(m2.group(2))) / 2.0
            return v, 'range', s
        # non numeric, return None but keep raw
        return None, None, s


def format_lab_number(value) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 0.1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def latest_test_summary(df: pd.DataFrame, test: str) -> str:
    data = df[df["test"] == test].sort_values("Date")
    if data.empty:
        return ""
    latest = data.iloc[-1]
    value = format_lab_number(latest["Value"])
    unit = str(latest.get("unit", "")).strip()
    return f"{value} {unit}".strip()

def normalize_all_data(df: pd.DataFrame) -> pd.DataFrame:
    # Detect date columns dynamically
    date_cols = detect_date_cols(df.columns)
    core_cols = [c for c in ("Test","Unit") if c in df.columns]
    long = df.melt(id_vars=core_cols, value_vars=date_cols,
                   var_name="Date", value_name="RawResult")
    long["Date"] = pd.to_datetime(long["Date"], errors="coerce")
    long = long.dropna(subset=["Date"])
    parsed = long["RawResult"].apply(parse_value)
    long["Value"] = parsed.apply(lambda x: x[0])
    long["Qualifier"] = parsed.apply(lambda x: x[1])
    long["Raw"] = parsed.apply(lambda x: x[2])
    long = long.dropna(subset=["Value"])
    long = long.rename(columns={"Test":"test","Unit":"unit"})
    if "test" in long.columns:
        long["test"] = long["test"].astype(str).str.strip()
    if "unit" in long.columns:
        long["unit"] = long["unit"].astype(str).str.strip().replace("", np.nan)
    return long


def _empty_dexa_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_order", "category", "region", "metric", "test", "unit", "Date",
            "Value", "Qualifier", "Raw", "lower", "upper", "borderline",
        ]
    )


def _is_probable_date(value) -> bool:
    if pd.isna(value):
        return False
    try:
        return pd.notna(pd.to_datetime(value, errors="coerce"))
    except Exception:
        return False


def format_dexa_test_name(region: str, metric: str) -> str:
    region = str(region or "").strip()
    metric = str(metric or "").strip()
    if not region or region.lower() == "whole body":
        return metric
    return f"{region} - {metric}"


def get_dexa_health_bounds(metric: str, region: str, unit: str = "") -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return lower, upper, borderline bounds for commonly recognized DEXA cutoffs."""
    metric_key = str(metric or "").strip().lower()
    region_key = str(region or "").strip().lower()
    unit_key = str(unit or "").strip().lower()

    if metric_key == "tissue %fat" and region_key == "whole body":
        # Adult male body-fat categories commonly put 25%+ in the obese range.
        return 6.0, 25.0, 24.0
    if metric_key == "vat area" and unit_key in {"cm^2", "cm²"}:
        # DEXA VAT area: <100 cm^2 is commonly used as lower cardiometabolic risk.
        return None, 160.0, 100.0
    if metric_key == "relative skeletal muscle index":
        # Common low-muscle-mass cutoff for men is roughly 7.0 kg/m^2.
        return 7.0, None, None
    if metric_key == "t-score":
        # WHO bone-density interpretation: normal >= -1, osteoporosis <= -2.5.
        return -2.5, None, -1.0
    if metric_key == "z-score":
        # Z-score below -2.0 is commonly described as below expected range for age.
        return -2.0, None, None
    if metric_key == "android/gynoid ratio":
        # A/G ratio above 1.0 is a common central-adiposity risk signal in men.
        return None, 1.0, None
    return None, None, None


def dexa_health_status(value, lower, upper, borderline=None) -> str:
    if pd.isna(value):
        return "neutral"
    has_lower = pd.notna(lower)
    has_upper = pd.notna(upper)
    has_borderline = pd.notna(borderline)
    if not has_lower and not has_upper:
        return "neutral"
    if has_lower and value < lower:
        return "alert"
    if has_upper and value > upper:
        return "alert"
    if has_borderline:
        if has_upper and borderline < upper and value > borderline:
            return "caution"
        if has_lower and borderline > lower and value < borderline:
            return "caution"
    return "good"


def normalize_dexa_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Dexa sheet, whose scan dates live in the Metadata/Date row."""
    if df is None or df.empty:
        return _empty_dexa_frame()

    raw = df.copy()
    headers = [str(col).strip() for col in raw.columns]
    lower_headers = [h.lower() for h in headers]

    def header_index(name: str) -> Optional[int]:
        try:
            return lower_headers.index(name.lower())
        except ValueError:
            return None

    category_idx = header_index("Category")
    region_idx = header_index("Region")
    metric_idx = header_index("Metric")
    unit_idx = header_index("Unit")
    if None in (category_idx, region_idx, metric_idx, unit_idx):
        return _empty_dexa_frame()

    change_indices = [i for i, h in enumerate(lower_headers) if h.startswith("change")]
    first_change_idx = min(change_indices) if change_indices else len(headers)
    measurement_indices = list(range(unit_idx + 1, first_change_idx))

    date_row = None
    metric_values = raw.iloc[:, metric_idx].astype(str).str.strip().str.lower()
    date_matches = raw[metric_values == "date"]
    if not date_matches.empty:
        date_row = date_matches.iloc[0]

    dated_columns = []
    for idx in measurement_indices:
        date_raw = date_row.iloc[idx] if date_row is not None and idx < len(date_row) else headers[idx]
        if not _is_probable_date(date_raw):
            continue
        dated_columns.append((idx, pd.to_datetime(date_raw, errors="coerce")))

    rows = []
    for row_order, row in raw.iterrows():
        category = str(row.iloc[category_idx] if category_idx < len(row) else "").strip()
        region = str(row.iloc[region_idx] if region_idx < len(row) else "").strip()
        metric = str(row.iloc[metric_idx] if metric_idx < len(row) else "").strip()
        unit = str(row.iloc[unit_idx] if unit_idx < len(row) else "").strip()
        if not category or not metric or metric.lower() == "date":
            continue
        if category.lower() == "metadata":
            continue

        test = format_dexa_test_name(region, metric)
        lower, upper, borderline = get_dexa_health_bounds(metric, region, unit)
        for idx, scan_date in dated_columns:
            parsed_value, qualifier, raw_value = parse_value(row.iloc[idx] if idx < len(row) else np.nan)
            if parsed_value is None or pd.isna(scan_date):
                continue
            rows.append(
                {
                    "row_order": int(row_order),
                    "category": category,
                    "region": region,
                    "metric": metric,
                    "test": test,
                    "unit": unit,
                    "Date": pd.to_datetime(scan_date),
                    "Value": parsed_value,
                    "Qualifier": qualifier,
                    "Raw": raw_value,
                    "lower": lower,
                    "upper": upper,
                    "borderline": borderline,
                }
            )

    if not rows:
        return _empty_dexa_frame()

    long = pd.DataFrame(rows)
    long["status"] = long.apply(
        lambda r: dexa_health_status(r["Value"], r.get("lower"), r.get("upper"), r.get("borderline")),
        axis=1,
    )
    return long


def normalize_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    df2 = df2.rename(columns={"Test Instalab Values": "Test"})
    # Keep essential columns if present
    keep = [c for c in ["Test","Unit","Optimal Range (lower)","Optimal Range (borderline)","Optimal Range (upper)"] if c in df2.columns]
    df2 = df2[keep]
    if "Test" not in df2.columns:
        return pd.DataFrame(columns=["test", "unit", "lower", "upper", "borderline"])
    df2 = df2[df2["Test"].notna()]
    df2 = df2[df2["Test"].astype(str).str.lower()!="instalab values"]
    # Coerce numeric
    for c in ["Optimal Range (lower)","Optimal Range (upper)","Optimal Range (borderline)"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.rename(columns={
        "Test":"test","Unit":"unit",
        "Optimal Range (lower)":"lower",
        "Optimal Range (upper)":"upper",
        "Optimal Range (borderline)":"borderline"
    })
    if "test" in df2.columns:
        df2["test"] = df2["test"].astype(str).str.strip()
    if "unit" in df2.columns:
        df2["unit"] = df2["unit"].astype(str).str.strip().replace("", np.nan)
    return df2


def normalize_centiles(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    required = {"biomarker", "side", "direction"}
    if not required.issubset(df2.columns):
        return pd.DataFrame()

    df2["biomarker"] = df2["biomarker"].astype(str).str.strip()
    df2 = df2[df2["biomarker"].ne("")]
    df2["test_key"] = df2["biomarker"].str.lower()
    df2["side"] = df2["side"].astype(str).str.strip().str.lower().replace("", "main")
    df2["direction"] = df2["direction"].astype(str).str.strip().str.lower()

    for col, _score, _label in CENTILE_COLUMNS:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
        else:
            df2[col] = np.nan

    return df2


def attach_ranges(long: pd.DataFrame, ranges: pd.DataFrame, policy: str = "union") -> pd.DataFrame:
    r2 = ranges.copy()
    l2 = long.copy()
    l2["test_key"] = l2["test"].astype(str).str.strip().str.lower()
    r2["test_key"] = r2["test"].astype(str).str.strip().str.lower()

    def first_valid(s):
        for v in s:
            if pd.notna(v) and v != "":
                return v
        return np.nan

    def numeric_bound(s, reducer):
        values = pd.to_numeric(s, errors="coerce").dropna()
        if values.empty:
            return np.nan
        return reducer(values)

    if policy == "intersection":
        lower_agg = lambda s: numeric_bound(s, np.max)
        upper_agg = lambda s: numeric_bound(s, np.min)
    else:  # "union" (default)
        lower_agg = lambda s: numeric_bound(s, np.min)
        upper_agg = lambda s: numeric_bound(s, np.max)

    # One canonical row per test_key
    rcanon = (
        r2.groupby("test_key", as_index=False)
          .agg({
              "lower": lower_agg,
              "upper": upper_agg,
              "borderline": first_valid,
              "unit": first_valid,  # keep a representative unit if present
          })
    )

    merged = l2.merge(rcanon, on="test_key", how="left", suffixes=("", "_rng")).drop(columns=["test_key"])
    # keep the measurement unit; drop any extra unit col from ranges
    if "unit_rng" in merged.columns:
        merged = merged.drop(columns=["unit_rng"])
    return merged

def attach_lab_notes(long: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    df = notes.copy()
    if "Sample date" in df.columns:
        df["Sample date"] = pd.to_datetime(df["Sample date"], errors="coerce")
        df = df.rename(columns={"Sample date":"Date"})
        keep = [c for c in ["Date","Lab","Notes"] if c in df.columns]
        df = df[keep]
        return long.merge(df, on="Date", how="left")
    return long


def convert_units_to_us(
    merged: pd.DataFrame,
    centiles: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Re-express measurements, reference bounds and centiles in US-typical units.

    For every biomarker listed in ``US_UNIT_CONVERSIONS`` the stored SI value is
    mapped through ``us = si * factor + offset`` and the unit label is replaced.
    The identical transform is applied to the measurement columns, the optimal
    range bounds and each centile boundary, so status, z-scores and centile
    shading (all invariant under a positive affine transform) are unchanged.
    Markers not in the mapping are passed through untouched.
    """
    merged = merged.copy()
    keys = merged["test"].astype(str).str.strip().str.lower()
    value_cols = [c for c in ("Value", "lower", "upper", "borderline") if c in merged.columns]
    for key, conv in US_UNIT_CONVERSIONS.items():
        mask = keys == key
        if not mask.any():
            continue
        factor = conv["factor"]
        offset = conv.get("offset", 0.0)
        for col in value_cols:
            merged.loc[mask, col] = merged.loc[mask, col] * factor + offset
        if "unit" in merged.columns:
            merged.loc[mask, "unit"] = conv["unit"]

    if centiles is not None and not centiles.empty and "test_key" in centiles.columns:
        centiles = centiles.copy()
        centile_value_cols = [col for col, _score, _label in CENTILE_COLUMNS if col in centiles.columns]
        for key, conv in US_UNIT_CONVERSIONS.items():
            mask = centiles["test_key"] == key
            if not mask.any():
                continue
            factor = conv["factor"]
            offset = conv.get("offset", 0.0)
            for col in centile_value_cols:
                centiles.loc[mask, col] = centiles.loc[mask, col] * factor + offset
            if "unit" in centiles.columns:
                centiles.loc[mask, "unit"] = conv["unit"]

    return merged, centiles


def status_from_bounds(value, lower, upper) -> str:
    if pd.notna(lower) and value < lower:
        return "low"
    if pd.notna(upper) and value > upper:
        return "high"
    if pd.notna(lower) or pd.notna(upper):
        return "normal"
    return "unknown"

def highlight_status(status):
    """Return background-color CSS for a full row based on status."""
    if status == "normal":
        return "background-color: rgba(129, 178, 154, 0.3)"  # sage green
    if status in ("high but improved", "low but improved"):
        return "background-color: rgba(242, 204, 143, 0.35)"  # soft amber
    if status in ("high", "low"):
        return "background-color: rgba(224, 122, 95, 0.3)"  # soft rose
    return ""

def compute_zscore(value, lower, upper) -> Optional[float]:
    if pd.notna(lower) and pd.notna(upper):
        mid = (lower + upper)/2.0
        half_width = (upper - lower)/2.0
        if half_width and half_width != 0:
            return (value - mid) / half_width
    return None


def compute_heatmap_health_score(value, lower, upper, borderline=None) -> Optional[float]:
    """Return a score in [-1, 1] for heatmap coloring.

    Negative values are out-of-range (red).
    Values near 0 are borderline (yellow).
    Positive values are safely in-range (green).
    """
    if pd.isna(value):
        return None

    has_lower = pd.notna(lower)
    has_upper = pd.notna(upper)
    if not has_lower and not has_upper:
        return None

    if has_lower and has_upper:
        span = upper - lower
        if span <= 0:
            return None
        scale = max(span / 2.0, 1e-9)
        if lower <= value <= upper:
            if pd.notna(borderline) and lower < borderline < upper:
                if value <= borderline:
                    optimal_span = max(borderline - lower, 1e-9)
                    return float(0.25 + 0.75 * min((value - lower) / optimal_span, 1.0))
                borderline_span = max(upper - borderline, 1e-9)
                # Borderline stays near yellow, trending toward red as it nears upper.
                return float(0.15 * max(1.0 - ((value - borderline) / borderline_span), 0.0))
            distance = min(value - lower, upper - value)
            return float(min(distance / scale, 1.0))
        if value < lower:
            return float(-min((lower - value) / scale, 1.0))
        return float(-min((value - upper) / scale, 1.0))

    if has_lower:
        scale = max(abs(lower), 1.0)
        if value >= lower:
            if pd.notna(borderline) and borderline > lower:
                if value < borderline:
                    borderline_span = max(borderline - lower, 1e-9)
                    return float(0.15 * min((value - lower) / borderline_span, 1.0))
                optimal_span = max(scale, 1e-9)
                return float(0.25 + 0.75 * min((value - borderline) / optimal_span, 1.0))
            return float(min((value - lower) / scale, 1.0))
        return float(-min((lower - value) / scale, 1.0))

    scale = max(abs(upper), 1.0)
    if value <= upper:
        if pd.notna(borderline) and borderline < upper:
            if value <= borderline:
                optimal_span = max(scale, 1e-9)
                return float(0.25 + 0.75 * min((borderline - value) / optimal_span, 1.0))
            borderline_span = max(upper - borderline, 1e-9)
            return float(0.15 * max(1.0 - ((value - borderline) / borderline_span), 0.0))
        return float(min((upper - value) / scale, 1.0))
    return float(-min((value - upper) / scale, 1.0))

def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["test","Date"])
    df["PrevValue"] = df.groupby("test")["Value"].shift(1)
    df["DeltaAbs"] = df["Value"] - df["PrevValue"]
    df["DeltaPct"] = df["DeltaAbs"] / df["PrevValue"] * 100.0
    return df

def compute_trends(df: pd.DataFrame) -> pd.DataFrame:
    # simple linear regression using numpy polyfit
    rows = []
    for test, g in df.groupby("test"):
        g = g.dropna(subset=["Value"]).sort_values("Date")
        if len(g) < 3:
            continue
        x = (g["Date"] - g["Date"].min()).dt.days.values.astype(float)
        y = g["Value"].values.astype(float)
        # polyfit
        slope, intercept = np.polyfit(x, y, 1)
        # R^2
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
        rows.append({
            "test": test,
            "slope_per_year": float(slope * 365.25),
            "r2": float(r2),
            "n": int(len(g))
        })
    return pd.DataFrame(rows).sort_values("r2", ascending=False)

def load_groups_from_sheets(all_sheets: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    groups = {}
    for name, df in all_sheets.items():
        if name in EXCLUDE_SHEETS:
            continue
        # Read the test list using a normalized header match so sheets with
        # minor header inconsistencies still appear in the category list.
        col_lookup = {str(col).strip().lower(): col for col in df.columns}
        test_col = col_lookup.get("test")
        # Fall back to the first column if "test" header is missing/blank
        use_iloc = False
        if test_col is None and len(df.columns) > 0 and str(df.columns[0]).strip() == "":
            use_iloc = True
        if test_col is not None or use_iloc:
            col_data = df.iloc[:, 0] if use_iloc else df[test_col]
            tests = col_data.dropna().astype(str).str.strip()
            tests = [t for t in tests.unique().tolist() if t]
            if len(tests) > 0:
                groups[name] = tests
    return groups

# -----------------------------
# Data loaders
# -----------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_from_gsheets(spreadsheet_id: str, service_account_info: dict) -> Dict[str, pd.DataFrame]:
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    client = gspread.authorize(creds)

    sh = client.open_by_key(spreadsheet_id)

    sheets = {}
    for ws in sh.worksheets():
        rows = ws.get_all_values()
        if not rows:
            continue
        df = pd.DataFrame(rows[1:], columns=rows[0])
        # try to convert date-like column headers back to proper datetime strings where appropriate
        sheets[ws.title] = df
    return sheets

@st.cache_data(show_spinner=False, ttl=0)
def load_from_xlsx(uploaded_file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(uploaded_file)
    return {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}

# -----------------------------
# Visualization helpers
# -----------------------------
def _pick_bound(vals, kind="lower", policy="union"):
    """
    vals: array-like of candidate bounds (possibly from multiple labs)
    kind: "lower" or "upper"
    policy:
      - "union": lower = min(all lowers); upper = max(all uppers)
      - "intersection": lower = max(all lowers); upper = min(all uppers)
      - "first": first non-null value
    """
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if v.empty:
        return None
    if policy == "union":
        return float(v.min() if kind == "lower" else v.max())
    if policy == "intersection":
        return float(v.max() if kind == "lower" else v.min())
    return float(v.iloc[0])


def apply_warm_theme(fig: go.Figure) -> go.Figure:
    """Apply a consistent warm theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family="Manrope, sans-serif", color="#18322f"),
        hoverlabel=dict(
            bgcolor="#18322f",
            font_color="#FFFFFF",
            font_size=13,
            font_family="Manrope, sans-serif",
            bordercolor="#18322f",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(24, 50, 47, 0.08)", linecolor="rgba(24, 50, 47, 0.14)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(24, 50, 47, 0.08)", linecolor="rgba(24, 50, 47, 0.14)", zeroline=False)
    return fig


def centile_band_color(score: float) -> str:
    score = min(1.0, max(0.0, float(score)))
    for (s0, c0), (s1, c1) in zip(CENTILE_GRADIENT, CENTILE_GRADIENT[1:]):
        if score <= s1:
            t = 0.0 if s1 == s0 else (score - s0) / (s1 - s0)
            rgb0 = tuple(int(c0[i:i+2], 16) for i in (1, 3, 5))
            rgb1 = tuple(int(c1[i:i+2], 16) for i in (1, 3, 5))
            r, g, b = (round(a + t * (b_ - a)) for a, b_ in zip(rgb0, rgb1))
            return f"#{r:02X}{g:02X}{b:02X}"
    return CENTILE_GRADIENT[-1][1]


def centile_points(row: pd.Series) -> List[Tuple[float, float, str]]:
    """Return (value, score, label) per defined centile, sorted by value."""
    points = []
    for col, score, label in CENTILE_COLUMNS:
        value = row.get(col)
        if pd.notna(value):
            points.append((float(value), score, label))
    points.sort(key=lambda item: item[0])

    deduped = []
    for value, score, label in points:
        if deduped and abs(value - deduped[-1][0]) < 1e-12:
            if score > deduped[-1][1]:
                deduped[-1] = (value, score, label)
        else:
            deduped.append((value, score, label))
    return deduped


def centile_score_at(points: List[Tuple[float, float, str]], value: float) -> float:
    """Interpolate a centile score at `value` from one side's points.

    Beyond the worst defined centile the score clamps to that centile; beyond
    the best it clamps to 1.0, since blank good-end columns (e.g. the high side
    of HDL) mean "no penalty in this direction", not "top 20% at best".
    """
    best_first = points[0][1] >= points[-1][1]
    if value <= points[0][0]:
        return 1.0 if best_first else points[0][1]
    if value >= points[-1][0]:
        return points[-1][1] if best_first else 1.0
    for (v0, s0, _l0), (v1, s1, _l1) in zip(points, points[1:]):
        if v0 <= value <= v1:
            t = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
            return s0 + t * (s1 - s0)
    return points[-1][1]


def get_centile_rows(centiles: Optional[pd.DataFrame], test: str) -> pd.DataFrame:
    if centiles is None or centiles.empty or "test_key" not in centiles.columns:
        return pd.DataFrame()
    test_key = str(test).strip().lower()
    return centiles[centiles["test_key"] == test_key].copy()


def get_centile_metric_names(centiles: Optional[pd.DataFrame], available_tests: List[str]) -> List[str]:
    if centiles is None or centiles.empty or "biomarker" not in centiles.columns:
        return []

    available = {str(test).strip() for test in available_tests}
    names = []
    seen = set()
    for biomarker in centiles["biomarker"].dropna().astype(str).str.strip():
        if biomarker in available and biomarker not in seen:
            names.append(biomarker)
            seen.add(biomarker)
    return names


def render_us_units_toggle(scope: str) -> bool:
    """Sidebar toggle that switches the page into US-typical units.

    Uses the same shadow-key pattern as the biohacker controls so the choice
    survives page switches, and is read early in the page (before the widget is
    rendered) via ``st.session_state.get(f"{scope}_us_units_on", False)``.
    """
    persist_key = f"{scope}_us_units_on"
    us_units = st.sidebar.toggle(
        "US units",
        value=st.session_state.get(persist_key, False),
        key=f"{scope}_us_units",
        help="Display values in US-customary units (mg/dL, %, μIU/mL, …) instead of SI/metric units.",
    )
    st.session_state[persist_key] = us_units
    return us_units


def render_background_view_controls(scope: str) -> str:
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    render_us_units_toggle(scope)
    # Shadow key keeps the choice alive across page switches, since Streamlit
    # drops widget state for pages that don't render the widget.
    persist_key = f"{scope}_biohacker_on"
    biohacker = st.sidebar.toggle(
        "Biohacker mode",
        value=st.session_state.get(persist_key, False),
        key=f"{scope}_biohacker_mode",
        help="Shade graph backgrounds by population centiles instead of healthy ranges.",
    )
    st.session_state[persist_key] = biohacker
    if not biohacker:
        return "Standard"

    enhanced_persist_key = f"{scope}_biohacker_enhanced_on"
    enhanced = st.sidebar.toggle(
        "Enhanced view",
        value=st.session_state.get(enhanced_persist_key, False),
        key=f"{scope}_biohacker_enhanced",
        help="Zoom the axis to your data instead of the full centile range, making small centile shifts easier to see.",
    )
    st.session_state[enhanced_persist_key] = enhanced
    return "Biohacker Enhanced" if enhanced else "Biohacker"


def add_centile_zones(
    fig: go.Figure,
    y: pd.Series,
    centile_rows: pd.DataFrame,
    x0,
    x1,
    fit_to_data: bool = False,
) -> bool:
    point_sets = [centile_points(row) for _, row in centile_rows.iterrows()]
    point_sets = [points for points in point_sets if points]
    if not point_sets:
        return False

    # Merge boundaries from all sides (e.g. low_side + high_side rows) into one
    # set of bands; each band is scored at its midpoint as the worst centile
    # across sides, so two-sided metrics shade green in the middle and degrade
    # toward both extremes.
    boundaries = []
    for points in point_sets:
        for value, _score, label in points:
            if not any(abs(value - b) < 1e-9 * max(1.0, abs(value)) for b, _ in boundaries):
                boundaries.append((value, label))
    boundaries.sort(key=lambda item: item[0])

    # An open-ended extreme is encoded as 0 in the sheet (e.g. "bottom 1% = 0"
    # for Albumin, or "top 1% = 0" for lower-is-better metrics like ApoB).
    # Keep such a sentinel off the axis when it sits beyond the data and is
    # separated from its neighbour by a gap far larger than the typical
    # centile gap (which spares genuine zeros like hs-CRP's); the outermost
    # band still shades with that centile's colour.
    values = [value for value, _label in boundaries]
    if len(values) >= 3:
        gaps = [b - a for a, b in zip(values, values[1:])]
        typical_gap = float(np.median(gaps))
        if (
            typical_gap > 0
            and values[0] <= 0
            and values[1] - values[0] > 3 * typical_gap
            and values[0] < float(y.min())
        ):
            boundaries = boundaries[1:]

    y_candidates = [y.min(), y.max()]
    if not fit_to_data:
        y_candidates.extend(value for value, _label in boundaries)
    y_min, y_max = min(y_candidates), max(y_candidates)
    span = (y_max - y_min) or 1.0
    pad = 0.12 if fit_to_data else 0.05
    y_min -= pad * span
    y_max += pad * span
    fig.update_yaxes(range=[y_min, y_max])

    edges = [y_min] + [value for value, _label in boundaries if y_min < value < y_max] + [y_max]
    for lo, hi in zip(edges, edges[1:]):
        if hi <= lo:
            continue
        mid = (lo + hi) / 2.0
        score = min(centile_score_at(points, mid) for points in point_sets)
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=lo, y1=hi,
            fillcolor=centile_band_color(score), opacity=0.22, line_width=0, layer="below"
        )

    # Label the transitions on a right-hand axis, skipping ticks that would
    # collide with the previous one.
    min_sep = 0.035 * (y_max - y_min)
    tick_vals, tick_text = [], []
    for value, label in boundaries:
        if y_min < value < y_max and (not tick_vals or value - tick_vals[-1] >= min_sep):
            tick_vals.append(value)
            tick_text.append(label)
    if tick_vals:
        fig.add_trace(go.Scatter(
            x=[x0], y=[tick_vals[0]], yaxis="y2", mode="markers",
            marker=dict(opacity=0), showlegend=False, hoverinfo="skip",
        ))
        fig.update_layout(yaxis2=dict(
            overlaying="y", side="right", matches="y",
            showgrid=False, zeroline=False, automargin=True,
            tickvals=tick_vals, ticktext=tick_text,
            ticks="outside", ticklen=3, tickcolor="rgba(24, 50, 47, 0.25)",
            tickfont=dict(size=9, color="#4F5D5A"),
        ))

    return True


def plot_single_test(df: pd.DataFrame, test: str,
                     show_ref: bool=True, show_regression: bool=False,
                     show_zones: bool=True, range_policy: str="union",
                     date_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                     background_view: str="Standard",
                     centiles: Optional[pd.DataFrame]=None,
                     centile_test: Optional[str]=None) -> go.Figure:
    g = df[df["test"] == test].sort_values("Date")
    fig = go.Figure()
    if g.empty:
        return fig

    x = g["Date"]
    y = g["Value"]

    # ---- pick a single set of bounds (handles multiple lab rows) ----
    low_vals = g["lower"].dropna().unique() if "lower" in g.columns else []
    up_vals  = g["upper"].dropna().unique() if "upper" in g.columns else []
    bor_vals = g["borderline"].dropna().unique() if "borderline" in g.columns else []

    lower = _pick_bound(low_vals,  "lower", range_policy) if len(low_vals) else None
    upper = _pick_bound(up_vals,   "upper", range_policy) if len(up_vals) else None
    borderline = float(bor_vals[0]) if len(bor_vals) else None

    # make sure lower <= upper if both present (swap if accidental reverse)
    if pd.notna(lower) and pd.notna(upper) and lower > upper:
        lower, upper = upper, lower

    if date_window is not None:
        x0 = pd.to_datetime(date_window[0])
        x1 = pd.to_datetime(date_window[1])
    else:
        x0, x1 = x.min(), x.max()

    # ---- background zones (green/orange/red) ----
    centile_rows = get_centile_rows(centiles, centile_test or test)
    use_centile_zones = (
        show_zones
        and background_view in ("Biohacker", "Biohacker Enhanced")
        and not centile_rows.empty
        and add_centile_zones(
            fig, y, centile_rows, x0, x1,
            fit_to_data=(background_view == "Biohacker Enhanced"),
        )
    )

    if show_zones and not use_centile_zones and (pd.notna(lower) or pd.notna(upper)):
        # compute a sensible y-range so the shading fully covers the plot
        y_candidates = [y.min(), y.max()]
        for v in (lower, upper, borderline):
            if v is not None and not pd.isna(v):
                y_candidates.append(v)
        y_min, y_max = min(y_candidates), max(y_candidates)
        span = (y_max - y_min) or 1.0
        y_min -= 0.05 * span
        y_max += 0.05 * span
        fig.update_yaxes(range=[y_min, y_max])

        def rect(y0, y1, color_hex):
            # draw behind series
            fig.add_shape(
                type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1,
                fillcolor=color_hex, opacity=0.22, line_width=0, layer="below"
            )

        # warm palette zone colors
        GREEN = "#81B29A"; ORANGE = "#F2CC8F"; RED = "#E07A5F"

        if pd.notna(lower) and pd.notna(upper):
            if pd.notna(borderline) and lower < borderline < upper:
                rect(y_min,     lower,     RED)    # below lower
                rect(lower,     borderline, GREEN)  # optimal
                rect(borderline, upper,     ORANGE) # borderline
                rect(upper,     y_max,     RED)    # above upper
            else:
                rect(y_min,     lower,     RED)
                rect(lower,     upper,     GREEN)
                rect(upper,     y_max,     RED)
        elif pd.notna(upper):  # only upper bound
            if pd.notna(borderline) and borderline < upper:
                rect(y_min,     borderline, GREEN)
                rect(borderline, upper,     ORANGE)
            else:
                rect(y_min,     upper,     GREEN)
            rect(upper,     y_max,     RED)
        elif pd.notna(lower):  # only lower bound
            rect(y_min,     lower,     RED)
            if pd.notna(borderline) and borderline > lower:
                rect(lower,     borderline, ORANGE)
                rect(borderline, y_max,     GREEN)
            else:
                rect(lower,     y_max,     GREEN)

    # ---- main series & optional trend ----
    lab_exists = "Lab" in g.columns
    notes_exists = "Notes" in g.columns
    def _hover_row(r, lab_exists=lab_exists, notes_exists=notes_exists):
        dt = pd.to_datetime(r.get("Date")).date() if pd.notna(r.get("Date")) else ""
        value_part = f"{r.get('Value')}"
        if pd.notna(r.get("unit")):
            value_part += f" {r.get('unit')}"
        if pd.notna(r.get("Qualifier")):
            value_part += f" ({r.get('Qualifier')})"
        parts = [f"{r.get('test','')}", f"{dt}", value_part]
        if lab_exists and pd.notna(r.get("Lab")):
            parts.append(f"Lab: {r.get('Lab')}")
        if notes_exists and pd.notna(r.get("Notes")):
            parts.append(f"Notes: {r.get('Notes')}")
        return "<br>".join(parts)

    hover = g.apply(_hover_row, axis=1)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name=test, hoverinfo="text", text=hover,
        line=dict(color="#E07A5F", width=2.5),
        marker=dict(size=7, color="#E07A5F", line=dict(width=1, color="#FFFFFF")),
    ))

    if show_regression and len(g) >= 3:
        days = (g["Date"] - g["Date"].min()).dt.days.values.astype(float)
        slope, intercept = np.polyfit(days, y.values.astype(float), 1)
        x_line = pd.date_range(start=g["Date"].min(), end=g["Date"].max(), periods=50)
        x_days = (x_line - g["Date"].min()).days.values.astype(float)
        y_line = slope * x_days + intercept
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend",
                                 line=dict(dash="dash", color="#D4C5B5", width=2)))

    # Optional dotted lines exactly at bounds
    if show_ref and not use_centile_zones:
        if pd.notna(lower):     fig.add_hline(y=lower,     line_dash="dot")
        if pd.notna(upper):     fig.add_hline(y=upper,     line_dash="dot")
        if pd.notna(borderline):fig.add_hline(y=borderline, line_dash="dot")

    # Labels
    unit_vals = g["unit"].dropna()
    y_label = f"Value ({unit_vals.iloc[0]})" if not unit_vals.empty else "Value"
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=40), height=450,
                      xaxis_title="Date", yaxis_title=y_label)
    if date_window is not None:
        fig.update_xaxes(range=[pd.to_datetime(date_window[0]), pd.to_datetime(date_window[1])])

    apply_warm_theme(fig)
    return fig


def plot_heatmap(df: pd.DataFrame, tests: List[str]) -> go.Figure:
    # Build matrix of health scores by date.
    g = df[df["test"].isin(tests)].copy()
    g["health_score"] = g.apply(
        lambda r: compute_heatmap_health_score(
            r["Value"], r.get("lower"), r.get("upper"), r.get("borderline")
        ),
        axis=1,
    )
    g = g.dropna(subset=["health_score"])
    if g.empty:
        return go.Figure()
    pivot = g.pivot_table(index="test", columns="Date", values="health_score", aggfunc="mean")
    ordered_tests = [test for test in tests if test in pivot.index]
    pivot = pivot.reindex(list(reversed(ordered_tests)))
    warm_scale = [
        [0.0, "#D96B42"],   # out of range
        [0.25, "#F1B07B"],  # slightly out of range
        [0.5, "#F2CC8F"],   # borderline
        [0.75, "#BED8C7"],  # safely in range
        [1.0, "#6F9A86"],   # strongly in range
    ]
    fig = px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        zmin=-1,
        zmax=1,
        color_continuous_scale=warm_scale,
        labels=dict(color="Health vs cutoff"),
    )
    fig.update_layout(height=300 + 20*len(pivot), margin=dict(l=10,r=10,t=40,b=40), title="Heatmap: distance from healthy range")
    apply_warm_theme(fig)
    return fig

def make_sparkline(df: pd.DataFrame, test: str) -> go.Figure:
    g = df[df["test"] == test].sort_values("Date")
    fig = go.Figure(go.Scatter(x=g["Date"], y=g["Value"], mode="lines", line=dict(color="#E07A5F", width=2)))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=80, xaxis=dict(visible=False), yaxis=dict(visible=False))
    apply_warm_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Fitbit visualization helpers
# ---------------------------------------------------------------------------

def plot_fitbit_timeseries(df: pd.DataFrame, y_col: str, title: str,
                           y_label: str, color: str = "#E07A5F",
                           show_trend: bool = True,
                           show_primary_series: bool = True,
                           date_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                           show_title: bool = False) -> go.Figure:
    """Generic time-series chart for Fitbit metrics."""
    fig = go.Figure()
    if df.empty or y_col not in df.columns:
        return fig

    df = df.dropna(subset=[y_col]).sort_values("Date")
    if df.empty:
        return fig

    visible_df = df
    if date_window is not None:
        window_start = pd.to_datetime(date_window[0])
        window_end = pd.to_datetime(date_window[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        visible_df = df[(df["Date"] >= window_start) & (df["Date"] <= window_end)].copy()
        if visible_df.empty:
            return fig

    if show_primary_series:
        fig.add_trace(go.Scatter(
            x=visible_df["Date"], y=visible_df[y_col],
            mode="lines+markers", name=title,
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color, line=dict(width=1, color="#FFFFFF")),
            hovertemplate=f"<b>{title}</b><br>%{{x|%Y-%m-%d}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
        ))

    # 7-day centered rolling average
    if len(df) >= 7:
        rolling = (
            df.set_index("Date")[y_col]
            .rolling("7D", min_periods=3, center=True)
            .mean()
            .reset_index()
        )
        if date_window is not None:
            rolling = rolling[(rolling["Date"] >= window_start) & (rolling["Date"] <= window_end)].copy()
        fig.add_trace(go.Scatter(
            x=rolling["Date"], y=rolling[y_col],
            mode="lines", name="7-day avg",
            line=dict(color="#81B29A", width=2, dash="dash"),
            hovertemplate=f"<b>7-day avg</b><br>%{{x|%Y-%m-%d}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
        ))

    # Linear trend
    if show_trend and len(visible_df) >= 14:
        days = (visible_df["Date"] - visible_df["Date"].min()).dt.days.values.astype(float)
        y_vals = visible_df[y_col].values.astype(float)
        slope, intercept = np.polyfit(days, y_vals, 1)
        x_line = pd.date_range(start=visible_df["Date"].min(), end=visible_df["Date"].max(), periods=50)
        x_days = (x_line - visible_df["Date"].min()).days.values.astype(float)
        y_line = slope * x_days + intercept
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name="Trend",
            line=dict(dash="dot", color="#D4C5B5", width=1.5),
        ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=40),
        height=400,
        xaxis_title="Date",
        yaxis_title=y_label,
    )
    if show_title:
        fig.update_layout(
            title=dict(
                text=title,
                x=0,
                xanchor="left",
                y=0.97,
                font=dict(size=14, color="#18322f"),
            )
        )
    if date_window is not None:
        fig.update_xaxes(range=[pd.to_datetime(date_window[0]), pd.to_datetime(date_window[1])])
    apply_warm_theme(fig)
    return fig


def plot_lift_timeseries(
    df: pd.DataFrame,
    title: str,
    color: str = "#E07A5F",
    date_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    standards_thresholds: Optional[Dict[str, float]] = None,
    strength_classification: Optional[Dict[str, object]] = None,
) -> go.Figure:
    """Plot session-best estimated 1RM over time for a single lift."""
    fig = go.Figure()
    if df.empty:
        return fig

    ordered = df.sort_values("workout_start_time").copy()
    if ordered.empty:
        return fig

    visible_df = ordered
    if date_window is not None:
        window_start = pd.to_datetime(date_window[0])
        window_end = pd.to_datetime(date_window[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        visible_df = ordered[
            (ordered["workout_start_time"] >= window_start) &
            (ordered["workout_start_time"] <= window_end)
        ].copy()
        if visible_df.empty:
            return fig

    y_axis_range = None
    if standards_thresholds:
        y_axis_range = compute_lift_standard_axis_range(
            visible_df["estimated_1rm_kg"],
            standards_thresholds,
            strength_classification,
        )
        add_lift_standard_bands(fig, standards_thresholds, y_axis_range)

    hover_custom = np.stack(
        [
            visible_df["best_weight_kg"].astype(float).to_numpy(),
            visible_df["best_reps"].astype(float).to_numpy(),
            visible_df["working_set_count"].astype(float).to_numpy(),
        ],
        axis=-1,
    )
    fig.add_trace(
        go.Scatter(
            x=visible_df["workout_start_time"],
            y=visible_df["estimated_1rm_kg"],
            customdata=hover_custom,
            mode="lines+markers",
            name="Session best",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color, line=dict(width=1, color="#FFFFFF")),
            hovertemplate=(
                f"<b>{title}</b><br>%{{x|%Y-%m-%d}}"
                "<br>Estimated 1RM: %{y:.1f} kg"
                "<br>Best set: %{customdata[0]:.1f} kg × %{customdata[1]:.0f}"
                "<br>Working sets: %{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )

    if len(visible_df) >= 6:
        days = (
            visible_df["workout_start_time"] - visible_df["workout_start_time"].min()
        ).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        y_vals = visible_df["estimated_1rm_kg"].astype(float).to_numpy()
        slope, intercept = np.polyfit(days, y_vals, 1)
        x_line = pd.date_range(
            start=visible_df["workout_start_time"].min(),
            end=visible_df["workout_start_time"].max(),
            periods=50,
        )
        x_days = (
            x_line - visible_df["workout_start_time"].min()
        ).total_seconds() / 86400.0
        y_line = slope * x_days + intercept
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend",
                line=dict(dash="dot", color="#D4C5B5", width=1.5),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title="Estimated 1RM (kg)",
    )
    if date_window is not None:
        fig.update_xaxes(range=[pd.to_datetime(date_window[0]), pd.to_datetime(date_window[1])])
    if y_axis_range is not None:
        fig.update_yaxes(range=list(y_axis_range))
    apply_warm_theme(fig)
    return fig


def format_display_date(value, fmt: str = "%d %b %Y", empty: str = "No data") -> str:
    if pd.isna(value):
        return empty
    return pd.to_datetime(value).strftime(fmt)


def format_fitbit_metric_value(
    value: float,
    unit: str = "",
    decimals: int = 1,
    *,
    thousands: bool = False,
    signed: bool = False,
) -> str:
    if pd.isna(value):
        return "—"

    if decimals == 0:
        number = f"{value:+,.0f}" if signed else f"{value:,.0f}"
    else:
        number = f"{value:+,.{decimals}f}" if signed else f"{value:,.{decimals}f}"

    if not unit:
        return number
    if unit == "%":
        return f"{number}%"
    return f"{number} {unit}"


def format_dexa_metric_value(value, unit: str = "") -> str:
    if pd.isna(value):
        return "—"
    unit = str(unit or "").strip()
    value = float(value)
    abs_value = abs(value)
    if unit in {"%", "percentage points"}:
        number = f"{value:.1f}"
    elif unit in {"g", "cal/day"}:
        number = f"{value:,.0f}"
    elif unit in {"kg", "years", "cm", "cm^2", "cm²", "cm^3", "cm³"}:
        number = f"{value:,.1f}" if abs_value < 100 else f"{value:,.0f}"
    elif unit in {"ratio", "score", "kg/m^2", "kg/m²", "g/cm^2", "g/cm²"}:
        number = f"{value:.2f}" if abs_value < 10 else f"{value:.1f}"
    else:
        number = format_lab_number(value)

    if not unit:
        return number
    if unit == "%":
        return f"{number}%"
    return f"{number} {unit}"


def format_dexa_change(value, unit: str = "") -> str:
    if pd.isna(value):
        return ""
    return format_dexa_metric_value(value, unit)


def get_dexa_latest_rows(df: pd.DataFrame, category: Optional[str] = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    data = df.copy()
    if category is not None:
        data = data[data["category"] == category].copy()
    if data.empty:
        return data
    data = data.sort_values(["test", "Date"])
    data["PrevValue"] = data.groupby("test")["Value"].shift(1)
    data["DeltaAbs"] = data["Value"] - data["PrevValue"]
    latest_idx = data.groupby("test")["Date"].idxmax()
    sort_columns = ["row_order"] if "row_order" in data.columns else ["category", "region", "metric"]
    return data.loc[latest_idx].sort_values(sort_columns).reset_index(drop=True)


def latest_dexa_metric_summary(df: pd.DataFrame, test: str) -> str:
    data = df[df["test"] == test].sort_values("Date")
    if data.empty:
        return ""
    latest = data.iloc[-1]
    return format_dexa_metric_value(latest["Value"], latest.get("unit", ""))


def get_dexa_summary_section(row: pd.Series) -> str:
    metric = str(row.get("metric", "")).strip()
    region = str(row.get("region", "")).strip()
    metric_key = metric.lower()
    region_key = region.lower()

    if metric in {"Dexa total mass", "Fat mass", "Lean mass"}:
        return "Mass metrics"
    if metric in {"Tissue %Fat", "Tissue %Lean"}:
        return "Body composition"
    if metric_key.startswith("vat "):
        return "VAT metrics"
    if metric == "Bone mineral content" or "bone density" in region_key:
        return "Bone density metrics"
    if metric in {"Left lean mass", "Right lean mass", "Right-left lean mass difference"}:
        return "Symmetry metrics"
    return "Other"


def render_dexa_summary_card_grid(cards: List[Dict[str, str]]) -> None:
    for start in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, card in zip(cols, cards[start:start + 4]):
            title_html = (
                f"<div class='dexa-summary-label'>{html.escape(card['title'])}</div>"
                if card.get("title") else ""
            )
            footnote_html = (
                f"<div class='dexa-summary-footnote'>{html.escape(card['footnote'])}</div>"
                if card.get("footnote") else ""
            )
            with col:
                card_html = (
                    f'<div class="dexa-summary-card dexa-summary-card--{html.escape(card["status"])}">'
                    f'{title_html}'
                    f'<div class="dexa-summary-value">{html.escape(card["value"])}</div>'
                    f'{footnote_html}'
                    '</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)


def render_dexa_summary_subsection(title: str, cards: List[Dict[str, str]]) -> None:
    if not cards:
        return
    st.markdown(f"<div class='dexa-summary-subsection'>{html.escape(title)}</div>", unsafe_allow_html=True)
    render_dexa_summary_card_grid(cards)


def render_dexa_summary_cards(summary_rows: pd.DataFrame, latest_date: pd.Timestamp) -> None:
    render_dexa_summary_card_grid(
        [
            {
                "title": "Latest sample",
                "value": format_display_date(latest_date),
                "footnote": "Dexa scan date",
                "status": "neutral",
            }
        ]
    )

    grouped_cards = {
        "Mass metrics": [],
        "Body composition": [],
        "VAT metrics": [],
        "Bone density metrics": [],
        "Symmetry metrics": [],
        "Other": [],
    }
    for _, row in summary_rows.iterrows():
        delta_text = ""
        if pd.notna(row.get("DeltaAbs")):
            delta_text = f"Change: {format_dexa_change(row.get('DeltaAbs'), row.get('unit'))}"
        if pd.to_datetime(row.get("Date")) != pd.to_datetime(latest_date):
            measured = format_display_date(row.get("Date"))
            delta_text = f"Last measured: {measured}" if not delta_text else f"{delta_text} • {measured}"
        section = get_dexa_summary_section(row)
        grouped_cards[section].append(
            {
                "title": str(row.get("test", "")),
                "value": format_dexa_metric_value(row.get("Value"), row.get("unit")),
                "footnote": delta_text,
                "status": dexa_health_status(
                    row.get("Value"),
                    row.get("lower"),
                    row.get("upper"),
                    row.get("borderline"),
                ),
            }
        )

    for section, cards in grouped_cards.items():
        render_dexa_summary_subsection(section, cards)


def formatted_fitbit_value_is_zero(display_value: str) -> bool:
    match = re.match(r"^[+-]?([0-9,]+(?:\.[0-9]+)?)", display_value.strip())
    if not match:
        return False
    numeric_value = float(match.group(1).replace(",", ""))
    return numeric_value == 0.0


def compute_fitbit_trend_per_month(df: pd.DataFrame, value_col: str) -> Optional[float]:
    """Estimate a meaningful linear trend over the visible window."""
    if df.empty or value_col not in df.columns:
        return None

    data = df.dropna(subset=[value_col]).sort_values("Date").copy()
    if len(data) < 7:
        return None

    data["Date"] = pd.to_datetime(data["Date"])
    x = (data["Date"] - data["Date"].min()).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    span_days = float(x.max() - x.min()) if len(x) else 0.0
    if span_days < 14:
        return None

    y = data[value_col].astype(float).to_numpy()
    if np.allclose(y, y[0]):
        return None

    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope):
        return None

    y_hat = slope * x + intercept
    ss_x = float(np.sum((x - x.mean()) ** 2))
    if ss_x <= 0:
        return None

    ss_res = float(np.sum((y - y_hat) ** 2))
    if ss_res <= 1e-12:
        t_stat = np.inf if abs(slope) > 0 else 0.0
    else:
        se_slope = np.sqrt((ss_res / max(len(y) - 2, 1)) / ss_x)
        t_stat = np.inf if se_slope <= 1e-12 else abs(slope) / se_slope

    fitted_change = abs(slope * span_days)
    variation = float(np.nanstd(y))
    if t_stat < 2:
        return None
    if variation > 0 and fitted_change < 0.25 * variation:
        return None

    return float(slope * 30.4375)


def compute_lift_trend_per_month(df: pd.DataFrame, value_col: str) -> Optional[float]:
    """Compute a monthly trend for lift data, trimming obvious negative deload outliers."""
    if df.empty or value_col not in df.columns:
        return None

    data = df.dropna(subset=[value_col]).sort_values("Date").copy()
    if len(data) < 3:
        return None

    data["Date"] = pd.to_datetime(data["Date"])
    x = (data["Date"] - data["Date"].min()).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    span_days = float(x.max() - x.min()) if len(x) else 0.0
    if span_days < 21:
        return None

    y = data[value_col].astype(float).to_numpy()
    if np.allclose(y, y[0]):
        return 0.0

    filtered = data.copy()
    if len(filtered) >= 5:
        median = float(filtered[value_col].median())
        mad = float(np.median(np.abs(filtered[value_col].to_numpy(dtype=float) - median)))
        if mad > 0:
            lower_bound = median - (2.5 * 1.4826 * mad)
            trimmed = filtered[filtered[value_col] >= lower_bound].copy()
            if len(trimmed) >= 3:
                filtered = trimmed

    x = (filtered["Date"] - filtered["Date"].min()).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    span_days = float(x.max() - x.min()) if len(x) else 0.0
    if span_days < 21:
        return None

    y = filtered[value_col].astype(float).to_numpy()
    if np.allclose(y, y[0]):
        return 0.0

    slope, _intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope):
        return None

    return float(slope * 30.4375)


def get_latest_fitbit_weight_kg() -> Optional[float]:
    try:
        weight_df = garmin_client.load_merged_weight()
    except Exception:
        return None

    if weight_df.empty or "Weight" not in weight_df.columns:
        return None

    weight_data = weight_df.dropna(subset=["Date", "Weight"]).sort_values("Date")
    if weight_data.empty:
        return None

    return float(weight_data.iloc[-1]["Weight"])


def format_strength_gap_to_next(
    classification: Optional[Dict[str, object]],
    trend_per_month: Optional[float] = None,
) -> Optional[str]:
    if not classification:
        return None

    next_category = classification.get("next_category")
    kg_to_next = classification.get("kg_to_next")
    if next_category is None or kg_to_next is None:
        return "Elite standard achieved"

    kg_value = float(kg_to_next)
    if kg_value <= 0:
        return f"At {next_category} threshold"

    subtitle = f"{format_fitbit_metric_value(kg_value, 'kg', 1)} to {next_category}"
    if trend_per_month is None or trend_per_month <= 0:
        return subtitle

    months_to_next = kg_value / float(trend_per_month)
    if not np.isfinite(months_to_next) or months_to_next <= 0:
        return subtitle

    if months_to_next < 1:
        estimate_text = "<1 month"
    elif months_to_next < 9.5:
        estimate_text = f"~{months_to_next:.1f} months"
    else:
        estimate_text = f"~{months_to_next:.0f} months"
    return f"{subtitle} • {estimate_text} at current pace"


def compute_lift_standard_axis_range(
    values: pd.Series,
    thresholds: Dict[str, float],
    classification: Optional[Dict[str, object]],
) -> Tuple[float, float]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        data_min = float(thresholds["Noob"])
        data_max = float(thresholds["Intermediate"])
    else:
        data_min = float(numeric_values.min())
        data_max = float(numeric_values.max())

    category = classification.get("category") if classification else None
    current_band_upper = classification.get("upper_bound") if classification else None
    levels = list(strength_standards.STANDARD_LEVELS)

    if category in levels:
        category_index = levels.index(category)
        if category_index == 0:
            lower_focus = float(thresholds["Noob"]) * 0.75
        else:
            lower_focus = float(thresholds[levels[category_index - 1]])
    else:
        lower_focus = float(thresholds["Noob"]) * 0.9

    y_min = max(0.0, min(data_min * 0.95, lower_focus * 0.97))

    if current_band_upper is not None:
        desired_y_max = float(current_band_upper) / 0.96
        y_max = max(data_max * 1.04, desired_y_max)
    elif category == "Elite+":
        y_max = data_max * 1.08
    elif category == "Elite":
        elite_cutoff = float(thresholds["Elite"])
        y_max = max(data_max * 1.06, elite_cutoff * 1.04)
    else:
        y_max = data_max * 1.08

    if y_max <= y_min:
        y_max = y_min + max(10.0, y_min * 0.2)
    return y_min, y_max


def add_lift_standard_bands(
    fig: go.Figure,
    thresholds: Dict[str, float],
    y_axis_range: Tuple[float, float],
) -> None:
    y_min, y_max = y_axis_range
    for band in strength_standards.get_category_bands(thresholds):
        lower_bound = float(band["lower_bound"] or 0.0)
        upper_bound = y_max if band["upper_bound"] is None else float(band["upper_bound"])
        clipped_lower = max(y_min, lower_bound)
        clipped_upper = min(y_max, upper_bound)
        if clipped_upper <= clipped_lower:
            continue
        fig.add_hrect(
            y0=clipped_lower,
            y1=clipped_upper,
            fillcolor=STRENGTH_STANDARD_BAND_COLORS[str(band["category"])],
            line_width=0,
            layer="below",
        )


def render_metric_card(
    title: str,
    value: str,
    latest_text: Optional[str] = None,
    variant: str = "primary",
) -> None:
    latest_html = f"<div class='fitbit-metric-latest'>{latest_text}</div>" if latest_text else ""
    st.markdown(
        f"""
        <div class="fitbit-metric-card fitbit-metric-card--{variant}">
            <div class="fitbit-metric-title">{title}</div>
            <div class="fitbit-metric-value">{value}</div>
            {latest_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fitbit_metric_stack(
    df: pd.DataFrame,
    value_col: str,
    average_title: str,
    trend_title: str,
    value_formatter,
    trend_month_formatter,
    trend_year_formatter,
    trend_layout: str = "stacked",
) -> None:
    """Render an average card plus an optional visible-window trend card."""
    if df.empty or value_col not in df.columns:
        render_metric_card(average_title, "—")
        return

    data = df.dropna(subset=[value_col]).sort_values("Date").copy()
    if data.empty:
        render_metric_card(average_title, "—")
        return

    average_value = data[value_col].mean()
    latest_value = data.iloc[-1][value_col]
    trend_per_month = compute_fitbit_trend_per_month(data, value_col)
    trend_value = None
    if trend_per_month is not None:
        if abs(trend_per_month) < 0.1:
            trend_value = trend_year_formatter(trend_per_month * 12)
        else:
            trend_value = trend_month_formatter(trend_per_month)
        if formatted_fitbit_value_is_zero(trend_value):
            trend_value = None

    if trend_layout == "inline" and trend_value is not None:
        avg_col, trend_col = st.columns(2)
        with avg_col:
            render_metric_card(
                average_title,
                value_formatter(average_value),
                f"Latest: {value_formatter(latest_value)}",
            )
        with trend_col:
            render_metric_card(
                trend_title,
                trend_value,
                variant="trend",
            )
        return

    render_metric_card(
        average_title,
        value_formatter(average_value),
        f"Latest: {value_formatter(latest_value)}",
    )

    if trend_value is not None:
        render_metric_card(
            trend_title,
            trend_value,
            variant="trend",
        )


def render_page_hero(title: str, subtitle: str, pills: Optional[List[str]] = None, eyebrow: str = "Biomarker Studio"):
    st.markdown(
        f"""
        <section class="page-hero">
            <div class="page-hero-copy">
                <h1>{title}</h1>
            </div>
            <div class="hero-orb hero-orb-a"></div>
            <div class="hero-orb hero-orb-b"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", eyebrow: Optional[str] = None):
    eyebrow_html = f"<div class='section-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-shell">
            {eyebrow_html}
            <div class="section-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_row(chips: List[str]):
    chip_html = "".join(f"<span class='context-chip'>{chip}</span>" for chip in chips if chip)
    if chip_html:
        st.markdown(f"<div class='context-chip-row'>{chip_html}</div>", unsafe_allow_html=True)


def apply_plotly_zoom_event(event: Optional[Dict], min_date: pd.Timestamp, max_date: pd.Timestamp) -> bool:
    """Apply a zoom event emitted by the custom Plotly bridge."""
    if not isinstance(event, dict):
        return False

    event_id = event.get("eventId")
    if event_id and st.session_state.get(LAST_PLOTLY_ZOOM_EVENT_KEY) == event_id:
        return False

    min_date = pd.to_datetime(min_date).normalize()
    max_date = pd.to_datetime(max_date).normalize()
    mode = event.get("mode")

    if mode == "all":
        new_preset = "All time"
        new_start = min_date
        new_end = max_date
    elif mode == "range":
        start = pd.to_datetime(event.get("start"), errors="coerce")
        end = pd.to_datetime(event.get("end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return False
        new_start = max(min_date, min(start.normalize(), max_date))
        new_end = max(min_date, min(end.normalize(), max_date))
        if new_start > new_end:
            new_start, new_end = new_end, new_start
        if new_start == min_date and new_end == max_date:
            new_preset = "All time"
        else:
            new_preset = "Custom"
    else:
        return False

    current_preset = st.session_state.get("global_time_preset", "All time")
    current_start = pd.to_datetime(st.session_state.get("global_time_start", min_date.date())).normalize()
    current_end = pd.to_datetime(st.session_state.get("global_time_end", max_date.date())).normalize()

    if event_id:
        st.session_state[LAST_PLOTLY_ZOOM_EVENT_KEY] = event_id

    if current_preset == new_preset and current_start == new_start and current_end == new_end:
        return False

    st.session_state["global_time_preset"] = new_preset
    st.session_state["global_time_start"] = new_start.date()
    st.session_state["global_time_end"] = new_end.date()
    return True


def render_plotly_zoom_sync(scope: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> None:
    """Mount the custom Plotly zoom bridge and sync any emitted range into session state."""
    min_date = pd.to_datetime(min_date).normalize()
    max_date = pd.to_datetime(max_date).normalize()

    current_start = pd.to_datetime(st.session_state.get("global_time_start", min_date.date())).normalize()
    current_end = pd.to_datetime(st.session_state.get("global_time_end", max_date.date())).normalize()
    current_start = max(min_date, min(current_start, max_date))
    current_end = max(min_date, min(current_end, max_date))
    if current_start > current_end:
        current_start, current_end = min_date, max_date

    event = PLOTLY_ZOOM_SYNC(
        scope=scope,
        minDate=min_date.date().isoformat(),
        maxDate=max_date.date().isoformat(),
        currentPreset=st.session_state.get("global_time_preset", "All time"),
        currentStart=current_start.date().isoformat(),
        currentEnd=current_end.date().isoformat(),
        key=f"plotly_zoom_sync_{scope}",
        default=None,
    )

    # The component value update already triggered this rerun, so applying the
    # zoom state here is enough. Avoid forcing a second rerun, which causes
    # visible flicker across the page.
    apply_plotly_zoom_event(event, min_date, max_date)


def inject_main_scroll_restorer(page_key: str) -> None:
    """Preserve main-content scroll position within a page, reset on page switch."""
    html = """
        <script>
        (function() {
          const PAGE_KEY = __PAGE_KEY__;
          const STORAGE_KEY = "health-main-scroll::" + PAGE_KEY;
          const PAGE_TRACK_KEY = "health-main-scroll::__active_page__";
          const doc = window.parent.document;
          const scroller = doc.querySelector('section.main');
          if (!scroller) return;

          const prevPage = window.localStorage.getItem(PAGE_TRACK_KEY);
          window.localStorage.setItem(PAGE_TRACK_KEY, PAGE_KEY);

          if (prevPage !== PAGE_KEY) {
            scroller.scrollTop = 0;
            window.localStorage.removeItem(STORAGE_KEY);
          } else {
            const saved = Number(window.localStorage.getItem(STORAGE_KEY) || 0);
            if (saved > 0) scroller.scrollTop = saved;
          }

          if (!scroller.dataset.mainScrollAttached) {
            scroller.dataset.mainScrollAttached = "true";
            scroller.addEventListener("scroll", () => {
              window.localStorage.setItem(STORAGE_KEY, String(scroller.scrollTop || 0));
            }, { passive: true });
          }
        })();
        </script>
        """
    components.html(
        html.replace("__PAGE_KEY__", json.dumps(page_key)),
        height=0,
        width=0,
    )


def inject_page_switch_cleaner() -> None:
    """Hide the outgoing page's UI while a page-switch rerun is in flight.

    Streamlit keeps the previous run's elements on screen until the new run
    replaces them or finishes, so on slow page loads the old page stays
    visible underneath the new one. The frontend marks those leftovers with
    data-stale="true" as soon as the rerun starts; when the nav radio changes
    we tag <body> so CSS can hide them, and untag once no stale elements
    remain. Scoped to nav changes so ordinary in-page reruns keep the
    default (stable, non-flickering) behaviour.
    """
    # The watcher must live in the parent window's JS realm: timers created
    # from this component's iframe die when Streamlit recreates the iframe
    # mid-rerun, which would leave the hiding class stuck on <body>.
    html = """
        <script>
        (function() {
          const doc = window.parent.document;
          if (doc.getElementById("bp-page-switch-cleaner")) return;
          const script = doc.createElement("script");
          script.id = "bp-page-switch-cleaner";
          script.textContent = `(function() {
            const CLASS = "bp-page-switching";
            // Block containers (expanders, tabs, forms, column rows) never get
            // the data-stale marker themselves - only the leaf elements inside
            // them do - so their chrome (headers, tab buttons) would survive
            // the CSS rule. Tag blocks whose marked descendants are ALL stale.
            const BLOCKS = '[data-testid="stExpander"], [data-testid="stTabs"], [data-testid="stForm"], [data-testid="stHorizontalBlock"]';
            let timer = null;
            let observer = null;

            function updateStaleBlocks() {
              document.querySelectorAll(BLOCKS).forEach((block) => {
                let anyStale = false;
                let anyFresh = false;
                block.querySelectorAll('[data-stale]').forEach((el) => {
                  if (el.getAttribute('data-stale') === 'true') anyStale = true;
                  else anyFresh = true;
                });
                block.classList.toggle('bp-stale-block', anyStale && !anyFresh);
              });
            }

            function endSwitch() {
              document.body.classList.remove(CLASS);
              if (timer) { clearInterval(timer); timer = null; }
              if (observer) { observer.disconnect(); observer = null; }
              document.querySelectorAll('.bp-stale-block').forEach((el) => {
                el.classList.remove('bp-stale-block');
              });
            }

            document.body.addEventListener("change", (ev) => {
              const input = ev.target;
              if (!input || !input.matches) return;
              if (!input.matches('div[role="radiogroup"][aria-label="Navigation"] input[type="radio"]')) return;

              document.body.classList.add(CLASS);
              if (timer) { clearInterval(timer); timer = null; }
              if (observer) { observer.disconnect(); observer = null; }

              updateStaleBlocks();
              // MutationObserver callbacks are batched microtasks and still
              // fire in throttled/background tabs (unlike rAF), so call the
              // update directly.
              observer = new MutationObserver(updateStaleBlocks);
              observer.observe(document.body, {
                subtree: true,
                childList: true,
                attributes: true,
                attributeFilter: ["data-stale"],
              });

              const start = Date.now();
              let seenStale = false;
              timer = setInterval(() => {
                const stale = document.querySelector('[data-stale="true"]');
                if (stale) seenStale = true;
                const elapsed = Date.now() - start;
                const switchDone = seenStale && !stale;
                const neverStarted = !seenStale && elapsed > 3000;
                if (switchDone || neverStarted || elapsed > 30000) endSwitch();
              }, 120);
            }, true);
          })();`;
          doc.body.appendChild(script);
        })();
        </script>
        """
    components.html(html, height=0, width=0)


def inject_sidebar_scroll_restorer(page_key: str) -> None:
    """Persist sidebar scroll position across Streamlit reruns."""
    fallback_page_key = page_key
    html = """
        <script>
        const STORAGE_PREFIX = "biomarker-studio-sidebar-scroll::";
        const FALLBACK_PAGE_KEY = __FALLBACK_PAGE_KEY__;

        function getActivePageKey() {
          try {
            const parentDoc = window.parent.document;
            const checked = parentDoc.querySelector('section[data-testid="stSidebar"] input[type="radio"]:checked');
            if (checked) {
              const label = checked.closest("label");
              const text = label ? label.innerText.trim() : "";
              if (text) return text;
            }
          } catch (error) {}
          return FALLBACK_PAGE_KEY;
        }

        function getStorageKey() {
          return STORAGE_PREFIX + getActivePageKey();
        }

        function findSidebarScroller() {
          const parentDoc = window.parent.document;
          const userContent = parentDoc.querySelector('[data-testid="stSidebarUserContent"]');
          if (userContent) return userContent;

          const explicit = parentDoc.querySelector('[data-testid="stSidebarContent"]');
          if (explicit) return explicit;

          const sidebar = parentDoc.querySelector('section[data-testid="stSidebar"]');
          if (!sidebar) return null;

          const candidates = [...sidebar.querySelectorAll("*")];
          return candidates.find((el) => {
            const style = window.parent.getComputedStyle(el);
            return (
              (style.overflowY === "auto" || style.overflowY === "scroll") &&
              el.scrollHeight > el.clientHeight + 4
            );
          }) || null;
        }

        function save(scroller) {
          if (!scroller) return;
          window.localStorage.setItem(getStorageKey(), String(scroller.scrollTop || 0));
        }

        function restore(scroller) {
          const saved = window.localStorage.getItem(getStorageKey());
          if (saved === null) {
            scroller.scrollTop = 0;
            return;
          }
          const top = Number(saved);
          if (Number.isNaN(top)) return;
          scroller.scrollTop = top;
        }

        function attach() {
          const scroller = findSidebarScroller();
          if (!scroller) {
            window.requestAnimationFrame(attach);
            return null;
          }

          restore(scroller);
          window.setTimeout(() => restore(scroller), 60);
          window.setTimeout(() => restore(scroller), 180);

          if (!scroller.dataset.scrollPersistAttached) {
            scroller.dataset.scrollPersistAttached = "true";
            scroller.addEventListener(
              "scroll",
              () => save(scroller),
              { passive: true }
            );

            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar) {
              const saveSoon = () => save(scroller);
              sidebar.addEventListener("pointerdown", saveSoon, true);
              sidebar.addEventListener("keydown", saveSoon, true);
              sidebar.addEventListener("input", saveSoon, true);
              sidebar.addEventListener("change", saveSoon, true);
            }
          }

          return scroller;
        }

        let scroller = attach();
        let lastSidebarNode = scroller;
        let lastPageKey = getActivePageKey();

        const watchSidebar = () => {
          const nextScroller = findSidebarScroller();
          const nextPageKey = getActivePageKey();
          if (!nextScroller) return;
          if (nextPageKey !== lastPageKey) {
            lastPageKey = nextPageKey;
            lastSidebarNode = nextScroller;
            attach();
            return;
          }
          if (nextScroller !== lastSidebarNode) {
            lastSidebarNode = nextScroller;
            attach();
          } else {
            restore(nextScroller);
          }
        };

        window.setInterval(watchSidebar, 350);
        </script>
        """
    components.html(
        html.replace("__FALLBACK_PAGE_KEY__", json.dumps(fallback_page_key)),
        height=0,
        width=0,
    )


def render_print_button() -> None:
    """In-page "Print / Save as PDF" button (lives in each view's Share & archive
    block). Clicking it flips a one-shot session flag and reruns IN-SESSION — no
    page reload — so all sidebar state (page, category, time range) is preserved.
    The rerun renders PRINT_MODE: charts rebuilt shorter (render_chart) and the
    print layout applied. inject_print_runner() then handles the dialog."""
    st.caption("Print this view (or save it as a PDF)")
    if st.button("🖨️  Print / Save as PDF", key="print_button"):
        st.session_state["_print_mode"] = True
        # Bump a nonce so the runner iframe below reloads (and re-runs) on every
        # print, not just the first.
        st.session_state["_print_nonce"] = st.session_state.get("_print_nonce", 0) + 1
        st.rerun()


def inject_print_runner() -> None:
    """Print-mode only: apply the print-width layout, open the print dialog, then
    restore the normal view by clicking the hidden restore button — an in-session
    rerun that (the print flag already popped) renders the normal app with ALL
    session state intact.

    Runs INSIDE the components iframe and reaches into window.parent — the same
    technique as inject_sidebar_scroll_restorer (a parent <script> injection was
    used before and silently failed). A nonce in the markup forces the iframe to
    reload each print so this re-runs. ?noauto=1 skips the auto-dialog for tests.
    """
    nonce = st.session_state.get("_print_nonce", 0)
    html = """
        <script>
        (function () {
          var pwin = window.parent, pdoc = pwin.document;
          function fireResize() { try { pwin.dispatchEvent(new pwin.Event('resize')); } catch (e) {} }

          // Apply the print-width layout so container-width charts re-fit to the
          // 680px canvas (their height is already reduced server-side).
          try { pdoc.body.classList.add('biomarker-printing'); } catch (e) {}
          var n = 0;
          (function tick() { fireResize(); if (++n < 9) pwin.setTimeout(tick, 110); })();

          function restore() {
            try { pdoc.body.classList.remove('biomarker-printing'); } catch (e) {}
            try {
              var btns = [].slice.call(pdoc.querySelectorAll('button'));
              var b = btns.filter(function (x) { return (x.textContent || '').indexOf('bmRestorePrint') >= 0; })[0];
              if (b) { b.click(); }   // .click() fires React's handler even off-screen
            } catch (e) {}
          }

          if (/[?&]noauto=1/.test(pwin.location.search)) return;   // test hook

          pwin.setTimeout(function () {
            pwin.print();
            pwin.setTimeout(restore, 400);
          }, 1100);
        })();
        /* print-nonce __NONCE__ */
        </script>
        """
    # Sentinel + hidden button the runner clicks to rerun back to normal. The
    # label avoids Markdown (e.g. __x__ renders bold and drops the underscores);
    # the sentinel lets the global CSS keep the button off-screen at all times.
    st.markdown('<span id="bm-restore-anchor"></span>', unsafe_allow_html=True)
    st.button("bmRestorePrint", key="_bm_restore")
    components.html(html.replace("__NONCE__", str(nonce)), height=0, width=0)


def render_time_controls(scope: str, min_date: pd.Timestamp, max_date: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Render shared time controls in the sidebar and return the active range."""
    preset_key = "global_time_preset"
    start_key = "global_time_start"
    end_key = "global_time_end"

    min_date = pd.to_datetime(min_date).normalize()
    max_date = pd.to_datetime(max_date).normalize()

    if st.session_state.get(preset_key) not in TIME_PRESETS:
        st.session_state[preset_key] = "All time"
    if start_key not in st.session_state:
        st.session_state[start_key] = min_date.date()
    if end_key not in st.session_state:
        st.session_state[end_key] = max_date.date()

    # Clamp any persisted global dates into the current page's available range.
    persisted_start = pd.to_datetime(st.session_state[start_key]).normalize()
    persisted_end = pd.to_datetime(st.session_state[end_key]).normalize()
    persisted_start = max(min_date, min(persisted_start, max_date))
    persisted_end = max(min_date, min(persisted_end, max_date))
    if persisted_start > persisted_end:
        persisted_start, persisted_end = min_date, max_date
    st.session_state[start_key] = persisted_start.date()
    st.session_state[end_key] = persisted_end.date()

    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header" style="border-bottom:none; margin-top:0.5rem; font-size:1.1rem;">Time</div>', unsafe_allow_html=True)

    # Label every slider stop ourselves (Streamlit only labels the ends) and
    # highlight the active one; the floating thumb value and built-in end
    # labels are hidden via CSS.
    short_labels = {"30 days": "30d", "90 days": "90d", "1 year": "1y", "All time": "All", "Custom": "Custom"}
    active_preset = st.session_state[preset_key]
    stops = []
    last_index = len(TIME_PRESETS) - 1
    for i, p in enumerate(TIME_PRESETS):
        if i == 0:
            pos = "left:0;"
        elif i == last_index:
            pos = "right:0;"
        else:
            pos = f"left:{i / last_index * 100:.0f}%; transform:translateX(-50%);"
        cls = "time-stop active" if p == active_preset else "time-stop"
        stops.append(f'<span class="{cls}" style="{pos}">{short_labels.get(p, p)}</span>')
    st.sidebar.markdown(f'<div class="time-stop-row">{"".join(stops)}</div>', unsafe_allow_html=True)

    preset = st.sidebar.select_slider(
        "Time window",
        options=TIME_PRESETS,
        key=preset_key,
        label_visibility="collapsed",
    )

    if preset == "Custom":
        start_col, end_col = st.sidebar.columns(2)
        with start_col:
            start_value = st.date_input(
                "Start",
                value=st.session_state[start_key],
                min_value=min_date.date(),
                max_value=max_date.date(),
                key=f"{start_key}_input",
            )
        with end_col:
            end_value = st.date_input(
                "End",
                value=st.session_state[end_key],
                min_value=min_date.date(),
                max_value=max_date.date(),
                key=f"{end_key}_input",
            )
        start_ts = pd.to_datetime(start_value).normalize()
        end_ts = pd.to_datetime(end_value).normalize()
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        st.session_state[start_key] = start_ts.date()
        st.session_state[end_key] = end_ts.date()
        return start_ts, end_ts

    if preset == "All time":
        start_ts, end_ts = min_date, max_date
    else:
        if preset == "1 year":
            start_ts = max(min_date, (max_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)).normalize())
        else:
            preset_windows = {
                "30 days": pd.Timedelta(days=29),
                "90 days": pd.Timedelta(days=89),
            }
            window = preset_windows.get(preset, pd.Timedelta(days=29))
            start_ts = max(min_date, (max_date - window).normalize())
        end_ts = max_date

    st.session_state[start_key] = start_ts.date()
    st.session_state[end_key] = end_ts.date()
    return start_ts, end_ts


# =====================================================================
# PAGE FUNCTIONS
# =====================================================================

def page_blood_panel():
    """Original Blood Panel Explorer page — all existing logic intact."""
    hero_placeholder = st.empty()

    with st.sidebar:
        # Convenience: auto-load local sheet_api_key.json if present
        local_key_path = os.path.join(os.path.dirname(__file__), "sheet_api_key.json")
        sheets = None
        if os.path.exists(local_key_path):
            try:
                with open(local_key_path, "r") as f:
                    service_account_info = json.load(f)

                # You can hardcode your preferred Google Sheet URL/ID here
                default_sheet_url = "https://docs.google.com/spreadsheets/d/1pfYaK6t25gcKdBAUu8_geyGlQP6wp6px9IyNUKI4wdw/edit?usp=sharing"
                spreadsheet_id = parse_spreadsheet_id(default_sheet_url)

                sheets = load_from_gsheets(spreadsheet_id, service_account_info)
                # Hide the rest of the Data source UI
            except Exception as e:
                st.error(f"Auto-load of {local_key_path} failed: {e}")

        if sheets is None:
            # Fall back to manual UI
            source = st.radio("Choose source", ["Google Sheets (recommended)", "Upload Excel (offline)"], index=0)
            if source.startswith("Google"):
                url_or_id = st.text_input("Google Sheet URL or ID", value="https://docs.google.com/spreadsheets/d/1pfYaK6t25gcKdBAUu8_geyGlQP6wp6px9IyNUKI4wdw/edit?usp=sharing")
                spreadsheet_id = parse_spreadsheet_id(url_or_id)
                st.write("Provide Google service account JSON (the sheet must be shared with the service account email).")
                sa_file = st.file_uploader("Service Account JSON", type=["json"])
                refresh = st.button("Refresh")
                if sa_file is not None:
                    try:
                        service_account_info = json.load(sa_file)
                        if refresh:
                            st.cache_data.clear()
                        sheets = load_from_gsheets(spreadsheet_id, service_account_info)
                    except Exception as e:
                        st.error(f"Failed to load from Google Sheets: {e}")
            else:
                up = st.file_uploader("Upload your Excel export (.xlsx)", type=["xlsx"])
                if up is not None:
                    sheets = load_from_xlsx(up)

    show_ref = True
    show_trend = True
    show_zones = True

    if sheets is None:
        st.info("Load data from Google Sheets or upload an Excel export to begin.")
        st.stop()

    # Extract key sheets
    all_data = sheets.get("All Data")
    ranges = sheets.get("Optimal Ranges")
    centiles = sheets.get("Centiles")
    notes = sheets.get("Labs and notes")
    if all_data is None or ranges is None:
        st.error("Expected sheets 'All Data' and 'Optimal Ranges' not found.")
        st.stop()

    # Normalize
    long = normalize_all_data(all_data)
    ranges_n = normalize_ranges(ranges)
    centiles_n = normalize_centiles(centiles)
    merged = attach_ranges(long, ranges_n)
    if isinstance(notes, pd.DataFrame):
        merged = attach_lab_notes(merged, notes)

    # US units: convert before status/z so all downstream views (charts, delta
    # table, heatmap, CSV) inherit the converted values and unit labels. The
    # toggle widget renders later (render_background_view_controls), so we read
    # its state here from session_state. Prefer the live widget key — Streamlit
    # restores it to the user's latest choice at the start of each rerun, so the
    # toggle takes effect immediately — and fall back to the shadow key, which
    # survives page switches when the widget key is dropped.
    us_units_on = st.session_state.get(
        "blood_panel_us_units",
        st.session_state.get("blood_panel_us_units_on", False),
    )
    if us_units_on:
        merged, centiles_n = convert_units_to_us(merged, centiles_n)

    # Status & z-scores
    merged["status"] = merged.apply(lambda r: status_from_bounds(r["Value"], r.get("lower"), r.get("upper")), axis=1)
    merged["z"] = merged.apply(lambda r: compute_zscore(r["Value"], r.get("lower"), r.get("upper")), axis=1)

    # Groups
    groups = load_groups_from_sheets(sheets)
    centile_metric_names = get_centile_metric_names(centiles_n, sorted(merged["test"].unique().tolist()))
    group_names = ["(All)"]
    if centile_metric_names:
        group_names.append(CENTILE_METRICS_CATEGORY)
    group_names.extend(groups.keys())
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="section-header" style="border-bottom:none; margin-top:0.5rem; font-size:1.1rem;">Category</div>', unsafe_allow_html=True)
    default_group = "Inflamation"
    default_selection = default_group if default_group in group_names else group_names[0]
    pending_category = st.session_state.pop("blood_panel_category_target", None)
    if pending_category in group_names:
        st.session_state["blood_panel_category"] = pending_category
    if st.session_state.get("blood_panel_category") not in group_names:
        st.session_state["blood_panel_category"] = default_selection
    grp = st.sidebar.selectbox("Category", options=group_names, key="blood_panel_category")
    if grp == CENTILE_METRICS_CATEGORY:
        selected_tests = centile_metric_names
    elif grp != "(All)":
        selected_tests = groups.get(grp, [])
    else:
        selected_tests = sorted(merged["test"].unique().tolist())

    render_plotly_zoom_sync(
        "blood_panel",
        merged["Date"].min(),
        merged["Date"].max(),
    )
    blood_time_start, blood_time_end = render_time_controls(
        "blood_panel",
        merged["Date"].min(),
        merged["Date"].max(),
    )
    background_view = render_background_view_controls("blood_panel")

    hero_title = grp if grp != "(All)" else "Blood Panel"
    with hero_placeholder.container():
        current_index = group_names.index(grp)
        render_page_hero(
            hero_title,
            "A polished view of your longitudinal lab results, reference ranges, and recent changes across every marker that matters.",
            pills=["Longitudinal trends", "Reference zones", "Consumer-grade detail"],
            eyebrow="Biomarker Studio",
        )
        st.markdown("<div class='hero-nav-slot'>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3 = st.columns([1.4, 1.4, 8])
        with nav_col1:
            if st.button("← Prev", key="blood_panel_prev_category", use_container_width=True, disabled=current_index == 0):
                st.session_state["blood_panel_category_target"] = group_names[current_index - 1]
                st.rerun()
        with nav_col2:
            if st.button("Next →", key="blood_panel_next_category", use_container_width=True, disabled=current_index == len(group_names) - 1):
                st.session_state["blood_panel_category_target"] = group_names[current_index + 1]
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar.expander("Filters", expanded=False):
        search = st.text_input("Search test name")
        if search:
            candidates = [t for t in selected_tests if search.lower() in t.lower()]
        else:
            candidates = selected_tests

        tests_selected = st.multiselect("Select tests to visualize", options=candidates, default=candidates[:len(candidates)])

    data = merged[
        (merged["Date"] >= blood_time_start) &
        (merged["Date"] <= blood_time_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    ].copy()

    # Insights
    data = compute_deltas(data)

    # Restrict to selected tests (use full data if nothing selected)
    data_sel = data[data["test"].isin(tests_selected)] if tests_selected else data.copy()

    if data_sel.empty:
        st.info("No data is available in the selected time window.")
        st.stop()

    # Use the same filtered set as delta table
    latest_date = pd.to_datetime(data_sel["Date"].max())
    out_now = data_sel[(data_sel["Date"] == latest_date) & (data_sel["status"].isin(["low","high"]))]["test"].nunique()
    total_measured = data_sel[data_sel["Date"] == latest_date]["test"].nunique()
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Latest sample", format_display_date(latest_date))
    with k2:
        st.metric("Out of range", str(int(out_now)))
    with k3:
        st.metric("Measured markers", str(int(total_measured)))

    # "What's changed since last test"
    render_section_header(
        "Delta Since Last Test",
        "A ranked view of what moved most since each biomarker was last measured.",
        "Change analysis",
    )

    # Remove accidental duplicates (defensive; canonical ranges should already prevent these)
    data_sel = (data_sel.sort_values(["test", "Date"])
                        .drop_duplicates(subset=["test", "Date", "Value"], keep="last"))

    # Compute deltas within the current selection
    dsel = compute_deltas(data_sel)

    # Latest per test (within selection)
    if dsel.empty:
        st.write("No data in the current selection.")
    else:
        latest_idx = dsel.groupby("test")["Date"].idxmax()
        latest_per_test = dsel.loc[latest_idx].copy()
        latest_per_test = latest_per_test[latest_per_test["Date"] == latest_date].copy()

        # If any selected test has no previous value in the selection, fill from full history
        needs_prev = latest_per_test["PrevValue"].isna()
        if needs_prev.any():
            hist = (merged[merged["test"].isin(latest_per_test["test"])]
                        .sort_values(["test","Date"])
                        .drop_duplicates(subset=["test","Date","Value"], keep="last"))
            dhist = compute_deltas(hist)
            latest_hist = dhist.loc[dhist.groupby("test")["Date"].idxmax(), ["test","PrevValue"]]
            latest_per_test = latest_per_test.drop(columns=["PrevValue"]).merge(latest_hist, on="test", how="left")

        # Build view
        latest_per_test["Δ"]  = (latest_per_test["Value"] - latest_per_test["PrevValue"]).round(2)
        latest_per_test["Δ%"] = (latest_per_test["Δ"] / latest_per_test["PrevValue"] * 100).round(1)
        latest_per_test["display_status"] = latest_per_test["status"]
        improved_high = (latest_per_test["status"] == "high") & (latest_per_test["Δ"] < 0)
        improved_low = (latest_per_test["status"] == "low") & (latest_per_test["Δ"] > 0)
        latest_per_test.loc[improved_high, "display_status"] = "high but improved"
        latest_per_test.loc[improved_low, "display_status"] = "low but improved"
        latest_per_test["status_order"] = latest_per_test["display_status"].map({
            "high": 0,
            "low": 0,
            "high but improved": 1,
            "low but improved": 1,
            "normal": 2,
            "unknown": 3,
        }).fillna(3)

        cols = ["test","PrevValue","Value","Δ%","display_status","unit"]
        table_df = (
            latest_per_test
            .sort_values(["status_order", "Δ%"], ascending=[True, False], na_position="last")[cols]
            .reset_index(drop=True)
        )

        # Friendly column names
        table_df = table_df.rename(columns={
            "test": "Test", "PrevValue": "Previous",
            "Value": "Current", "Δ%": "Change %", "display_status": "Status", "unit": "Unit",
        })

        # Format numeric columns
        for c in ["Previous", "Current"]:
            table_df[c] = table_df[c].apply(format_lab_number)
        table_df["Change %"] = table_df["Change %"].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")

        # Apply full-row tinting based on status
        styled = table_df.style.apply(
            lambda row: [highlight_status(row["Status"])] * len(row),
            axis=1,
        )
        styled = styled.set_properties(
            subset=["Unit"],
            **{
                "color": "#7A7F8C",
                "background-color": "rgba(61, 64, 91, 0.06)",
                "font-size": "0.85rem",
                "white-space": "nowrap",
            },
        )

        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Grid of charts for selected tests
    selected_tests_title = grp if grp != "(All)" else "All Markers"
    render_section_header(
        selected_tests_title,
        "",
        "Visual explorer",
    )
    if tests_selected:
        graph_tests = [t for t in tests_selected if t != "APOE Genotype"]
        ncols = 3
        rows = (len(graph_tests) + ncols - 1)//ncols
        for r in range(rows):
            cols = st.columns(ncols)
            for i in range(ncols):
                idx = r*ncols + i
                if idx >= len(graph_tests):
                    continue
                t = graph_tests[idx]
                with cols[i]:
                    latest_summary = latest_test_summary(data, t)
                    summary_html = f"<div class='chart-card-meta'>Latest: {latest_summary}</div>" if latest_summary else ""
                    st.markdown(
                        f"""
                        <div class='chart-card-title'>{t}</div>
                        {summary_html}
                        """,
                        unsafe_allow_html=True,
                    )
                    fig = plot_single_test(
                        data,
                        t,
                        show_ref=show_ref,
                        show_regression=show_trend,
                        show_zones=show_zones,
                        date_window=(blood_time_start, blood_time_end),
                        background_view=background_view,
                        centiles=centiles_n,
                    )
                    render_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Use the sidebar to select tests to visualize.")

    # Heatmap overview
    render_section_header(
        "Overview Heatmap",
        "A quick read on how your selected markers sit within their reference bands over time.",
        "Trends",
    )
    if len(tests_selected) >= 2:
        fig_hm = plot_heatmap(data, tests_selected[:30])  # limit to 30 for readability
        render_chart(fig_hm, use_container_width=True)
    else:
        st.caption("Select 2 or more tests to see the heatmap.")

    # Export / Share & archive (omitted from the printout itself).
    if PRINT_MODE:
        return
    render_section_header(
        "Export",
        "Take the current story with you as a static report or a filtered dataset.",
        "Share & archive",
    )
    colP, colA, colB = st.columns(3)
    with colP:
        render_print_button()
    with colA:
        st.caption("Export selected charts to standalone HTML")
        if st.button("Export HTML"):
            # Create a simple HTML with embedded plotly divs
            html_parts = []
            for t in tests_selected:
                fig = plot_single_test(
                    data,
                    t,
                    show_ref=show_ref,
                    show_regression=show_trend,
                    background_view=background_view,
                    centiles=centiles_n,
                )
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            html = f"<html><head><meta charset='utf-8'><title>Blood Panel Export</title></head><body>{''.join(html_parts)}</body></html>"
            st.download_button("Download file", data=html, file_name="blood_panel_export.html", mime="text/html")

    with colB:
        st.caption("Export current filtered dataset (CSV)")
        csv = data.sort_values(["test","Date"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="blood_panel_data.csv", mime="text/csv")


def page_dexa():
    """DEXA body-composition explorer backed by the Dexa Google Sheet tab."""
    hero_placeholder = st.empty()

    with st.sidebar:
        local_key_path = os.path.join(os.path.dirname(__file__), "sheet_api_key.json")
        sheets = None
        if os.path.exists(local_key_path):
            try:
                with open(local_key_path, "r") as f:
                    service_account_info = json.load(f)

                default_sheet_url = "https://docs.google.com/spreadsheets/d/1pfYaK6t25gcKdBAUu8_geyGlQP6wp6px9IyNUKI4wdw/edit?usp=sharing"
                spreadsheet_id = parse_spreadsheet_id(default_sheet_url)
                sheets = load_from_gsheets(spreadsheet_id, service_account_info)
            except Exception as e:
                st.error(f"Auto-load of {local_key_path} failed: {e}")

        if sheets is None:
            source = st.radio("Choose source", ["Google Sheets (recommended)", "Upload Excel (offline)"], index=0, key="dexa_source")
            if source.startswith("Google"):
                url_or_id = st.text_input(
                    "Google Sheet URL or ID",
                    value="https://docs.google.com/spreadsheets/d/1pfYaK6t25gcKdBAUu8_geyGlQP6wp6px9IyNUKI4wdw/edit?usp=sharing",
                    key="dexa_sheet_url",
                )
                spreadsheet_id = parse_spreadsheet_id(url_or_id)
                st.write("Provide Google service account JSON (the sheet must be shared with the service account email).")
                sa_file = st.file_uploader("Service Account JSON", type=["json"], key="dexa_service_account_json")
                refresh = st.button("Refresh", key="dexa_refresh")
                if sa_file is not None:
                    try:
                        service_account_info = json.load(sa_file)
                        if refresh:
                            st.cache_data.clear()
                        sheets = load_from_gsheets(spreadsheet_id, service_account_info)
                    except Exception as e:
                        st.error(f"Failed to load from Google Sheets: {e}")
            else:
                up = st.file_uploader("Upload your Excel export (.xlsx)", type=["xlsx"], key="dexa_xlsx")
                if up is not None:
                    sheets = load_from_xlsx(up)

    if sheets is None:
        st.info("Load data from Google Sheets or upload an Excel export to begin.")
        st.stop()

    dexa_sheet = sheets.get("Dexa")
    if dexa_sheet is None:
        st.error("Expected sheet 'Dexa' not found.")
        st.stop()
    centiles_n = normalize_centiles(sheets.get("Centiles"))

    dexa_long = normalize_dexa_data(dexa_sheet)
    if dexa_long.empty:
        st.error("No usable Dexa measurements were found.")
        st.stop()

    latest_date = pd.to_datetime(dexa_long["Date"].max())
    if st.session_state.get("_dexa_metric_selection_scope") != "all_rows":
        st.session_state.pop("dexa_metric_selection", None)
        st.session_state["_dexa_metric_selection_scope"] = "all_rows"

    render_plotly_zoom_sync(
        "dexa",
        dexa_long["Date"].min(),
        dexa_long["Date"].max(),
    )
    dexa_time_start, dexa_time_end = render_time_controls(
        "dexa",
        dexa_long["Date"].min(),
        dexa_long["Date"].max(),
    )
    background_view = "Biohacker"

    with hero_placeholder.container():
        render_page_hero(
            "Dexa",
            "Body composition, visceral fat, lean-mass balance, and bone-density trends from your DEXA scans.",
            pills=["Body composition", "Lean mass balance", "Bone density"],
            eyebrow="Biomarker Studio",
        )

    data = dexa_long[
        (dexa_long["Date"] >= dexa_time_start) &
        (dexa_long["Date"] <= dexa_time_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    ].copy()

    render_section_header(
        "Headline Numbers",
        "",
        "Summary",
    )
    summary_rows = get_dexa_latest_rows(dexa_long, "Summary")
    if summary_rows.empty:
        st.info("No Summary metrics are available in the selected time window.")
    else:
        render_dexa_summary_cards(summary_rows, latest_date)

    selected_tests = (
        dexa_long
        .sort_values("row_order")["test"]
        .drop_duplicates()
        .tolist()
    )

    with st.sidebar.expander("Filters", expanded=False):
        search = st.text_input("Search Dexa metric", key="dexa_search")
        if search:
            candidates = [t for t in selected_tests if search.lower() in t.lower()]
        else:
            candidates = selected_tests
        current_selection = st.session_state.get("dexa_metric_selection")
        if isinstance(current_selection, list):
            st.session_state["dexa_metric_selection"] = [t for t in current_selection if t in candidates]
        tests_selected = st.multiselect(
            "Select metrics to visualize",
            options=candidates,
            default=candidates[:len(candidates)],
            key="dexa_metric_selection",
        )

    render_section_header(
        "All Dexa Metrics",
        "",
        "Visual explorer",
    )
    if tests_selected:
        ncols = 3
        rows = (len(tests_selected) + ncols - 1) // ncols
        for r in range(rows):
            cols = st.columns(ncols)
            for i in range(ncols):
                idx = r * ncols + i
                if idx >= len(tests_selected):
                    continue
                test_name = tests_selected[idx]
                with cols[i]:
                    latest_summary = latest_dexa_metric_summary(data, test_name)
                    summary_html = f"<div class='chart-card-meta'>Latest: {html.escape(latest_summary)}</div>" if latest_summary else ""
                    st.markdown(
                        f"""
                        <div class='chart-card-title'>{html.escape(test_name)}</div>
                        {summary_html}
                        """,
                        unsafe_allow_html=True,
                    )
                    fig = plot_single_test(
                        data,
                        test_name,
                        show_ref=True,
                        show_regression=True,
                        show_zones=True,
                        date_window=(dexa_time_start, dexa_time_end),
                        background_view=background_view,
                        centiles=centiles_n,
                        centile_test=DEXA_CENTILE_TEST_MAP.get(test_name),
                    )
                    render_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Use the sidebar to select Dexa metrics to visualize.")

    if PRINT_MODE:
        return
    render_section_header(
        "Export",
        "Print this view, or download the current Dexa data as a filtered dataset.",
        "Share & archive",
    )
    colP, colB = st.columns(2)
    with colP:
        render_print_button()
    with colB:
        st.caption("Export current filtered dataset (CSV)")
        csv = data.sort_values(["category", "region", "metric", "Date"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="dexa_data.csv", mime="text/csv")


def page_fitbit_data():
    """Fitbit data visualization page — weight, HRV, and RHR."""
    render_page_hero(
        "Fitbit Data",
        "A calm, consumer-grade dashboard for recovery, sleep, movement, and body metrics streamed from Fitbit.",
        pills=["Daily rhythm", "Recovery signals", "Activity patterns"],
        eyebrow="Connected health",
    )

    if not fitbit_client.is_configured():
        st.warning("Fitbit is not configured yet. Go to **Settings** to connect your account.")
        st.stop()

    if not fitbit_client.has_valid_token():
        st.warning("Fitbit authorization has expired. Go to **Settings** to re-authorize.")
        st.stop()

    show_trend = True
    sync_mode = st.session_state.pop("fitbit_sync_mode", None)
    force_full = sync_mode == "full"
    loading_placeholder = st.empty()

    fetch_steps = [
        ("Weight", "fetch_weight", "weight"),
        ("HRV", "fetch_hrv", "hrv"),
        ("Resting Heart Rate", "fetch_rhr", "rhr"),
        ("Breathing Rate", "fetch_breathing_rate", "breathing_rate"),
        ("Sleep", "fetch_sleep", "sleep"),
        ("Activity", "fetch_activity", "activity"),
    ]

    results = {
        func_name: fitbit_client.load_cached_dataframe(metric_name)
        for _, func_name, metric_name in fetch_steps
    }
    # Weight now merges the frozen Fitbit history (incl. manually logged
    # entries) with live Garmin scale data; Garmin wins on shared dates.
    results["fetch_weight"] = garmin_client.load_merged_weight()

    def _weight_cache_fresh() -> bool:
        if garmin_client.is_configured():
            return garmin_client.is_cache_fresh()
        return fitbit_client.is_cache_fresh("weight")

    has_any_cache = any(not df.empty for df in results.values())
    stale_labels = [
        label for label, _, metric_name in fetch_steps
        if not (
            _weight_cache_fresh() if metric_name == "weight"
            else fitbit_client.is_cache_fresh(metric_name)
        )
    ]
    auto_refresh = sync_mode is None and has_any_cache and len(stale_labels) > 0
    if sync_mode is None and not has_any_cache:
        sync_mode = "incremental"

    def fetch_weight_all_sources(force_full: bool):
        """Refresh weight. Garmin is the live source once configured; the
        Fitbit weight API stopped receiving scale data on 2026-07-22, so it
        is only fetched while Garmin is not yet set up."""
        warnings = []
        if garmin_client.is_configured():
            try:
                garmin_client.fetch_weight(force_full=force_full)
            except Exception as e:
                warnings.append(f"Weight (Garmin): live sync failed, showing cached data ({e})")
        else:
            try:
                fitbit_client.fetch_weight(force_full=force_full)
            except Exception as e:
                warnings.append(f"Weight (Fitbit): live sync failed, showing cached data ({e})")
        merged = garmin_client.load_merged_weight()
        return merged, ("; ".join(warnings) or None)

    def fetch_fitbit_metric(label: str, func_name: str, metric_name: str, force_full: bool):
        if metric_name == "weight":
            return fetch_weight_all_sources(force_full)
        try:
            data = getattr(fitbit_client, func_name)(force_full=force_full)
            return data, None
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                return pd.DataFrame(), f"{label}: not available (403 Forbidden)"
            cached = fitbit_client.load_cached_dataframe(metric_name)
            if not cached.empty:
                return cached, f"{label}: live sync failed, showing cached data ({e})"
            return pd.DataFrame(), f"{label}: live sync failed and no cached data is available ({e})"
        except Exception as e:
            cached = fitbit_client.load_cached_dataframe(metric_name)
            if not cached.empty:
                return cached, f"{label}: live sync failed, showing cached data ({e})"
            return pd.DataFrame(), f"{label}: live sync failed and no cached data is available ({e})"

    fetch_warnings = []
    weight_df = results["fetch_weight"]
    hrv_df = results["fetch_hrv"]
    rhr_df = results["fetch_rhr"]
    br_df = results["fetch_breathing_rate"]
    sleep_df = results["fetch_sleep"]
    activity_df = results["fetch_activity"]
    if (
        sync_mode is None
        and not activity_df.empty
        and not {"MinutesFatBurn", "MinutesCardio", "MinutesPeak"}.issubset(activity_df.columns)
    ):
        sync_mode = "incremental"
        stale_labels.append("Activity heart-rate zones")

    # Fitbit's separate sleep-score endpoint is empty for this account;
    # use the efficiency field as the sleep score shown in the product UI.
    if not sleep_df.empty and "Efficiency" in sleep_df.columns:
        sleep_df = sleep_df.copy()
        sleep_df["SleepScore"] = sleep_df["Efficiency"]

    # Keep full-history copies for chart calculations so rolling averages
    # at the start of the visible window can still use preceding days.
    weight_chart_df = weight_df.copy()
    hrv_chart_df = hrv_df.copy()
    rhr_chart_df = rhr_df.copy()
    br_chart_df = br_df.copy()
    sleep_chart_df = sleep_df.copy()
    activity_chart_df = activity_df.copy()

    # Build Fitbit sidebar controls outside the content placeholder so the
    # sidebar DOM remains stable across reruns.
    all_dfs = [weight_df, hrv_df, rhr_df, br_df, sleep_df, activity_df]
    non_empty_dfs = [df for df in all_dfs if not df.empty]
    if non_empty_dfs:
        fitbit_min_date = min(pd.to_datetime(df["Date"]).min() for df in non_empty_dfs)
        fitbit_max_date = max(pd.to_datetime(df["Date"]).max() for df in non_empty_dfs)
        render_plotly_zoom_sync(
            "fitbit",
            fitbit_min_date,
            fitbit_max_date,
        )
        fitbit_time_start, fitbit_time_end = render_time_controls(
            "fitbit",
            fitbit_min_date,
            fitbit_max_date,
        )
        st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        show_primary_fitbit_series = st.sidebar.toggle(
            "Show daily data series",
            value=True,
            key="fitbit_show_primary_series",
        )
        end_inclusive = fitbit_time_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        weight_df = weight_df[(weight_df["Date"] >= fitbit_time_start) & (weight_df["Date"] <= end_inclusive)].copy() if not weight_df.empty else weight_df
        hrv_df = hrv_df[(hrv_df["Date"] >= fitbit_time_start) & (hrv_df["Date"] <= end_inclusive)].copy() if not hrv_df.empty else hrv_df
        rhr_df = rhr_df[(rhr_df["Date"] >= fitbit_time_start) & (rhr_df["Date"] <= end_inclusive)].copy() if not rhr_df.empty else rhr_df
        br_df = br_df[(br_df["Date"] >= fitbit_time_start) & (br_df["Date"] <= end_inclusive)].copy() if not br_df.empty else br_df
        sleep_df = sleep_df[(sleep_df["Date"] >= fitbit_time_start) & (sleep_df["Date"] <= end_inclusive)].copy() if not sleep_df.empty else sleep_df
        activity_df = activity_df[(activity_df["Date"] >= fitbit_time_start) & (activity_df["Date"] <= end_inclusive)].copy() if not activity_df.empty else activity_df
    else:
        fitbit_time_start = fitbit_time_end = None
        show_primary_fitbit_series = True

    refresh_notice = st.session_state.pop("fitbit_refresh_notice", None)
    if refresh_notice:
        st.success(refresh_notice)
    if auto_refresh:
        stale_text = ", ".join(stale_labels[:3])
        if len(stale_labels) > 3:
            stale_text += ", and more"
        st.info(f"Showing cached Fitbit data now while stale feeds refresh in the background: {stale_text}.")
    for warning in fetch_warnings:
        st.warning(warning)

    all_dfs = [weight_df, hrv_df, rhr_df, br_df, sleep_df, activity_df]
    latest_dates = []
    for df in all_dfs:
        if not df.empty:
            latest_dates.append(pd.to_datetime(df.iloc[-1]['Date']))
    if latest_dates:
        most_recent = max(latest_dates)
        stalest_feed_date = min(latest_dates)
        if stalest_feed_date == most_recent:
            k1 = st.columns(1)[0]
            with k1:
                st.metric("Latest data point", format_display_date(most_recent))
                st.caption("Most recent Fitbit date available across all feeds")
        else:
            k1, k2 = st.columns(2)
            with k1:
                st.metric("Latest data point", format_display_date(most_recent))
                st.caption("Most recent Fitbit date available across all feeds")
            with k2:
                st.metric("Stalest feed", format_display_date(stalest_feed_date))
                st.caption("Oldest last-available date across the Fitbit feeds shown here")

    # ==================================================================
    # WEIGHT SECTION
    # ==================================================================
    render_section_header("Weight", "", "Body composition")
    if not weight_df.empty:
        k1, k2 = st.columns([1.5, 3])
        with k1:
            render_fitbit_metric_stack(
                weight_df,
                "Weight",
                "Average Weight",
                "Weight Trend",
                lambda v: format_fitbit_metric_value(v, "kg", 1),
                lambda v: format_fitbit_metric_value(v, "kg / month", 1, signed=True),
                lambda v: format_fitbit_metric_value(v, "kg / year", 1, signed=True),
                trend_layout="inline",
            )
        fig_w = plot_fitbit_timeseries(weight_chart_df, "Weight", "Weight", "kg", color="#E07A5F", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
        render_chart(fig_w, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No weight data available.")

    # ==================================================================
    # HEALTH METRICS SECTION
    # ==================================================================
    render_section_header("Health Metrics", "", "Recovery")

    hm1, hm2, hm3 = st.columns(3)
    with hm1:
        render_fitbit_metric_stack(
            hrv_df,
            "RMSSD",
            "Average HRV (RMSSD)",
            "HRV Trend",
            lambda v: format_fitbit_metric_value(v, "ms", 0),
            lambda v: format_fitbit_metric_value(v, "ms / month", 1, signed=True),
            lambda v: format_fitbit_metric_value(v, "ms / year", 1, signed=True),
        )
    with hm2:
        render_fitbit_metric_stack(
            rhr_df,
            "RHR",
            "Average RHR",
            "RHR Trend",
            lambda v: format_fitbit_metric_value(v, "bpm", 0),
            lambda v: format_fitbit_metric_value(v, "bpm / month", 1, signed=True),
            lambda v: format_fitbit_metric_value(v, "bpm / year", 1, signed=True),
        )
    with hm3:
        render_fitbit_metric_stack(
            br_df,
            "BreathingRate",
            "Average Breathing Rate",
            "Breathing Trend",
            lambda v: format_fitbit_metric_value(v, "brpm", 1),
            lambda v: format_fitbit_metric_value(v, "brpm / month", 2, signed=True),
            lambda v: format_fitbit_metric_value(v, "brpm / year", 1, signed=True),
        )

    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    if not hrv_df.empty:
        fig_hrv = plot_fitbit_timeseries(hrv_chart_df, "RMSSD", "HRV", "ms", color="#81B29A", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_hrv, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No HRV data available.")

    if not rhr_df.empty:
        fig_rhr = plot_fitbit_timeseries(rhr_chart_df, "RHR", "Resting HR", "bpm", color="#F2CC8F", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_rhr, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No resting heart rate data available.")

    if not br_df.empty:
        fig_br = plot_fitbit_timeseries(br_chart_df, "BreathingRate", "Breathing Rate", "brpm", color="#7EB8DA", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_br, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No breathing rate data available.")

    # ==================================================================
    # SLEEP SECTION
    # ==================================================================
    render_section_header("Sleep", "", "Rest")
    if not sleep_df.empty:
        sl1, sl2 = st.columns(2)
        with sl1:
            render_fitbit_metric_stack(
                sleep_df,
                "DurationHours",
                "Average Sleep Duration",
                "Sleep Duration Trend",
                lambda v: format_fitbit_metric_value(v, "hrs", 1),
                lambda v: format_fitbit_metric_value(v, "hrs / month", 2, signed=True),
                lambda v: format_fitbit_metric_value(v, "hrs / year", 1, signed=True),
            )
        with sl2:
            render_fitbit_metric_stack(
                sleep_df,
                "SleepScore",
                "Average Sleep Score",
                "Sleep Score Trend",
                lambda v: format_fitbit_metric_value(v, "", 0),
                lambda v: format_fitbit_metric_value(v, "points / month", 1, signed=True),
                lambda v: format_fitbit_metric_value(v, "points / year", 1, signed=True),
            )

        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

        fig_dur = plot_fitbit_timeseries(sleep_chart_df, "DurationHours", "Sleep Duration", "hours", color="#8E7CC3", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_dur, use_container_width=True, config={"displayModeBar": False})

        if "SleepScore" in sleep_df.columns and sleep_df["SleepScore"].notna().any():
            fig_score = plot_fitbit_timeseries(sleep_chart_df, "SleepScore", "Sleep Score", "score", color="#6AA84F", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
            render_chart(fig_score, use_container_width=True, config={"displayModeBar": False})

        stage_cols = {"Deep": "#3D85C6", "REM": "#E06666", "Light": "#F6B26B", "Wake": "#CC0000"}
        has_stages = any(sleep_df[col].notna().any() for col in stage_cols if col in sleep_df.columns)
        if has_stages:
            render_section_header("Sleep Stages", "", "Rest architecture")
            stage_metric_cols = st.columns(4)
            stage_metric_defs = [
                ("Deep", "Average Deep", "Deep Trend"),
                ("REM", "Average REM", "REM Trend"),
                ("Light", "Average Light", "Light Trend"),
                ("Wake", "Average Wake", "Wake Trend"),
            ]
            for container, (col, avg_title, trend_title) in zip(stage_metric_cols, stage_metric_defs):
                with container:
                    render_fitbit_metric_stack(
                        sleep_df,
                        col,
                        avg_title,
                        trend_title,
                        lambda v: format_fitbit_metric_value(v, "min", 0),
                        lambda v: format_fitbit_metric_value(v, "min / month", 1, signed=True),
                        lambda v: format_fitbit_metric_value(v, "min / year", 1, signed=True),
                    )

            c1, c2 = st.columns(2)
            stage_items = [(col, color) for col, color in stage_cols.items() if col in sleep_df.columns and sleep_df[col].notna().any()]
            for i, (col, color) in enumerate(stage_items):
                with (c1 if i % 2 == 0 else c2):
                    fig_stage = plot_fitbit_timeseries(sleep_chart_df, col, col, "min", color=color, show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
                    fig_stage.update_layout(height=300)
                    render_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No sleep data available.")

    # ==================================================================
    # ACTIVITY SECTION
    # ==================================================================
    render_section_header("Activity", "", "Movement")
    if not activity_df.empty:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            render_fitbit_metric_stack(
                activity_df,
                "Steps",
                "Average Steps",
                "Steps Trend",
                lambda v: format_fitbit_metric_value(v, "", 0),
                lambda v: format_fitbit_metric_value(v, "steps / month", 0, signed=True),
                lambda v: format_fitbit_metric_value(v, "steps / year", 0, signed=True),
            )
        with a2:
            render_fitbit_metric_stack(
                activity_df,
                "ZoneMinutes",
                "Average Zone Minutes",
                "Zone Minutes Trend",
                lambda v: format_fitbit_metric_value(v, "min", 0),
                lambda v: format_fitbit_metric_value(v, "min / month", 1, signed=True),
                lambda v: format_fitbit_metric_value(v, "min / year", 1, signed=True),
            )
        with a3:
            render_fitbit_metric_stack(
                activity_df,
                "Distance",
                "Average Distance",
                "Distance Trend",
                lambda v: format_fitbit_metric_value(v, "km", 2),
                lambda v: format_fitbit_metric_value(v, "km / month", 2, signed=True),
                lambda v: format_fitbit_metric_value(v, "km / year", 1, signed=True),
            )
        with a4:
            render_fitbit_metric_stack(
                activity_df,
                "Calories",
                "Average Calories",
                "Calories Trend",
                lambda v: format_fitbit_metric_value(v, "kcal", 0),
                lambda v: format_fitbit_metric_value(v, "kcal / month", 0, signed=True),
                lambda v: format_fitbit_metric_value(v, "kcal / year", 0, signed=True),
            )

        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

        fig_steps = plot_fitbit_timeseries(activity_chart_df, "Steps", "Steps", "steps", color="#E07A5F", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_steps, use_container_width=True, config={"displayModeBar": False})

        if "ZoneMinutes" in activity_df.columns:
            fig_zm = plot_fitbit_timeseries(activity_chart_df, "ZoneMinutes", "Active Zone Minutes", "min", color="#81B29A", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
            render_chart(fig_zm, use_container_width=True, config={"displayModeBar": False})

        fig_dist = plot_fitbit_timeseries(activity_chart_df, "Distance", "Distance", "km", color="#7EB8DA", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

        fig_cal = plot_fitbit_timeseries(activity_chart_df, "Calories", "Calories", "kcal", color="#F2CC8F", show_trend=show_trend, show_primary_series=show_primary_fitbit_series, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None, show_title=True)
        render_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})

        zone_cols = {
            "MinutesFatBurn": ("#F6B26B", "Moderate"),
            "MinutesCardio": ("#E07A5F", "Vigorous"),
            "MinutesPeak": ("#CC0000", "Peak"),
        }
        has_zones = any(col in activity_df.columns and activity_df[col].notna().any() for col in zone_cols)
        if has_zones:
            render_section_header("Zone Minutes", "", "Heart rate zones")
            z1, z2, z3 = st.columns(3)
            zone_metric_defs = [
                ("MinutesFatBurn", "Average Moderate", "Moderate Trend"),
                ("MinutesCardio", "Average Vigorous", "Vigorous Trend"),
                ("MinutesPeak", "Average Peak", "Peak Trend"),
            ]
            for container, (col, avg_title, trend_title) in zip((z1, z2, z3), zone_metric_defs):
                with container:
                    render_fitbit_metric_stack(
                        activity_df,
                        col,
                        avg_title,
                        trend_title,
                        lambda v: format_fitbit_metric_value(v, "min", 0),
                        lambda v: format_fitbit_metric_value(v, "min / month", 1, signed=True),
                        lambda v: format_fitbit_metric_value(v, "min / year", 1, signed=True),
                    )

            zg1, zg2, zg3 = st.columns(3)
            zone_items = [(col, color, label) for col, (color, label) in zone_cols.items() if col in activity_df.columns and activity_df[col].notna().any()]
            for container, (col, color, label) in zip((zg1, zg2, zg3), zone_items):
                with container:
                    fig_zone = plot_fitbit_timeseries(
                        activity_chart_df,
                        col,
                        label,
                        "min",
                        color=color,
                        show_trend=show_trend,
                        show_primary_series=show_primary_fitbit_series,
                        date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None,
                        show_title=True,
                    )
                    fig_zone.update_layout(height=300)
                    render_chart(fig_zone, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No activity data available.")

    # ==================================================================
    # RAW DATA
    # ==================================================================
    render_section_header("Raw Data", "", "Audit trail")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Weight", "HRV", "RHR", "Breathing Rate", "Sleep", "Activity"])
    with tab1:
        if not weight_df.empty:
            st.dataframe(weight_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No data.")
    with tab2:
        if not hrv_df.empty:
            st.dataframe(hrv_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No data.")
    with tab3:
        if not rhr_df.empty:
            st.dataframe(rhr_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No data.")
    with tab4:
        if not br_df.empty:
            st.dataframe(br_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No data.")
    with tab5:
        if not sleep_df.empty:
            st.dataframe(sleep_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No data.")
    with tab6:
        if not activity_df.empty:
            st.dataframe(activity_df.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No data.")

    should_refresh = sync_mode is not None or auto_refresh
    if should_refresh:
        refreshed_results = {}
        with loading_placeholder.container():
            render_section_header(
                "Loading Fitbit",
                "",
                "Sync in progress",
            )
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for label, func_name, metric_name in fetch_steps:
                progress_text.caption(f"Fetching {label}...")
                refreshed_results[func_name], warning = fetch_fitbit_metric(
                    label, func_name, metric_name, force_full
                )
                if warning:
                    fetch_warnings.append(warning)
                progress_bar.progress((len(refreshed_results)) / len(fetch_steps))
            progress_text.caption("Fitbit data loaded.")
            progress_bar.progress(1.0)

        if sync_mode is not None:
            st.session_state["fitbit_refresh_notice"] = "Fitbit data refreshed."
            st.rerun()
        if auto_refresh and not fetch_warnings:
            st.session_state["fitbit_refresh_notice"] = "Fitbit data refreshed in the background."
            st.rerun()
        if fetch_warnings:
            loading_placeholder.empty()

    if PRINT_MODE:
        return
    render_section_header(
        "Export",
        "Print this view as a report or save it as a PDF.",
        "Share & archive",
    )
    render_print_button()


def page_lifts():
    """Hevy strength dashboard focused on session-best estimated 1RM."""
    render_page_hero(
        "Lifts",
        "Your key lifts from Hevy, translated into simple one-rep-max trend lines and strength-standard categories.",
        pills=["Five key lifts", "Estimated 1RM", "Strength standards"],
        eyebrow="Strength training",
    )

    if not hevy_client.has_api_key():
        st.warning("Hevy is not configured yet. Add `hevy_api_key.txt` to the project root.")
        st.stop()

    sync_mode = st.session_state.pop("hevy_sync_mode", None)
    force_refresh = sync_mode == "refresh"

    try:
        with st.spinner("Loading Hevy workouts..."):
            workouts, hevy_meta = hevy_client.get_workouts(force_refresh=force_refresh)
    except Exception as e:
        st.error(f"Failed to load Hevy workouts: {e}")
        st.stop()

    history_df = hevy_client.build_working_set_history(workouts)
    session_best_df = hevy_client.summarize_session_best(history_df)
    selected_source_titles = [
        title
        for item in LIFTS_PAGE_CONFIG
        for title in item["source_titles"]
    ]
    selected_sessions_df = session_best_df[
        session_best_df["exercise_title"].isin(selected_source_titles)
    ].copy()

    if selected_sessions_df.empty:
        st.info("No weighted Hevy sets with both load and reps were found yet.")
        st.stop()

    min_date = selected_sessions_df["workout_date"].min()
    max_date = selected_sessions_df["workout_date"].max()
    lifts_time_start, lifts_time_end = render_time_controls("lifts", min_date, max_date)
    end_inclusive = lifts_time_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered_sessions = selected_sessions_df[
        (selected_sessions_df["workout_start_time"] >= lifts_time_start) &
        (selected_sessions_df["workout_start_time"] <= end_inclusive)
    ].copy()

    if force_refresh and hevy_meta.get("source") == "live":
        st.success("Hevy data refreshed.")
    if hevy_meta.get("warning"):
        st.warning(hevy_meta["warning"])

    fetched_at = hevy_meta.get("fetched_at")
    if fetched_at is not None:
        source_label = "Live sync" if hevy_meta.get("source") == "live" else "Cached sync"
        st.caption(f"{source_label}: {format_display_date(fetched_at, fmt='%d %b %Y %H:%M')}")

    current_bodyweight_kg = get_latest_fitbit_weight_kg()
    if strength_standards.has_data() and current_bodyweight_kg is not None:
        st.caption(
            f"Strength standards use your latest cached weigh-in "
            f"({format_fitbit_metric_value(current_bodyweight_kg, 'kg', 1)}) and male thresholds."
        )

    stats_in_view = filtered_sessions if not filtered_sessions.empty else selected_sessions_df
    s1, s2 = st.columns(2)
    with s1:
        render_metric_card("Workouts", f"{len(workouts):,}")
    with s2:
        render_metric_card("Latest workout", format_display_date(stats_in_view["workout_start_time"].max()))

    if filtered_sessions.empty:
        st.info("No lift sessions fall inside the selected time window.")
        st.stop()

    for index, exercise_config in enumerate(LIFTS_PAGE_CONFIG):
        exercise_title = exercise_config["label"]
        source_titles = exercise_config["source_titles"]
        exercise_sessions = filtered_sessions[
            filtered_sessions["exercise_title"].isin(source_titles)
        ].sort_values("workout_start_time")
        exercise_all_sessions = selected_sessions_df[
            selected_sessions_df["exercise_title"].isin(source_titles)
        ].sort_values("workout_start_time")

        render_section_header(
            exercise_title,
            "",
            "Strength trend",
        )

        best_1rm = exercise_sessions["estimated_1rm_kg"].max() if not exercise_sessions.empty else np.nan
        if exercise_all_sessions.empty:
            st.caption("No logged data found for this lift yet.")
            continue
        if exercise_sessions.empty:
            st.caption("No logged sessions for this lift in the selected time window.")
            continue

        standards_thresholds = None
        if current_bodyweight_kg is not None:
            standards_thresholds = strength_standards.get_thresholds(
                source_titles,
                current_bodyweight_kg,
                gender=STRENGTH_STANDARDS_GENDER,
            )

        recent_cutoff = exercise_all_sessions["workout_start_time"].max() - pd.Timedelta(days=90)
        recent_trend_df = exercise_all_sessions[
            exercise_all_sessions["workout_start_time"] >= recent_cutoff
        ][["workout_start_time", "estimated_1rm_kg"]].rename(
            columns={"workout_start_time": "Date", "estimated_1rm_kg": "Value"}
        )
        trend_per_month = compute_lift_trend_per_month(recent_trend_df, "Value")
        trend_display = (
            format_fitbit_metric_value(trend_per_month, "kg / month", 1, signed=True)
            if trend_per_month is not None
            else "—"
        )

        strength_classification = (
            strength_standards.classify_1rm(best_1rm, standards_thresholds)
            if standards_thresholds is not None and pd.notna(best_1rm)
            else None
        )
        if strength_classification:
            category_value = str(strength_classification["category"])
            category_subtitle = format_strength_gap_to_next(strength_classification, trend_per_month)
        elif current_bodyweight_kg is None:
            category_value = "—"
            category_subtitle = "Needs Fitbit weight data"
        else:
            category_value = "—"
            category_subtitle = "No standards available"

        one_rep_col, trend_col, category_col = st.columns(3)
        with one_rep_col:
            render_metric_card(
                "One Rep Max",
                format_fitbit_metric_value(best_1rm, "kg", 1) if pd.notna(best_1rm) else "—",
            )
        with trend_col:
            render_metric_card(
                "Trend (90 days)",
                trend_display,
            )
        with category_col:
            render_metric_card(
                "Current Category",
                category_value,
                latest_text=category_subtitle,
            )
        st.markdown("<div style='margin-bottom: 0.35rem;'></div>", unsafe_allow_html=True)
        fig = plot_lift_timeseries(
            exercise_sessions,
            exercise_title,
            color=LIFT_SERIES_COLOR,
            date_window=(lifts_time_start, lifts_time_end),
            standards_thresholds=standards_thresholds,
            strength_classification=strength_classification,
        )
        render_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if PRINT_MODE:
        return
    render_section_header(
        "Export",
        "Print this view as a report or save it as a PDF.",
        "Share & archive",
    )
    render_print_button()


def page_settings():
    """Settings page for data sources and Fitbit configuration."""
    render_page_hero(
        "Settings",
        "Manage the technical setup for data sources, local credentials, and Fitbit connectivity.",
        pills=["Sources", "Connection", "Maintenance"],
        eyebrow="Configuration",
    )

    local_key_path = os.path.join(os.path.dirname(__file__), "sheet_api_key.json")
    default_sheet_url = "https://docs.google.com/spreadsheets/d/1pfYaK6t25gcKdBAUu8_geyGlQP6wp6px9IyNUKI4wdw/edit?usp=sharing"
    spreadsheet_id = parse_spreadsheet_id(default_sheet_url)
    config = fitbit_client.load_config()
    fitbit_connected = fitbit_client.has_valid_token()

    render_section_header("Data Source", "Technical configuration for the Google Sheets source used by the blood panel dashboard.", "Sources")
    if os.path.exists(local_key_path):
        st.success("Local Google Sheets key detected.")
        st.caption(f"Using local key: `{local_key_path}`")
    else:
        st.info("No local Google Sheets key detected. The blood panel page will prompt for a service account file.")
    st.caption(f"Default spreadsheet ID: `{spreadsheet_id}`")
    if st.button("Force refresh Google Sheets"):
        load_from_gsheets.clear()
        st.success("Google Sheets cache cleared. The next Blood Panel visit will fetch fresh data.")

    render_section_header("Hevy", "Connection health and maintenance controls for the Lifts page.", "Sources")
    hevy_key_path = hevy_client.get_api_key_path()
    hevy_cache_updated = hevy_client.cache_last_updated()
    if hevy_key_path:
        st.success("Hevy API key detected.")
        st.caption(f"Using local key: `{hevy_key_path}`")
    else:
        st.info("No local Hevy API key detected yet. Add `hevy_api_key.txt` to enable the Lifts page.")
    if hevy_cache_updated is not None:
        st.caption(f"Cached Hevy workouts last updated: {format_display_date(hevy_cache_updated, fmt='%Y-%m-%d %H:%M')}")
    hevy_col1, hevy_col2 = st.columns(2)
    with hevy_col1:
        if st.button("Sync latest Hevy data"):
            st.session_state["hevy_sync_mode"] = "refresh"
            st.success("The next Lifts visit will refresh your Hevy data.")
    with hevy_col2:
        if st.button("Clear Hevy cache"):
            hevy_client.clear_cache()
            st.success("Hevy cache cleared.")

    # Garmin (weight source since the scale stopped reaching the Fitbit API)
    render_section_header(
        "Garmin",
        "Weigh-ins from the Garmin scale. Garmin Connect is the live weight source; "
        "the Fitbit weight history (including manual entries) is kept and merged in.",
        "Sources",
    )
    if garmin_client.is_configured():
        st.success("Connected to Garmin Connect")
        garmin_weight_df = garmin_client.load_cached_dataframe()
        if not garmin_weight_df.empty:
            last_garmin = pd.to_datetime(garmin_weight_df["Date"]).max()
            st.caption(
                f"{len(garmin_weight_df)} cached weigh-ins, latest "
                f"{format_display_date(last_garmin)}"
            )
        garmin_col1, garmin_col2 = st.columns(2)
        with garmin_col1:
            if st.button("Sync Garmin weight now"):
                try:
                    with st.spinner("Fetching Garmin weigh-ins..."):
                        garmin_df = garmin_client.fetch_weight()
                    st.success(f"Garmin weight synced ({len(garmin_df)} weigh-ins).")
                except Exception as e:
                    st.error(f"Garmin sync failed: {e}")
        with garmin_col2:
            if st.button("Full Garmin re-pull"):
                try:
                    with st.spinner("Re-fetching full Garmin weight history..."):
                        garmin_df = garmin_client.fetch_weight(force_full=True)
                    st.success(f"Garmin history re-pulled ({len(garmin_df)} weigh-ins).")
                except Exception as e:
                    st.error(f"Garmin full re-pull failed: {e}")
    else:
        st.info(
            "Not connected. Run this once in a terminal, then reload this page "
            "(your password stays in the terminal, it is never stored):"
        )
        st.code('.venv/bin/python garmin_login.py', language="bash")

    # Status
    render_section_header("Connection Status", "A quick health check of your local Fitbit connection and token state.", "Setup")
    if fitbit_connected:
        st.success("Connected to Fitbit")
        st.caption(f"User ID: `{config.get('user_id', 'unknown')}`")
        from datetime import datetime as dt
        expires = config.get("expires_at", 0)
        if expires:
            st.caption(f"Token expires: {dt.fromtimestamp(expires).strftime('%Y-%m-%d %H:%M')}")
    elif fitbit_client.is_configured():
        st.warning("Fitbit is configured but needs (re-)authorization.")
    else:
        st.info("Not configured. Follow the steps below to connect.")

    # Step 1: Client ID
    register_container = st.expander("Register A Fitbit App", expanded=not fitbit_connected)
    with register_container:
        st.caption("Create a personal Fitbit app once, then use it as the secure bridge for this dashboard.")
        st.markdown("""
1. Go to [dev.fitbit.com/apps/new](https://dev.fitbit.com/apps/new)
2. Fill in the form:
   - **OAuth 2.0 Application Type**: Personal
   - **Redirect URL**: `http://localhost:8501`
   - **Default Access Type**: Read Only
3. Copy the **OAuth 2.0 Client ID** below.
""")

        client_id = st.text_input("Client ID", value=config.get("client_id", ""), type="default")

        if client_id:
            # Save client_id
            if client_id != config.get("client_id"):
                config["client_id"] = client_id
                fitbit_client.save_config(config)
                st.success("Client ID saved.")

            # Step 2: Authorize
            render_section_header("Authorize", "Generate an approval link and complete the OAuth handshake in the browser.", "Step 2")

            redirect_uri = "http://localhost:8501"

            # Persist code_verifier to disk so it survives page reloads and redirects
            stored_verifier = config.get("code_verifier")
            if stored_verifier:
                # Rebuild auth URL with the stored verifier's challenge
                import hashlib, base64
                digest = hashlib.sha256(stored_verifier.encode("ascii")).digest()
                challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
                code_verifier, code_challenge = stored_verifier, challenge
            else:
                code_verifier, code_challenge = fitbit_client.generate_pkce_pair()
                config["code_verifier"] = code_verifier
                fitbit_client.save_config(config)

            auth_params = urllib.parse.urlencode({
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": "profile weight heartrate respiratory_rate sleep activity",
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "prompt": "consent",
            })
            auth_url = f"https://www.fitbit.com/oauth2/authorize?{auth_params}"

            st.markdown(f"[Click here to authorize with Fitbit]({auth_url})")
            st.caption("After authorizing, you'll be redirected back and connected automatically.")

            if st.button("Generate new auth link"):
                config.pop("code_verifier", None)
                fitbit_client.save_config(config)
                st.rerun()

    # Disconnect
    render_section_header("Manage Connection", "Clear local cache or disconnect the integration when you want a clean reset.", "Maintenance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Sync latest Fitbit data"):
            st.session_state["fitbit_sync_mode"] = "incremental"
            st.success("The next Fitbit Data visit will refresh recent Fitbit data.")
    with col2:
        if st.button("Full Fitbit re-sync"):
            st.session_state["fitbit_sync_mode"] = "full"
            st.success("The next Fitbit Data visit will run a full Fitbit history sync.")
    with col3:
        if st.button("Clear cached data"):
            fitbit_client.clear_cache()
            st.success("Cache cleared.")
    with col4:
        if st.button("Disconnect Fitbit"):
            fitbit_client.disconnect()
            st.success("Disconnected. Config and cache removed.")
            st.rerun()


# =====================================================================
# MAIN APP — page config, global CSS, sidebar nav
# =====================================================================

st.set_page_config(
    page_title="Biomarker Studio",
    page_icon=str(Path(__file__).with_name("app_icon.png")),
    layout="wide",
)

# ---- Print mode ----
# A separate, denser print layout, triggered by the in-page "Print" button in
# each view's Share & archive block (render_print_button). The button sets a
# one-shot session flag and reruns IN-SESSION (no page reload), so all sidebar
# state — page, category, time range — is preserved. In print mode charts are
# rebuilt shorter (render_chart) so several rows fit per printed page; the
# normal on-screen app is unchanged. Popped here so the print render is one-shot
# (the next rerun, triggered after the dialog, renders the normal view).
PRINT_MODE = bool(st.session_state.pop("_print_mode", False))

# Charts shrink to this fraction of their on-screen height in print mode. At
# ~0.6, a 450px chart becomes ~270px, so three chart rows fit one portrait page.
PRINT_CHART_SCALE = 0.6


def render_chart(fig, **kwargs):
    """st.plotly_chart, but shrinks tall charts when rendering for print."""
    if PRINT_MODE:
        try:
            h = fig.layout.height
            if h and h >= 200:        # leave sparklines/heatmap-cells alone
                fig.update_layout(height=int(h * PRINT_CHART_SCALE))
        except Exception:
            pass
    return st.plotly_chart(fig, **kwargs)


# ---- Global CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --bg-top: #f4efe7;
    --bg-bottom: #e9ece7;
    --panel: rgba(255, 251, 247, 0.88);
    --panel-strong: rgba(255, 255, 255, 0.94);
    --panel-border: rgba(38, 57, 53, 0.1);
    --ink: #18322f;
    --muted: #667874;
    --accent: #d96b42;
    --accent-dark: #b65433;
    --accent-soft: rgba(217, 107, 66, 0.12);
    --moss: #7a9d8d;
    --moss-soft: rgba(122, 157, 141, 0.18);
    --line: rgba(24, 50, 47, 0.08);
    --shadow: 0 18px 60px rgba(21, 38, 35, 0.08);
    --radius-xl: 28px;
    --radius-lg: 22px;
    --radius-md: 16px;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at top left, rgba(217, 107, 66, 0.14), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(122, 157, 141, 0.18), transparent 24%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
}

.block-container {
    max-width: 1460px;
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(251, 245, 237, 0.96) 0%, rgba(242, 236, 226, 0.98) 100%);
    border-right: 1px solid rgba(24, 50, 47, 0.08);
}

section[data-testid="stSidebar"] > div {
    background: transparent;
}

section[data-testid="stSidebar"] * {
    color: var(--ink);
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: var(--muted) !important;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stFileUploader section {
    background: rgba(255, 255, 255, 0.72) !important;
    border: 1px solid rgba(24, 50, 47, 0.1) !important;
    border-radius: 16px !important;
}

section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: rgba(122, 157, 141, 0.16) !important;
    border: 1px solid rgba(122, 157, 141, 0.26) !important;
    border-radius: 999px !important;
}

section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] * {
    color: #35524c !important;
    -webkit-text-fill-color: #35524c !important;
}

section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] svg {
    fill: #35524c !important;
}

/* ---- Sidebar navigation menu (the page radio, restyled as a nav list) ---- */
section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label {
    display: flex;
    align-items: center;
    width: 100%;
    margin: 0 !important;
    padding: 0.55rem 0.85rem !important;
    border-radius: 12px !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer;
    transition: background 0.15s ease;
}

/* hide the radio dot */
section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label > div:first-child {
    display: none !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label p {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    transition: color 0.15s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label:hover {
    background: rgba(24, 50, 47, 0.06) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label:hover p {
    color: var(--ink) !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label:has(input:checked) {
    background: var(--accent-soft) !important;
    box-shadow: inset 3px 0 0 var(--accent);
}

section[data-testid="stSidebar"] div[role="radiogroup"][aria-label="Navigation"] > label:has(input:checked) p {
    color: var(--accent-dark) !important;
    font-weight: 800 !important;
}

/* While a page-switch rerun is in flight (class set by inject_page_switch_cleaner),
   hide the previous page's elements instead of leaving them stacked with the new
   ones until the run completes. Streamlit marks leaf elements data-stale at run
   start; block chrome (expanders, tabs) is tagged bp-stale-block by the cleaner. */
body.bp-page-switching [data-stale="true"],
body.bp-page-switching .bp-stale-block {
    display: none !important;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(24, 50, 47, 0.08);
    margin: 1.1rem 0 1.25rem;
}

.sidebar-brand {
    padding: 0.25rem 0 0.35rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--ink);
}

.sidebar-brand-subtitle {
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.15rem;
}

.section-header {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

.fitbit-metric-card {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(250, 246, 240, 0.92) 100%);
    border: 1px solid rgba(24, 50, 47, 0.08);
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(21, 38, 35, 0.06);
    padding: 0.95rem 1rem 0.85rem;
    min-height: 120px;
    margin-bottom: 0.7rem;
}

.fitbit-metric-card--trend {
    background: linear-gradient(180deg, rgba(244, 240, 234, 0.82) 0%, rgba(239, 235, 229, 0.86) 100%);
    border-color: rgba(24, 50, 47, 0.05);
    box-shadow: none;
    min-height: 96px;
    padding: 0.8rem 0.95rem 0.72rem;
}

.fitbit-metric-title {
    font-size: 0.86rem;
    font-weight: 700;
    color: var(--muted);
    letter-spacing: -0.01em;
    margin-bottom: 0.45rem;
}

.fitbit-metric-value {
    font-size: 1.75rem;
    line-height: 1.1;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.04em;
}

.fitbit-metric-card--trend .fitbit-metric-title {
    font-size: 0.8rem;
    color: #7a7f8c;
}

.fitbit-metric-card--trend .fitbit-metric-value {
    font-size: 1.32rem;
    font-weight: 700;
    color: #35524c;
}

.fitbit-metric-latest {
    margin-top: 0.55rem;
    font-size: 0.82rem;
    color: #7a7f8c;
    line-height: 1.35;
}

.fitbit-metric-card--trend .fitbit-metric-latest {
    margin-top: 0.4rem;
    font-size: 0.76rem;
    color: #8a8f9b;
}

.dexa-summary-card {
    min-height: 138px;
    margin-bottom: 0.9rem;
    padding: 1rem 1.05rem 0.95rem;
    border-radius: 18px;
    border: 1px solid rgba(24, 50, 47, 0.08);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.9) 0%, rgba(250, 246, 240, 0.92) 100%);
    box-shadow: 0 10px 28px rgba(21, 38, 35, 0.06);
}

.dexa-summary-card--good {
    background: linear-gradient(180deg, rgba(236, 248, 241, 0.94) 0%, rgba(218, 237, 226, 0.92) 100%);
    border-color: rgba(129, 178, 154, 0.32);
}

.dexa-summary-card--caution {
    background: linear-gradient(180deg, rgba(255, 248, 230, 0.95) 0%, rgba(248, 232, 194, 0.92) 100%);
    border-color: rgba(242, 204, 143, 0.48);
}

.dexa-summary-card--alert {
    background: linear-gradient(180deg, rgba(255, 238, 232, 0.96) 0%, rgba(248, 217, 207, 0.92) 100%);
    border-color: rgba(224, 122, 95, 0.38);
}

.dexa-summary-subsection {
    margin: 1.05rem 0 0.55rem;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

.dexa-summary-label {
    min-height: 2.25rem;
    font-size: 0.78rem;
    line-height: 1.25;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
}

.dexa-summary-value {
    margin-top: 0.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.35rem, 1.5vw, 1.9rem);
    line-height: 1.05;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--ink);
    overflow-wrap: anywhere;
}

.dexa-summary-footnote {
    margin-top: 0.55rem;
    font-size: 0.78rem;
    line-height: 1.35;
    color: #7a7f8c;
}

.page-hero {
    position: relative;
    overflow: hidden;
    padding: 2rem 2.2rem;
    margin-bottom: 1.25rem;
    border-radius: var(--radius-xl);
    border: 1px solid rgba(255, 255, 255, 0.65);
    background:
        linear-gradient(135deg, rgba(255,255,255,0.82) 0%, rgba(255,246,239,0.84) 52%, rgba(242,247,244,0.86) 100%);
    box-shadow: var(--shadow);
}

.page-hero-copy {
    position: relative;
    z-index: 2;
    max-width: 760px;
}

.eyebrow,
.section-eyebrow {
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
}

.page-hero h1 {
    margin: 0.45rem 0 0.45rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.4rem, 4vw, 3.8rem);
    line-height: 0.96;
    letter-spacing: -0.05em;
    color: var(--ink);
}

.page-hero p {
    max-width: 640px;
    margin: 0;
    font-size: 1.02rem;
    line-height: 1.65;
    color: var(--muted);
}

.context-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin-top: 1.15rem;
}

.context-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.52rem 0.9rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.66);
    border: 1px solid rgba(24, 50, 47, 0.08);
    color: var(--ink);
    font-size: 0.88rem;
    font-weight: 600;
}

.hero-orb {
    position: absolute;
    border-radius: 999px;
    filter: blur(2px);
}

.hero-orb-a {
    width: 260px;
    height: 260px;
    right: -80px;
    top: -80px;
    background: radial-gradient(circle, rgba(217, 107, 66, 0.28), rgba(217, 107, 66, 0));
}

.hero-orb-b {
    width: 220px;
    height: 220px;
    right: 120px;
    bottom: -120px;
    background: radial-gradient(circle, rgba(122, 157, 141, 0.24), rgba(122, 157, 141, 0));
}

.hero-nav-slot {
    margin-top: -3.9rem;
    margin-left: 2.2rem;
    position: relative;
    z-index: 4;
}

.hero-nav-slot .stButton > button {
    min-height: 2.2rem;
    padding: 0.4rem 0.8rem !important;
    border-radius: 999px !important;
    font-size: 0.88rem !important;
    box-shadow: 0 8px 18px rgba(217, 107, 66, 0.16);
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.6rem;
}

.stat-card {
    padding: 1.15rem 1.2rem 1.1rem;
    border-radius: var(--radius-lg);
    background: var(--panel);
    border: 1px solid var(--panel-border);
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
}

.stat-label {
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

.stat-value {
    margin-top: 0.35rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.7rem, 2vw, 2.4rem);
    line-height: 1;
    letter-spacing: -0.05em;
    color: var(--ink);
}

.stat-footnote {
    margin-top: 0.55rem;
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.45;
}

.section-shell {
    margin: 1.7rem 0 0.9rem;
}

.section-title {
    margin-top: 0.22rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: var(--ink);
}

.section-shell p {
    max-width: 760px;
    margin: 0.28rem 0 0;
    color: var(--muted);
    line-height: 1.55;
}

.chart-card-title {
    margin: 0.2rem 0 0.45rem;
    padding: 0 0.15rem;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ink);
}

.chart-card-meta {
    margin: -0.15rem 0 0.5rem;
    padding: 0 0.15rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #7a7f8c;
    letter-spacing: -0.01em;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.9rem;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: #fffdfb !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.1rem !important;
    box-shadow: 0 12px 28px rgba(217, 107, 66, 0.22);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    filter: saturate(1.05);
    box-shadow: 0 14px 30px rgba(217, 107, 66, 0.26);
}

.stTextInput input,
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stDateInput input,
.stFileUploader section {
    background: rgba(255, 255, 255, 0.82) !important;
    border: 1px solid rgba(24, 50, 47, 0.1) !important;
    border-radius: 16px !important;
}

.stSlider [data-baseweb="slider"] > div:first-child > div {
    background: var(--accent) !important;
}

section[data-testid="stSidebar"] .stSlider {
    padding: 0 0.35rem;
}

section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {
    display: none !important;
}

.time-stop-row {
    position: relative;
    height: 1rem;
    margin: 0.1rem 0.35rem -0.9rem;
    font-size: 0.72rem;
}

.time-stop-row .time-stop {
    position: absolute;
    top: 0;
    color: var(--muted);
    letter-spacing: 0.02em;
    white-space: nowrap;
}

.time-stop-row .time-stop.active {
    color: var(--accent);
    font-weight: 700;
}

section[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: #FFFFFF !important;
    border: 1px solid rgba(24, 50, 47, 0.22) !important;
    box-shadow: 0 2px 6px rgba(24, 50, 47, 0.22) !important;
    width: 16px !important;
    height: 16px !important;
}

div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"],
div[data-testid="stMetric"] {
    border-radius: var(--radius-lg);
    border: 1px solid var(--panel-border);
    overflow: hidden;
    box-shadow: var(--shadow);
    background: var(--panel-strong);
}

div[data-testid="stPlotlyChart"] {
    padding: 0.4rem 0.45rem 0.2rem;
}

div[data-testid="stDataFrame"] {
    backdrop-filter: blur(10px);
}

div[data-testid="stMetric"] {
    padding: 1rem 1.15rem !important;
    border-left: none;
}

div[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    letter-spacing: -0.04em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.55rem 1rem;
    background: rgba(255, 255, 255, 0.58);
}

.stTabs [aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: var(--ink) !important;
}

.stAlert {
    border-radius: 18px;
    border: 1px solid rgba(24, 50, 47, 0.08);
    box-shadow: var(--shadow);
}

.stCaption {
    color: var(--muted);
}

@media (max-width: 980px) {
    .hero-nav-slot {
        margin-top: -2.2rem;
        margin-left: 1.2rem;
    }

    .stat-grid {
        grid-template-columns: 1fr;
    }

    .page-hero {
        padding: 1.4rem 1.2rem;
    }

    .page-hero h1 {
        font-size: 2.3rem;
    }
}

/* The print runner's restore button must never be visible (the runner clicks it
   to return to the normal view). Hide the sentinel's container and park the
   button's container off-screen — off-screen, not display:none, so .click() and
   textContent stay reliable. */
[data-testid="element-container"]:has(#bm-restore-anchor) {
    display: none !important;
}
[data-testid="element-container"]:has(#bm-restore-anchor) + [data-testid="element-container"] {
    position: absolute !important;
    left: -10000px !important;
    top: 0 !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
}

/* ---- Print styles: hide chrome, lay out the main view for paper ---- */
@media print {
    /* Force background/panel colors to render on paper instead of being
       stripped by the browser's ink-saving default. Also strip backdrop-filter
       everywhere: Chrome's print-to-PDF can't rasterize it and leaves grey
       boxes behind the cards/table (visible in Preview/Quick Look though the
       on-screen print preview composites it fine). */
    * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        box-shadow: none !important;   /* large soft shadows also halo in PDF */
    }

    /* Hide on-screen chrome and every interactive control. (.hero-nav-slot is
       just an empty marker — Streamlit renders Prev/Next as sibling stButton
       blocks — so the buttons are hidden directly. #biomarker-print-btn is the
       floating Print button injected by inject_print_button.) */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"],
    div[data-testid="stSidebarCollapsedControl"],
    div[data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"],
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stStatusWidget"],
    div[data-testid="stButton"],
    div[data-testid="stDownloadButton"],
    div[data-testid="stLinkButton"],
    .stButton,
    .hero-nav-slot,
    #biomarker-print-btn {
        display: none !important;
    }

    /* Flatten the heavy gradient backdrop to plain white for clean paper. */
    html, body, .stApp,
    section[data-testid="stMain"],
    div[data-testid="stAppViewContainer"] {
        background: #ffffff !important;
    }

    section[data-testid="stMain"],
    div[data-testid="stAppViewContainer"] {
        margin-left: 0 !important;
        padding-left: 0 !important;
        width: 100% !important;
    }

    /* Pin the content to a fixed page-width canvas (680px fits A4/Letter
       portrait with the @page margin below) and KEEP the on-screen
       multi-column layout. The Print button re-renders the charts to this
       width first, so two/three charts sit across the page exactly like on
       screen — no stacking, no clipping. */
    .block-container {
        width: 680px !important;
        max-width: 680px !important;
        margin: 0 auto !important;
        padding: 0.5rem 0 1rem !important;
        /* clip phantom Plotly overflow WITHOUT the scrollbar that overflow-x:
           hidden induces (hidden forces overflow-y to auto). */
        overflow-x: clip !important;
    }

    /* The on-screen scroll container must flow its content across printed pages
       instead of clipping it to one viewport (and printing a scrollbar). */
    section[data-testid="stMain"],
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMainBlockContainer"] {
        overflow: visible !important;
    }

    /* Streamlit wraps everything in flex containers, and Chrome won't honour
       break-inside:avoid on flex items — so chart rows get sliced across page
       breaks. Make the vertical stacking wrappers block so normal pagination
       (and the break-inside rules below) apply. Horizontal blocks stay flex to
       keep the columns side by side. */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
        display: block !important;
    }

    /* Tighten the inter-column gap so 3 charts comfortably fit 680px. */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.5rem !important;
    }

    /* Drop the title background box entirely for print. The on-screen hero
       (gradient + blurred orbs + shadow + border-radius/overflow:hidden) paints
       its filled background SHIFTED off the title in Chrome's print-to-PDF when
       it sits in an st.empty() placeholder (Blood Panel / Dexa) — a Chrome PDF
       compositing quirk I couldn't reliably reproduce in isolation. With no
       background box there is nothing to displace: the title prints as plain
       bold text (always positioned correctly), aligned with the content below. */
    .page-hero {
        position: static !important;
        overflow: visible !important;
        border: none !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0.25rem 0 0.5rem !important;
        margin-bottom: 0.6rem !important;
    }
    .hero-orb {
        display: none !important;
    }

    div[data-testid="stFullScreenFrame"],
    .stPlotlyChart,
    .js-plotly-plot,
    .plot-container,
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        max-width: 100% !important;
    }

    .stat-grid {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
        width: 100% !important;
    }

    /* st.metric cards default to a big nowrap value that gets clipped to an
       ellipsis ("02 Apr…") in the narrower print columns. Shrink it a touch and
       let both label and value wrap so the full text shows. */
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    /* Keep a chart row / card whole rather than splitting it across pages. */
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stColumn"],
    div[data-testid="column"],
    div[data-testid="stFullScreenFrame"],
    .stPlotlyChart,
    .js-plotly-plot,
    .page-hero,
    .metric-card,
    .stat-card,
    .context-chip-row,
    .dexa-summary-subsection,
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stTable"],
    div[data-testid="stDataFrame"],
    div[data-testid="stMetric"] {
        break-inside: avoid;
        page-break-inside: avoid;
    }

    /* Plotly draws its own toolbar on hover; never let it print. */
    .modebar-container {
        display: none !important;
    }
}

/* Page margins for the printout. 12mm leaves a printable width of ~703px (A4)
   / ~726px (Letter), which the 680px canvas above sits comfortably inside. */
@page {
    margin: 12mm;
}

/* "Print prep" — the Print button switches this class on (on the live page)
   BEFORE opening the print dialog. It reflows the page to the same fixed
   canvas so Streamlit's container-width Plotly charts redraw to the print
   width while still on screen, where the redraw reliably completes. The
   @media print rules above then carry that exact layout onto paper. */
body.biomarker-printing section[data-testid="stSidebar"],
body.biomarker-printing div[data-testid="stSidebarCollapsedControl"],
body.biomarker-printing div[data-testid="collapsedControl"],
body.biomarker-printing button[data-testid="baseButton-headerNoPadding"],
body.biomarker-printing header[data-testid="stHeader"],
body.biomarker-printing div[data-testid="stButton"],
body.biomarker-printing div[data-testid="stDownloadButton"],
body.biomarker-printing div[data-testid="stLinkButton"],
body.biomarker-printing .stButton,
body.biomarker-printing .hero-nav-slot,
body.biomarker-printing #biomarker-print-btn {
    display: none !important;
}
/* Strip backdrop-filter (and soft shadows) in print prep too — they rasterize
   to grey boxes/halos in the saved PDF (see the @media print note). */
body.biomarker-printing * {
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    box-shadow: none !important;
}
body.biomarker-printing .block-container {
    width: 680px !important;
    max-width: 680px !important;
    margin: 0 auto !important;
    /* Zero the ~80px Streamlit side padding so the columns get the full 680px
       — must match the @media print rule, or the charts redraw to the wrong
       width during prep and keep it when printed. */
    padding: 0.5rem 0 1rem !important;
    overflow-x: clip !important;
}
body.biomarker-printing .page-hero {
    position: static !important;
    overflow: visible !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0.25rem 0 0.5rem !important;
    margin-bottom: 0.6rem !important;
}
body.biomarker-printing .hero-orb {
    display: none !important;
}
body.biomarker-printing div[data-testid="stVerticalBlock"],
body.biomarker-printing div[data-testid="stVerticalBlockBorderWrapper"] {
    display: block !important;
}
body.biomarker-printing div[data-testid="stHorizontalBlock"] {
    gap: 0.5rem !important;
}
body.biomarker-printing div[data-testid="stFullScreenFrame"],
body.biomarker-printing .stPlotlyChart,
body.biomarker-printing .js-plotly-plot,
body.biomarker-printing .plot-container,
body.biomarker-printing div[data-testid="stDataFrame"] {
    max-width: 100% !important;
}
body.biomarker-printing .stat-grid {
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
    width: 100% !important;
}
body.biomarker-printing div[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
body.biomarker-printing div[data-testid="stMetricValue"] > div,
body.biomarker-printing div[data-testid="stMetricLabel"],
body.biomarker-printing div[data-testid="stMetricLabel"] * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar navigation ----
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">Personal Biomarker Studio</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Lifts", "Fitbit Data", "Blood Panel", "Dexa", "Settings"],
        label_visibility="collapsed",
        key="page_nav",
    )

# ---- Handle Fitbit OAuth redirect (before page routing) ----
_oauth_code = st.query_params.get("code")
if _oauth_code:
    _config = fitbit_client.load_config()
    if _config.get("code_verifier") and _config.get("client_id"):
        try:
            fitbit_client.exchange_code_for_token(
                _config["client_id"],
                _oauth_code,
                _config["code_verifier"],
                "http://localhost:8501",
            )
            _config = fitbit_client.load_config()
            _config.pop("code_verifier", None)
            fitbit_client.save_config(_config)
            st.query_params.clear()
            st.rerun()
        except Exception as _e:
            st.error(f"Fitbit authorization failed: {_e}")
            st.query_params.clear()

# ---- Reset scroll on page switch (runs before content renders) ----
_prev_page = st.session_state.get("_active_page")
st.session_state["_active_page"] = page
if _prev_page is not None and _prev_page != page:
    st.markdown(
        "<script>window.parent.document.querySelector('section.main').scrollTop=0;</script>",
        unsafe_allow_html=True,
    )

# ---- Route to selected page ----
page_root = st.empty()
with page_root.container():
    if page == "Blood Panel":
        page_blood_panel()
    elif page == "Dexa":
        page_dexa()
    elif page == "Fitbit Data":
        page_fitbit_data()
    elif page == "Lifts":
        page_lifts()
    elif page == "Settings":
        page_settings()

inject_page_switch_cleaner()
inject_sidebar_scroll_restorer(page)
inject_main_scroll_restorer(page)
if PRINT_MODE:
    inject_print_runner()
else:
    # Safety net: clear any lingering print-layout class so the normal view is
    # never stuck with the sidebar hidden / 680px canvas.
    components.html(
        "<script>try{window.parent.document.body.classList.remove('biomarker-printing');}catch(e){}</script>",
        height=0,
        width=0,
    )
