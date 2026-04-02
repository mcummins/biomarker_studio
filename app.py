
import os
import io
import json
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

# -----------------------------
# Config
# -----------------------------
EXCLUDE_SHEETS = {"All Data", "Optimal Ranges", "Graphs", "Labs and notes", "NN Metabolic Scorecard"}
DEFAULT_GROUP_SHEETS = []  # will be filled dynamically
TIME_PRESETS = ["30 days", "90 days", "1 year", "All time", "Custom"]
LAST_PLOTLY_ZOOM_EVENT_KEY = "_last_plotly_zoom_event_id"
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
    return long

def normalize_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # Keep essential columns if present
    keep = [c for c in ["Test","Unit","Optimal Range (lower)","Optimal Range (borderline)","Optimal Range (upper)"] if c in df2.columns]
    df2 = df2[keep]
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

    if policy == "intersection":
        lower_agg = lambda s: np.nanmax(pd.to_numeric(s, errors="coerce"))
        upper_agg = lambda s: np.nanmin(pd.to_numeric(s, errors="coerce"))
    else:  # "union" (default)
        lower_agg = lambda s: np.nanmin(pd.to_numeric(s, errors="coerce"))
        upper_agg = lambda s: np.nanmax(pd.to_numeric(s, errors="coerce"))

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


def plot_single_test(df: pd.DataFrame, test: str,
                     show_ref: bool=True, show_regression: bool=False,
                     show_zones: bool=True, range_policy: str="union",
                     date_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> go.Figure:
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

    # ---- background zones (green/orange/red) ----
    if show_zones and (pd.notna(lower) or pd.notna(upper)):
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

        x0, x1 = x.min(), x.max()

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
    if show_ref:
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
                           date_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> go.Figure:
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if date_window is not None:
        fig.update_xaxes(range=[pd.to_datetime(date_window[0]), pd.to_datetime(date_window[1])])
    apply_warm_theme(fig)
    return fig


def format_display_date(value, fmt: str = "%d %b %Y", empty: str = "No data") -> str:
    if pd.isna(value):
        return empty
    return pd.to_datetime(value).strftime(fmt)


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
    preset = st.sidebar.radio("Time window", TIME_PRESETS, key=preset_key, label_visibility="collapsed")

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
    notes = sheets.get("Labs and notes")
    if all_data is None or ranges is None:
        st.error("Expected sheets 'All Data' and 'Optimal Ranges' not found.")
        st.stop()

    # Normalize
    long = normalize_all_data(all_data)
    ranges_n = normalize_ranges(ranges)
    merged = attach_ranges(long, ranges_n)
    if isinstance(notes, pd.DataFrame):
        merged = attach_lab_notes(merged, notes)

    # Status & z-scores
    merged["status"] = merged.apply(lambda r: status_from_bounds(r["Value"], r.get("lower"), r.get("upper")), axis=1)
    merged["z"] = merged.apply(lambda r: compute_zscore(r["Value"], r.get("lower"), r.get("upper")), axis=1)

    # Groups
    groups = load_groups_from_sheets(sheets)
    group_names = ["(All)"] + list(groups.keys())
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
    if grp != "(All)":
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
        "Small-multiple trend cards make it easy to compare movement, range position, and momentum at a glance.",
        "Visual explorer",
    )
    if tests_selected:
        ncols = 3
        rows = (len(tests_selected) + ncols - 1)//ncols
        for r in range(rows):
            cols = st.columns(ncols)
            for i in range(ncols):
                idx = r*ncols + i
                if idx >= len(tests_selected):
                    continue
                t = tests_selected[idx]
                with cols[i]:
                    st.markdown(f"<div class='chart-card-title'>{t}</div>", unsafe_allow_html=True)
                    fig = plot_single_test(
                        data,
                        t,
                        show_ref=show_ref,
                        show_regression=show_trend,
                        show_zones=show_zones,
                        date_window=(blood_time_start, blood_time_end),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.caption("Select 2 or more tests to see the heatmap.")

    # Export
    render_section_header(
        "Export",
        "Take the current story with you as a static report or a filtered dataset.",
        "Share & archive",
    )
    colA, colB = st.columns(2)
    with colA:
        st.caption("Export selected charts to standalone HTML")
        if st.button("Export HTML"):
            # Create a simple HTML with embedded plotly divs
            html_parts = []
            for t in tests_selected:
                fig = plot_single_test(data, t, show_ref=show_ref, show_regression=show_trend)
                html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
            html = f"<html><head><meta charset='utf-8'><title>Blood Panel Export</title></head><body>{''.join(html_parts)}</body></html>"
            st.download_button("Download file", data=html, file_name="blood_panel_export.html", mime="text/html")

    with colB:
        st.caption("Export current filtered dataset (CSV)")
        csv = data.sort_values(["test","Date"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="blood_panel_data.csv", mime="text/csv")


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
    content_placeholder = st.empty()
    loading_placeholder = st.empty()

    fetch_steps = [
        ("Weight", "fetch_weight", "weight"),
        ("HRV", "fetch_hrv", "hrv"),
        ("Resting Heart Rate", "fetch_rhr", "rhr"),
        ("Breathing Rate", "fetch_breathing_rate", "breathing_rate"),
        ("Sleep", "fetch_sleep", "sleep"),
        ("Sleep Scores", "fetch_sleep_score", "sleep_score"),
        ("Activity", "fetch_activity", "activity"),
    ]

    results = {
        func_name: fitbit_client.load_cached_dataframe(metric_name)
        for _, func_name, metric_name in fetch_steps
    }
    has_any_cache = any(not df.empty for df in results.values())
    stale_labels = [
        label for label, _, metric_name in fetch_steps
        if not fitbit_client.is_cache_fresh(metric_name)
    ]
    auto_refresh = sync_mode is None and has_any_cache and len(stale_labels) > 0
    if sync_mode is None and not has_any_cache:
        sync_mode = "incremental"

    def fetch_fitbit_metric(label: str, func_name: str, metric_name: str, force_full: bool):
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
    sleep_score_df = results["fetch_sleep_score"]
    activity_df = results["fetch_activity"]
    with content_placeholder.container():
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

        # Merge sleep scores into sleep dataframe
        if not sleep_df.empty and not sleep_score_df.empty:
            sleep_df = sleep_df.drop(columns=["Score"], errors="ignore")
            sleep_df = sleep_df.merge(sleep_score_df[["Date", "SleepScore"]], on="Date", how="left")

        # Keep full-history copies for chart calculations so rolling averages
        # at the start of the visible window can still use preceding days.
        weight_chart_df = weight_df.copy()
        hrv_chart_df = hrv_df.copy()
        rhr_chart_df = rhr_df.copy()
        br_chart_df = br_df.copy()
        sleep_chart_df = sleep_df.copy()
        activity_chart_df = activity_df.copy()

        # Most recent data date
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
            end_inclusive = fitbit_time_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            weight_df = weight_df[(weight_df["Date"] >= fitbit_time_start) & (weight_df["Date"] <= end_inclusive)].copy() if not weight_df.empty else weight_df
            hrv_df = hrv_df[(hrv_df["Date"] >= fitbit_time_start) & (hrv_df["Date"] <= end_inclusive)].copy() if not hrv_df.empty else hrv_df
            rhr_df = rhr_df[(rhr_df["Date"] >= fitbit_time_start) & (rhr_df["Date"] <= end_inclusive)].copy() if not rhr_df.empty else rhr_df
            br_df = br_df[(br_df["Date"] >= fitbit_time_start) & (br_df["Date"] <= end_inclusive)].copy() if not br_df.empty else br_df
            sleep_df = sleep_df[(sleep_df["Date"] >= fitbit_time_start) & (sleep_df["Date"] <= end_inclusive)].copy() if not sleep_df.empty else sleep_df
            activity_df = activity_df[(activity_df["Date"] >= fitbit_time_start) & (activity_df["Date"] <= end_inclusive)].copy() if not activity_df.empty else activity_df
        else:
            fitbit_time_start = fitbit_time_end = None

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
            k1, k2 = st.columns([1, 3])
            with k1:
                latest_w = weight_df.iloc[-1]
                st.metric("Latest Weight", f"{latest_w['Weight']:.1f} kg",
                           delta=f"{weight_df['Weight'].iloc[-1] - weight_df['Weight'].iloc[-2]:.1f} kg" if len(weight_df) >= 2 else None)
            fig_w = plot_fitbit_timeseries(weight_chart_df, "Weight", "Weight", "kg", color="#E07A5F", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_w, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No weight data available.")

        # ==================================================================
        # HEALTH METRICS SECTION
        # ==================================================================
        render_section_header("Health Metrics", "", "Recovery")

        hm1, hm2, hm3 = st.columns(3)
        with hm1:
            if not hrv_df.empty:
                latest_hrv = hrv_df.dropna(subset=["RMSSD"])
                if not latest_hrv.empty:
                    st.metric("Latest HRV (RMSSD)", f"{latest_hrv.iloc[-1]['RMSSD']:.0f} ms",
                               delta=f"{latest_hrv['RMSSD'].iloc[-1] - latest_hrv['RMSSD'].iloc[-2]:.0f} ms" if len(latest_hrv) >= 2 else None)
                else:
                    st.metric("Latest HRV (RMSSD)", "—")
            else:
                st.metric("Latest HRV (RMSSD)", "—")
        with hm2:
            if not rhr_df.empty:
                latest_rhr = rhr_df.iloc[-1]
                st.metric("Latest RHR", f"{latest_rhr['RHR']:.0f} bpm",
                           delta=f"{rhr_df['RHR'].iloc[-1] - rhr_df['RHR'].iloc[-2]:.0f} bpm" if len(rhr_df) >= 2 else None,
                           delta_color="inverse")
            else:
                st.metric("Latest RHR", "—")
        with hm3:
            if not br_df.empty:
                latest_br = br_df.iloc[-1]
                st.metric("Latest Breathing Rate", f"{latest_br['BreathingRate']:.1f} brpm",
                           delta=f"{br_df['BreathingRate'].iloc[-1] - br_df['BreathingRate'].iloc[-2]:.1f} brpm" if len(br_df) >= 2 else None)
            else:
                st.metric("Latest Breathing Rate", "—")

        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

        if not hrv_df.empty:
            fig_hrv = plot_fitbit_timeseries(hrv_chart_df, "RMSSD", "HRV", "ms", color="#81B29A", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_hrv, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No HRV data available.")

        if not rhr_df.empty:
            fig_rhr = plot_fitbit_timeseries(rhr_chart_df, "RHR", "Resting HR", "bpm", color="#F2CC8F", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_rhr, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No resting heart rate data available.")

        if not br_df.empty:
            fig_br = plot_fitbit_timeseries(br_chart_df, "BreathingRate", "Breathing Rate", "brpm", color="#7EB8DA", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_br, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No breathing rate data available.")

        # ==================================================================
        # SLEEP SECTION
        # ==================================================================
        render_section_header("Sleep", "", "Rest")
        if not sleep_df.empty:
            sl1, sl2, sl3 = st.columns(3)
            with sl1:
                latest_sleep = sleep_df.iloc[-1]
                hrs = latest_sleep.get("DurationHours", 0)
                st.metric("Latest Sleep Duration", f"{hrs:.1f} hrs",
                           delta=f"{sleep_df['DurationHours'].iloc[-1] - sleep_df['DurationHours'].iloc[-2]:.1f} hrs" if len(sleep_df) >= 2 else None)
            with sl2:
                if "SleepScore" in sleep_df.columns and sleep_df["SleepScore"].notna().any():
                    score_data = sleep_df.dropna(subset=["SleepScore"])
                    if not score_data.empty:
                        st.metric("Latest Sleep Score", f"{score_data.iloc[-1]['SleepScore']:.0f}",
                                   delta=f"{score_data['SleepScore'].iloc[-1] - score_data['SleepScore'].iloc[-2]:.0f}" if len(score_data) >= 2 else None)
                    else:
                        st.metric("Latest Sleep Score", "—")
                else:
                    st.metric("Latest Sleep Score", "—")
            with sl3:
                if not sleep_df["Efficiency"].isna().all():
                    eff_data = sleep_df.dropna(subset=["Efficiency"])
                    if not eff_data.empty:
                        st.metric("Latest Efficiency", f"{eff_data.iloc[-1]['Efficiency']:.0f}%",
                                   delta=f"{eff_data['Efficiency'].iloc[-1] - eff_data['Efficiency'].iloc[-2]:.0f}%" if len(eff_data) >= 2 else None)
                    else:
                        st.metric("Latest Efficiency", "—")
                else:
                    st.metric("Latest Efficiency", "—")

            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

            fig_dur = plot_fitbit_timeseries(sleep_chart_df, "DurationHours", "Sleep Duration", "hours", color="#8E7CC3", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_dur, use_container_width=True, config={"displayModeBar": False})

            if "SleepScore" in sleep_df.columns and sleep_df["SleepScore"].notna().any():
                fig_score = plot_fitbit_timeseries(sleep_chart_df, "SleepScore", "Sleep Score", "score", color="#6AA84F", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
                st.plotly_chart(fig_score, use_container_width=True, config={"displayModeBar": False})

            stage_cols = {"REM": "#E06666", "Deep": "#3D85C6", "Light": "#F6B26B", "Wake": "#CC0000"}
            has_stages = any(sleep_df[col].notna().any() for col in stage_cols if col in sleep_df.columns)
            if has_stages:
                st.caption("Sleep Stages")
                c1, c2 = st.columns(2)
                stage_items = [(col, color) for col, color in stage_cols.items() if col in sleep_df.columns and sleep_df[col].notna().any()]
                for i, (col, color) in enumerate(stage_items):
                    with (c1 if i % 2 == 0 else c2):
                        fig_stage = plot_fitbit_timeseries(sleep_chart_df, col, col, "min", color=color, show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
                        fig_stage.update_layout(height=300)
                        st.plotly_chart(fig_stage, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No sleep data available.")

        # ==================================================================
        # ACTIVITY SECTION
        # ==================================================================
        render_section_header("Activity", "", "Movement")
        if not activity_df.empty:
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                latest_act = activity_df.iloc[-1]
                st.metric("Latest Steps", f"{latest_act.get('Steps', 0):,.0f}",
                           delta=f"{activity_df['Steps'].iloc[-1] - activity_df['Steps'].iloc[-2]:,.0f}" if len(activity_df) >= 2 else None)
            with a2:
                if "ZoneMinutes" in activity_df.columns:
                    st.metric("Latest Zone Minutes", f"{latest_act.get('ZoneMinutes', 0):.0f} min",
                               delta=f"{activity_df['ZoneMinutes'].iloc[-1] - activity_df['ZoneMinutes'].iloc[-2]:.0f} min" if len(activity_df) >= 2 else None)
            with a3:
                st.metric("Latest Distance", f"{latest_act.get('Distance', 0):.2f} km",
                           delta=f"{activity_df['Distance'].iloc[-1] - activity_df['Distance'].iloc[-2]:.2f} km" if len(activity_df) >= 2 else None)
            with a4:
                st.metric("Latest Calories", f"{latest_act.get('Calories', 0):,.0f}",
                           delta=f"{activity_df['Calories'].iloc[-1] - activity_df['Calories'].iloc[-2]:,.0f}" if len(activity_df) >= 2 else None)

            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

            fig_steps = plot_fitbit_timeseries(activity_chart_df, "Steps", "Steps", "steps", color="#E07A5F", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_steps, use_container_width=True, config={"displayModeBar": False})

            if "ZoneMinutes" in activity_df.columns:
                fig_zm = plot_fitbit_timeseries(activity_chart_df, "ZoneMinutes", "Active Zone Minutes", "min", color="#81B29A", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
                st.plotly_chart(fig_zm, use_container_width=True, config={"displayModeBar": False})

            fig_dist = plot_fitbit_timeseries(activity_chart_df, "Distance", "Distance", "km", color="#7EB8DA", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

            fig_cal = plot_fitbit_timeseries(activity_chart_df, "Calories", "Calories", "kcal", color="#F2CC8F", show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
            st.plotly_chart(fig_cal, use_container_width=True, config={"displayModeBar": False})

            zone_cols = {
                "MinutesVeryActive": ("#CC0000", "Peak"),
                "MinutesFairlyActive": ("#E07A5F", "Moderate"),
            }
            has_zones = any(col in activity_df.columns and activity_df[col].notna().any() for col in zone_cols)
            if has_zones:
                st.caption("Active Zone Minutes Breakdown")
                zc1, zc2 = st.columns(2)
                zone_items = [(col, color, label) for col, (color, label) in zone_cols.items() if col in activity_df.columns and activity_df[col].notna().any()]
                for i, (col, color, label) in enumerate(zone_items):
                    with (zc1 if i % 2 == 0 else zc2):
                        fig_zone = plot_fitbit_timeseries(activity_chart_df, col, label, "min", color=color, show_trend=show_trend, date_window=(fitbit_time_start, fitbit_time_end) if fitbit_time_start is not None else None)
                        fig_zone.update_layout(height=300)
                        st.plotly_chart(fig_zone, use_container_width=True, config={"displayModeBar": False})
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

st.set_page_config(page_title="Biomarker Studio", layout="wide")

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

section[data-testid="stSidebar"] .stRadio > div {
    gap: 0.5rem;
}

section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid rgba(24, 50, 47, 0.08);
    border-radius: 14px;
    padding: 0.7rem 0.9rem;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    gap: 0.55rem;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    min-height: 48px;
    background: rgba(255, 255, 255, 0.72) !important;
    border: 1px solid rgba(24, 50, 47, 0.08) !important;
    border-radius: 16px !important;
    padding: 0.75rem 0.9rem !important;
    transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.92) !important;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label p {
    color: var(--ink) !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    border-color: transparent !important;
    box-shadow: 0 12px 28px rgba(217, 107, 66, 0.22) !important;
}

section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) p {
    color: #fffdfb !important;
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

.stSlider [data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
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
        ["Blood Panel", "Fitbit Data", "Settings"],
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

# ---- Route to selected page ----
page_root = st.empty()
with page_root.container():
    if page == "Blood Panel":
        page_blood_panel()
    elif page == "Fitbit Data":
        page_fitbit_data()
    elif page == "Settings":
        page_settings()
