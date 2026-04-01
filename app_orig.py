
import os
import io
import json
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Config
# -----------------------------
EXCLUDE_SHEETS = {"All Data", "Optimal Ranges", "Graphs", "Labs and notes"}
DEFAULT_GROUP_SHEETS = []  # will be filled dynamically

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

def highlight_status(val, status):
    """Return background-color CSS for the 'Value' column based on status."""
    if status == "normal":
        return "background-color: rgba(76, 175, 80, 0.25)"  # green
    if status == "high":
        return "background-color: rgba(244, 67, 54, 0.25)"  # red
    if status == "low":
        return "background-color: rgba(244, 67, 54, 0.25)"  # red
    return ""

def compute_zscore(value, lower, upper) -> Optional[float]:
    if pd.notna(lower) and pd.notna(upper):
        mid = (lower + upper)/2.0
        half_width = (upper - lower)/2.0
        if half_width and half_width != 0:
            return (value - mid) / half_width
    return None

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
        # try to read tests listed in this tab
        if "Test" in df.columns:
            tests = df["Test"].dropna().astype(str).unique().tolist()
            if len(tests) > 0:
                groups[name] = tests
    return groups

# -----------------------------
# Data loaders
# -----------------------------
@st.cache_data(show_spinner=False, ttl=600)
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


def plot_single_test(df: pd.DataFrame, test: str,
                     show_ref: bool=True, show_regression: bool=False,
                     show_zones: bool=True, range_policy: str="union") -> go.Figure:
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
                fillcolor=color_hex, opacity=0.18, line_width=0, layer="below"
            )

        # colors: green, orange, red
        GREEN = "#4CAF50"; ORANGE = "#FF9800"; RED = "#F44336"

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
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=test, hoverinfo="text", text=hover))

    if show_regression and len(g) >= 3:
        days = (g["Date"] - g["Date"].min()).dt.days.values.astype(float)
        slope, intercept = np.polyfit(days, y.values.astype(float), 1)
        x_line = pd.date_range(start=g["Date"].min(), end=g["Date"].max(), periods=50)
        x_days = (x_line - g["Date"].min()).days.values.astype(float)
        y_line = slope * x_days + intercept
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Trend", line=dict(dash="dash")))

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

    return fig


def plot_heatmap(df: pd.DataFrame, tests: List[str]) -> go.Figure:
    # Build matrix of z-scores by date
    g = df[df["test"].isin(tests)].copy()
    g["z"] = g.apply(lambda r: compute_zscore(r["Value"], r.get("lower"), r.get("upper")), axis=1)
    g = g.dropna(subset=["z"])
    if g.empty:
        return go.Figure()
    pivot = g.pivot_table(index="test", columns="Date", values="z", aggfunc="mean")
    pivot = pivot.sort_index()
    fig = px.imshow(pivot, aspect="auto", origin="lower", color_continuous_scale="RdBu_r", labels=dict(color="Z vs range"))
    fig.update_layout(height=300 + 20*len(pivot), margin=dict(l=10,r=10,t=40,b=40), title="Heatmap: position vs reference range")
    return fig

def make_sparkline(df: pd.DataFrame, test: str) -> go.Figure:
    g = df[df["test"] == test].sort_values("Date")
    fig = go.Figure(go.Scatter(x=g["Date"], y=g["Value"], mode="lines"))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=80, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig

# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Blood Panel Explorer", layout="wide")

st.title("🧪 Blood Panel Explorer")
st.caption("Interactive, local-first visualization of longitudinal lab results with reference ranges, trend analysis, and quick insights.")

with st.sidebar:
    st.header("Data source")

    # Convenience: auto-load local sheet_api_key.json if present
    local_key_path = os.path.join(os.path.dirname(__file__), "sheet_api_key.json")
    sheets = None
    if os.path.exists(local_key_path):
        try:
            import json
            with open(local_key_path, "r") as f:
                service_account_info = json.load(f)

            # You can hardcode your preferred Google Sheet URL/ID here
            default_sheet_url = "https://docs.google.com/spreadsheets/d/1pfYaK6t25gcKdBAUu8_geyGlQP6wp6px9IyNUKI4wdw/edit?usp=sharing"
            spreadsheet_id = parse_spreadsheet_id(default_sheet_url)

            st.caption(f"Using local key: `{local_key_path}`")
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
            refresh = st.button("🔄 Refresh")
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
group_names = ["(All)"] + sorted(list(groups.keys()))
st.sidebar.header("Filters")
grp = st.sidebar.selectbox("Category", options=group_names)
if grp != "(All)":
    selected_tests = groups.get(grp, [])
else:
    selected_tests = sorted(merged["test"].unique().tolist())

# Search/select tests
search = st.sidebar.text_input("Search test name")
if search:
    candidates = [t for t in selected_tests if search.lower() in t.lower()]
else:
    candidates = selected_tests

tests_selected = st.sidebar.multiselect("Select tests to visualize", options=candidates, default=candidates[:len(candidates)])

# Date range filter
min_date = pd.to_datetime(merged["Date"].min())
max_date = pd.to_datetime(merged["Date"].max())
date_range = st.sidebar.slider("Date range", min_value=min_date.to_pydatetime(), max_value=max_date.to_pydatetime(), value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
mask = (merged["Date"] >= pd.to_datetime(date_range[0])) & (merged["Date"] <= pd.to_datetime(date_range[1]))
data = merged[mask]

# Insights
data = compute_deltas(data)

# Restrict to selected tests (use full data if nothing selected)
data_sel = data[data["test"].isin(tests_selected)] if tests_selected else data.copy()

# Use the same filtered set as Δ table
latest_date = pd.to_datetime(data_sel["Date"].max())
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Latest sample date", latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "—")
with k2:
    out_now = data_sel[(data_sel["Date"] == latest_date) & (data_sel["status"].isin(["low","high"]))]["test"].nunique()
    st.metric("Out of range (latest)", int(out_now))
with k3:
    total_measured = data_sel[data_sel["Date"] == latest_date]["test"].nunique()
    st.metric("Tests measured (latest)", int(total_measured))

# "What's changed since last test"
st.subheader("Δ Since last test")


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
    latest_per_test["Δ"]  = (latest_per_test["Value"] - latest_per_test["PrevValue"]).round(3)
    latest_per_test["Δ%"] = (latest_per_test["Δ"] / latest_per_test["PrevValue"] * 100).round(1)

    cols = ["test","unit","PrevValue","Value","Δ","Δ%","status"]
    table_df = latest_per_test.sort_values("Δ%", ascending=False)[cols].reset_index(drop=True)
    
    # Apply coloring to the 'Value' column based on 'status'
    styled = table_df.style.apply(
    lambda row: [highlight_status(row["Value"], row["status"])] + [""] * (len(row) - 1),
    axis=1)
    
    st.dataframe(styled, use_container_width=True, hide_index=True)
                 
# Grid of charts (sparklines) for selected tests
st.subheader("Selected tests")
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
                st.markdown(f"**{t}**")
                fig = plot_single_test(data, t, show_ref=show_ref, show_regression=show_trend, show_zones=show_zones)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Use the sidebar to select tests to visualize.")

# Heatmap overview
st.subheader("Overview heatmap (position vs ref range)")
if len(tests_selected) >= 2:
    fig_hm = plot_heatmap(data, tests_selected[:30])  # limit to 30 for readability
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.caption("Select 2 or more tests to see the heatmap.")

# Export
st.subheader("Export")
colA, colB = st.columns(2)
with colA:
    st.caption("Export selected charts to standalone HTML")
    if st.button("⬇️ Export HTML"):
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
    st.download_button("⬇️ Download CSV", data=csv, file_name="blood_panel_data.csv", mime="text/csv")
