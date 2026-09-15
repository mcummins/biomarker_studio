"""Parsers for the OmicMAge / TruDiagnostic epigenetic-clock workbook.

Pure pandas, no Streamlit: ``app.py`` loads the workbook (Google Sheets or an
.xlsx export) and hands each tab's frame to the functions here.

The tabs the app charts are laid out "metrics down, sample dates across":

* ``EpiA - main`` — epigenetic clocks (OMICm Age, DunedinPoA, SymphonyAge),
  the SymphonyAge organ-system ages, the OmicMAge component centiles,
  telomeres, immunosenescence and the DNAm proxy biomarkers. Blank rows
  separate sections; a row with a label but no unit and no values is a
  section header (or a sub-section header when it follows data directly);
  a "Delta age" row or an unlabelled "centile" row belongs to the metric
  above it.
* ``TruHealth biomarkers`` — percentile scores per category and per marker,
  one value + status column pair per sample date, plus the optimal
  percentile window per marker.
* ``Sample collection notes`` — free-text circumstances per sample date.

Every metric is normalised into one long frame with a ``form`` column that
says which of the three shapes a row is: ``absolute`` (an age in years, a
pace, a length, a ratio, a cell fraction), ``delta`` (years relative to
calendar age) or ``centile`` (population percentile).
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MAIN_SHEET = "EpiA - main"
TRUHEALTH_SHEET = "TruHealth biomarkers"
NOTES_SHEET = "Sample collection notes"

FORM_ABSOLUTE = "absolute"
FORM_DELTA = "delta"
FORM_CENTILE = "centile"

CALENDAR_AGE = "Calendar Age"

SECTION_CLOCKS = "Epigenetic clocks"
SECTION_SYMPHONY = "SymphonyAge"
SECTION_ORGANS = "SymphonyAge organ systems"
SECTION_COMPONENTS = "OmicMAge components"
SECTION_TELOMERES = "Telomeres"
SECTION_IMMUNE = "Immunosenescence"
SECTION_DNAM = "DNAm proxy biomarkers"

# Canonical section names the page keys off, mapped from the (normalised)
# header labels in the sheet. The second element pins the form for every row
# of that section where the sheet does not say per row: the organ-system
# blocks encode the form in their titles, and the component / DNAm blocks
# are centiles throughout.
SECTION_MAP = {
    "epigenetic age": (SECTION_CLOCKS, None),
    "symphonyage - overall": (SECTION_SYMPHONY, None),
    "symphonyage - organ system age": (SECTION_ORGANS, FORM_ABSOLUTE),
    "symphonyage - organ system delta age": (SECTION_ORGANS, FORM_DELTA),
    "omicmage components (percentiles)": (SECTION_COMPONENTS, FORM_CENTILE),
    "telomeres": (SECTION_TELOMERES, None),
    "dnam proxy biomarkers": (SECTION_DNAM, FORM_CENTILE),
}

# Not charted: the organ-system centiles exist for only two samples and the
# mitotic clock has two data points. The fitness-age block is a separate
# model that is not tracked here; it has no header row of its own, so its
# metrics are skipped by name (their delta / centile rows go with them).
SKIP_SECTIONS = {
    "symphonyage - organ system centile",
    "mitotic clock",
}
SKIP_METRICS = {
    "fitness age",
    "gait speed",
    "grip strength",
    "vo2max",
    "fev1",
}

# (canonical section, lower-cased sheet label) -> (display metric, form override)
METRIC_ALIASES = {
    (SECTION_SYMPHONY, "age"): ("SymphonyAge", None),
    (SECTION_TELOMERES, "length"): ("Telomere length", None),
    (SECTION_TELOMERES, "centile for age"): ("Telomere length", FORM_CENTILE),
}

# Age clocks are lower-is-better by definition; the sheet leaves the
# direction blank on some of these rows (SymphonyAge and the organ systems).
LOWER_IS_BETTER_SECTIONS = {SECTION_CLOCKS, SECTION_SYMPHONY, SECTION_ORGANS}

# The immunosenescence rows changed meaning mid-history: reports up to
# 27/01/2025 hold absolute cell fractions / ratios, and from 26/05/2025 the
# same rows hold population centiles (per the sheet note on the Bcell row).
IMMUNE_CENTILE_FROM = pd.Timestamp("2025-05-26")

MAIN_COLUMNS = [
    "section", "subsection", "metric", "form", "unit", "direction",
    "Date", "Value", "notes", "row_order", "derived",
]
TRUHEALTH_COLUMNS = [
    "group", "marker", "Date", "Value", "status", "lower", "upper", "row_order",
]
TRUHEALTH_CATEGORY = "Category"
TRUHEALTH_MARKER = "Marker"

_DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$")
_MISSING = {"", "-", "—", "–", "n/a", "na", "none"}


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _cell(value) -> str:
    """A sheet cell as trimmed text; dates (from .xlsx exports) as dd/mm/yyyy."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%d/%m/%Y")
    text = str(value).strip()
    if text.lower() == "nan" or text.startswith("Unnamed:"):
        return ""
    return text


def _rows(df: pd.DataFrame) -> List[List[str]]:
    """Header + body as positional lists of strings (headers may repeat)."""
    header = [_cell(c) for c in df.columns]
    body = [[_cell(c) for c in row] for row in df.itertuples(index=False, name=None)]
    width = max([len(header)] + [len(r) for r in body])
    return [r + [""] * (width - len(r)) for r in [header] + body]


def parse_date(text: str) -> Optional[pd.Timestamp]:
    match = _DATE_RE.match(text or "")
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def parse_number(text: str) -> Optional[float]:
    text = (text or "").strip().replace(",", "")
    if text.lower() in _MISSING:
        return None
    text = text.rstrip("%").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _find_col(header: List[str], name: str, default: Optional[int] = None) -> Optional[int]:
    for i, cell in enumerate(header):
        if cell.strip().lower() == name:
            return i
    return default


def _date_columns(header: List[str]) -> List[Tuple[int, pd.Timestamp]]:
    return [(i, d) for i, cell in enumerate(header) if (d := parse_date(cell)) is not None]


def _canonical_section(label: str) -> Tuple[str, Optional[str]]:
    key = label.strip().lower()
    if key in SECTION_MAP:
        return SECTION_MAP[key]
    if key.startswith("immuno"):   # spelled "Immunoscenesence" in the sheet
        return SECTION_IMMUNE, None
    return label.strip(), None


def _normalise_direction(text: str) -> str:
    key = (text or "").strip().lower()
    if key.startswith("low"):
        return "Lower"
    if key.startswith("high"):
        return "Higher"
    return ""


def _is_years(unit: str) -> bool:
    return unit.strip().lower() in {"years", "year", "yrs"}


# ---------------------------------------------------------------------------
# EpiA - main
# ---------------------------------------------------------------------------
def parse_main(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Melt the ``EpiA - main`` tab into one row per (metric, form, date)."""
    empty = pd.DataFrame(columns=MAIN_COLUMNS)
    if df is None or df.empty:
        return empty
    rows = _rows(df)
    header = rows[0]
    date_cols = _date_columns(header)
    if not date_cols:
        return empty
    unit_col = _find_col(header, "unit", 1)
    dir_col = _find_col(header, "optimal direction", 2)
    notes_col = _find_col(header, "notes")

    section, section_form = _canonical_section(header[0] or "Epigenetic Age")
    skip = (header[0] or "").strip().lower() in SKIP_SECTIONS
    subsection = ""
    last_metric: Optional[str] = None
    after_gap = False
    records = []

    for row_index, row in enumerate(rows[1:], start=1):
        label = row[0]
        unit = row[unit_col] if unit_col is not None and unit_col < len(row) else ""
        direction = row[dir_col] if dir_col is not None and dir_col < len(row) else ""
        note = row[notes_col] if notes_col is not None and notes_col < len(row) else ""
        values = [(sample_date, parse_number(row[i])) for i, sample_date in date_cols]
        has_values = any(v is not None for _, v in values)

        if not label and not unit and not has_values:
            after_gap = True
            continue

        if label and not unit and not direction and not has_values:
            # A bare label opens a section after a gap, else a sub-section.
            if after_gap or last_metric is None:
                section, section_form = _canonical_section(label)
                skip = label.strip().lower() in SKIP_SECTIONS
                subsection = ""
            else:
                subsection = label
            last_metric = None
            after_gap = False
            continue

        # A gap between data rows is just spacing; the section carries on.
        after_gap = False
        if skip:
            continue

        unit_key = unit.lower()
        if label.lower().startswith("delta") and last_metric:
            metric, form = last_metric, FORM_DELTA
        elif not label and last_metric:
            metric = last_metric
            form = FORM_CENTILE if "centile" in unit_key else FORM_ABSOLUTE
        else:
            metric = label
            last_metric = label
            form = FORM_CENTILE if "centile" in unit_key else FORM_ABSOLUTE
        if metric.strip().lower() in SKIP_METRICS:
            continue
        if section_form:
            form = section_form
        alias = METRIC_ALIASES.get((section, metric.strip().lower()))
        if alias:
            metric, alias_form = alias
            form = alias_form or form
        if _is_years(unit):
            unit = "Years"
        elif not unit and section in LOWER_IS_BETTER_SECTIONS and form in (FORM_ABSOLUTE, FORM_DELTA):
            unit = "Years"   # the organ-system blocks leave the unit column blank

        direction = _normalise_direction(direction)
        if not direction and section in LOWER_IS_BETTER_SECTIONS and metric != CALENDAR_AGE:
            direction = "Lower"

        for sample_date, value in values:
            if value is None:
                continue
            row_form, row_unit = form, unit
            if section == SECTION_IMMUNE:
                if sample_date >= IMMUNE_CENTILE_FROM:
                    row_form, row_unit = FORM_CENTILE, "centile"
                elif not unit:
                    row_unit = "%"
            elif row_form == FORM_CENTILE:
                row_unit = "centile"
            records.append({
                "section": section,
                "subsection": subsection,
                "metric": metric,
                "form": row_form,
                "unit": row_unit,
                "direction": direction,
                "Date": sample_date,
                "Value": value,
                "notes": note,
                "row_order": row_index,
                "derived": False,
            })

    long = pd.DataFrame(records, columns=MAIN_COLUMNS)
    if long.empty:
        return long
    long = pd.concat([long, _derived_deltas(long)], ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"])
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce")
    return long.sort_values(["row_order", "form", "Date"]).reset_index(drop=True)


def _derived_deltas(long: pd.DataFrame) -> pd.DataFrame:
    """Age clocks reported only in years get a delta = age − calendar age."""
    calendar = calendar_age(long)
    if calendar.empty:
        return pd.DataFrame(columns=MAIN_COLUMNS)
    calendar_by_date = dict(zip(calendar["Date"], calendar["Value"]))
    records = []
    for (section, metric), group in long.groupby(["section", "metric"], sort=False):
        if metric == CALENDAR_AGE or FORM_DELTA in set(group["form"]):
            continue
        absolute = group[(group["form"] == FORM_ABSOLUTE) & group["unit"].map(_is_years)]
        for _, row in absolute.iterrows():
            base = calendar_by_date.get(row["Date"])
            if base is None:
                continue
            record = row.to_dict()
            record.update({
                "form": FORM_DELTA,
                "Value": float(row["Value"]) - float(base),
                "derived": True,
                "notes": "Derived: predicted age minus calendar age on the same date",
            })
            records.append(record)
    return pd.DataFrame(records, columns=MAIN_COLUMNS)


def calendar_age(long: pd.DataFrame) -> pd.DataFrame:
    """Calendar age per sample date (``Date``, ``Value``), sorted."""
    if long.empty:
        return pd.DataFrame(columns=["Date", "Value"])
    rows = long[(long["metric"] == CALENDAR_AGE) & (long["form"] == FORM_ABSOLUTE)]
    return rows[["Date", "Value"]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def calendar_age_at(calendar: pd.DataFrame, when) -> Optional[float]:
    """Calendar age on any date, from the linear fit of the sampled ages."""
    if calendar is None or len(calendar) == 0:
        return None
    when = pd.Timestamp(when)
    exact = calendar[calendar["Date"] == when]
    if not exact.empty:
        return float(exact["Value"].iloc[0])
    if len(calendar) == 1:
        base_date, base_age = calendar["Date"].iloc[0], float(calendar["Value"].iloc[0])
        return base_age + (when - base_date).days / 365.25
    days = (calendar["Date"] - calendar["Date"].min()).dt.days.astype(float).values
    slope, intercept = np.polyfit(days, calendar["Value"].astype(float).values, 1)
    return float(slope * (when - calendar["Date"].min()).days + intercept)


def metrics_in(long: pd.DataFrame, section: str, subsection: Optional[str] = None,
               unit: Optional[str] = None, form: Optional[str] = None) -> List[str]:
    """Metric names of a section in sheet order, optionally filtered."""
    rows = long[long["section"] == section]
    if subsection is not None:
        rows = rows[rows["subsection"] == subsection]
    if unit is not None:
        rows = rows[rows["unit"] == unit]
    if form is not None:
        rows = rows[rows["form"] == form]
    rows = rows[rows["metric"] != CALENDAR_AGE]
    return rows.sort_values("row_order")["metric"].drop_duplicates().tolist()


# ---------------------------------------------------------------------------
# Sample collection notes
# ---------------------------------------------------------------------------
def parse_sample_notes(df: Optional[pd.DataFrame]) -> Dict[pd.Timestamp, str]:
    notes: Dict[pd.Timestamp, str] = {}
    if df is None or df.empty:
        return notes
    for row in _rows(df):
        sample_date = parse_date(row[0])
        if sample_date is None:
            continue
        text = next((cell for cell in row[1:] if cell), "")
        if text:
            notes[sample_date] = text
    return notes


# ---------------------------------------------------------------------------
# TruHealth biomarkers
# ---------------------------------------------------------------------------
def parse_truhealth(df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, str]]:
    """Melt the TruHealth tab into (long frame, {sample date: note})."""
    empty = pd.DataFrame(columns=TRUHEALTH_COLUMNS)
    date_notes: Dict[pd.Timestamp, str] = {}
    if df is None or df.empty:
        return empty, date_notes
    rows = _rows(df)
    header = rows[0]
    date_cols = _date_columns(header)
    if not date_cols:
        return empty, date_notes
    status_col = {
        i: (i + 1 if i + 1 < len(header) and header[i + 1].strip().lower() == "status" else None)
        for i, _ in date_cols
    }
    lower_col = _find_col(header, "lower limit")
    upper_col = _find_col(header, "upper limit")
    notes_col = _find_col(header, "notes")

    group = TRUHEALTH_CATEGORY
    records = []
    for row_index, row in enumerate(rows[1:], start=1):
        if notes_col is not None:
            note_date = parse_date(row[notes_col])
            if note_date is not None:
                text = next((cell for cell in row[notes_col + 1:] if cell), "")
                if text:
                    date_notes[note_date] = text
        label = row[0]
        if not label:
            continue
        values = [
            (sample_date, parse_number(row[i]), row[status_col[i]] if status_col[i] is not None else "")
            for i, sample_date in date_cols
        ]
        if not any(v is not None for _, v, _ in values):
            if label.strip().lower().startswith("individual"):
                group = TRUHEALTH_MARKER
            continue
        lower = parse_number(row[lower_col]) if lower_col is not None else None
        upper = parse_number(row[upper_col]) if upper_col is not None else None
        marker = label
        if group == TRUHEALTH_CATEGORY:
            marker = re.sub(r"\s*\(overall\)\s*$", "", label, flags=re.IGNORECASE)
        for sample_date, value, status in values:
            if value is None:
                continue
            records.append({
                "group": group,
                "marker": marker,
                "Date": sample_date,
                "Value": value,
                "status": status.strip(),
                "lower": lower,
                "upper": upper,
                "row_order": row_index,
            })

    long = pd.DataFrame(records, columns=TRUHEALTH_COLUMNS)
    if long.empty:
        return long, date_notes
    long["Date"] = pd.to_datetime(long["Date"])
    for column in ("Value", "lower", "upper"):
        long[column] = pd.to_numeric(long[column], errors="coerce")
    return long.sort_values(["row_order", "Date"]).reset_index(drop=True), date_notes


def truhealth_latest(long: pd.DataFrame) -> pd.DataFrame:
    """Latest row per marker, with the previous value for a delta column."""
    if long.empty:
        return long.copy()
    data = long.sort_values(["marker", "Date"]).copy()
    data["PrevValue"] = data.groupby("marker")["Value"].shift(1)
    latest_idx = data.groupby("marker")["Date"].idxmax()
    return data.loc[latest_idx].sort_values("row_order").reset_index(drop=True)
