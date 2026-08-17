"""
Data-parity check: Fitbit Web API archive vs Google Health API.

Run after google_health_login.py has been completed:

    .venv/bin/python compare_fitbit_google.py [--start 2025-08-01]

Fetches the comparison window from the Google Health API (into the normal
.google_health_cache), aligns it day-by-day against the frozen Fitbit archive
(archive/fitbit_cache_2026-08-17), and writes a markdown report to
archive/fitbit_vs_google_report.md flagging:

  - days present in Fitbit but missing from Google (per metric)
  - matching days whose values differ beyond tolerance
  - Fitbit fields with no Google Health equivalent (all-empty columns)

Nothing here mutates the Fitbit archive.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta

import pandas as pd

import fitbit_client
import google_health_client as ghc

HERE = os.path.dirname(__file__)
ARCHIVE_DIR = os.path.join(HERE, "archive", "fitbit_cache_2026-08-17")
REPORT_PATH = os.path.join(HERE, "archive", "fitbit_vs_google_report.md")

# metric -> (fitbit archive file, [(fitbit_col, google_col, abs_tol, rel_tol)])
COMPARISONS = {
    "activity": [
        ("Steps", "Steps", 50, 0.03),
        ("Calories", "Calories", 50, 0.05),
        ("Distance", "Distance", 0.2, 0.05),
    ],
    "rhr": [("RHR", "RHR", 1, 0.03)],
    "hrv": [("RMSSD", "RMSSD", 1, 0.05)],
    "breathing_rate": [("BreathingRate", "BreathingRate", 0.5, 0.05)],
    "sleep": [
        ("DurationMinutes", "DurationMinutes", 15, 0.05),
        ("REM", "REM", 10, 0.10),
        ("Deep", "Deep", 10, 0.10),
        ("Light", "Light", 15, 0.10),
        ("Wake", "Wake", 10, 0.15),
        ("Efficiency", "Efficiency", 3, 0.05),
    ],
}

GOOGLE_FETCHERS = {
    "activity": ghc.fetch_activity,
    "rhr": ghc.fetch_rhr,
    "hrv": ghc.fetch_hrv,
    "breathing_rate": ghc.fetch_breathing_rate,
    "sleep": ghc.fetch_sleep,
}

FITBIT_LOADERS = {
    "activity": fitbit_client.load_cached_dataframe,
    "rhr": fitbit_client.load_cached_dataframe,
    "hrv": fitbit_client.load_cached_dataframe,
    "breathing_rate": fitbit_client.load_cached_dataframe,
    "sleep": fitbit_client.load_cached_dataframe,
}


def load_fitbit_archive(metric: str) -> pd.DataFrame:
    """Load a metric from the frozen archive copy (not the live cache)."""
    path = os.path.join(ARCHIVE_DIR, f"{metric}.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        records = json.load(f)
    # Reuse fitbit_client's record->DataFrame converters via a temp swap.
    converters = {
        "activity": fitbit_client._activity_records_to_df,
        "rhr": fitbit_client._rhr_records_to_df,
        "hrv": fitbit_client._hrv_records_to_df,
        "breathing_rate": fitbit_client._breathing_rate_records_to_df,
        "sleep": fitbit_client._sleep_records_to_df,
    }
    return converters[metric](records)


def compare_metric(metric: str, start: date, end: date, lines: list) -> None:
    fb = load_fitbit_archive(metric)
    gg = ghc.load_cached_dataframe(metric)

    lines.append(f"\n## {metric}\n")
    if fb.empty:
        lines.append("- Fitbit archive has no data — nothing to compare.")
        return
    if gg.empty:
        lines.append("- **Google returned no data for this metric at all.**")
        return

    fb = fb[(fb["Date"] >= pd.Timestamp(start)) & (fb["Date"] <= pd.Timestamp(end))]
    gg = gg[(gg["Date"] >= pd.Timestamp(start)) & (gg["Date"] <= pd.Timestamp(end))]

    fb_days = set(fb["Date"].dt.date)
    gg_days = set(gg["Date"].dt.date)
    missing = sorted(fb_days - gg_days)
    extra = sorted(gg_days - fb_days)
    lines.append(f"- window: {start} → {end}")
    lines.append(f"- Fitbit days: {len(fb_days)}, Google days: {len(gg_days)}")
    if missing:
        shown = ", ".join(str(d) for d in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        lines.append(f"- **Missing from Google: {len(missing)} days** — {shown}{more}")
    else:
        lines.append("- No missing days.")
    if extra:
        lines.append(f"- Days only in Google: {len(extra)}")

    merged = fb.merge(gg, on="Date", suffixes=("_fb", "_gg"))
    for fb_col, gg_col, abs_tol, rel_tol in COMPARISONS[metric]:
        a = merged.get(f"{fb_col}_fb", merged.get(fb_col))
        b = merged.get(f"{gg_col}_gg", merged.get(gg_col))
        if a is None or b is None:
            lines.append(f"- {fb_col}: column missing on one side — **check schema**")
            continue
        both = pd.DataFrame({"Date": merged["Date"], "fb": a, "gg": b}).dropna()
        if both.empty:
            lines.append(f"- {fb_col}: **no Google values** (all empty)")
            continue
        diff = (both["fb"] - both["gg"]).abs()
        rel = diff / both["fb"].abs().clip(lower=1e-9)
        bad = both[(diff > abs_tol) & (rel > rel_tol)]
        if bad.empty:
            lines.append(
                f"- {fb_col}: OK on {len(both)} shared days "
                f"(median |Δ| {diff.median():.2f})"
            )
        else:
            worst = bad.assign(d=diff[bad.index]).nlargest(5, "d")
            examples = "; ".join(
                f"{r.Date.date()}: {r.fb:.1f} vs {r.gg:.1f}" for r in worst.itertuples()
            )
            lines.append(
                f"- {fb_col}: **{len(bad)}/{len(both)} days differ** beyond "
                f"tolerance — worst: {examples}"
            )

    # Columns Fitbit had that are entirely empty on the Google side.
    empty_cols = [
        c for c in gg.columns
        if c != "Date" and gg[c].isna().all() and c in fb.columns and fb[c].notna().any()
    ]
    if empty_cols:
        lines.append(
            f"- Fields with Fitbit history but no Google equivalent: "
            f"**{', '.join(empty_cols)}** (history preserved in archive; "
            f"no new data going forward)"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None,
                        help="Comparison window start (default: 90 days ago)")
    parser.add_argument("--end", default=None,
                        help="Comparison window end (default: yesterday)")
    args = parser.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    start = (date.fromisoformat(args.start) if args.start
             else end - timedelta(days=90))

    lines = [
        "# Fitbit Web API vs Google Health API — data parity report",
        f"\nGenerated {date.today().isoformat()}. Window {start} → {end}.",
        "\nFitbit side is the frozen archive (archive/fitbit_cache_2026-08-17);",
        "Google side is fetched live into .google_health_cache.",
    ]

    for metric, fetcher in GOOGLE_FETCHERS.items():
        print(f"Fetching {metric} from Google Health API...")
        try:
            fetcher(start_date=start.isoformat())
        except Exception as e:
            lines.append(f"\n## {metric}\n- **Google fetch FAILED:** {e}")
            print(f"  FAILED: {e}")
            continue
        compare_metric(metric, start, end, lines)
        print(f"  done")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
