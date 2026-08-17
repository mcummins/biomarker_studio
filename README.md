# Blood Panel Explorer (Streamlit)

A local-first, developer-friendly app to visualize your longitudinal blood test data directly from **Google Sheets** (or an offline Excel export).

**Highlights**
- Pulls **directly from your Google Sheet** (source of truth).
- Interactive charts with **reference range bands**, **trend lines**, and **lab notes** overlays.
- Quick insights: **out-of-range flags**, **Δ since last test**, **heatmap** vs. reference range.
- Works entirely **locally** on your Mac (runs a local web server).

---

## 1) Prereqs

- Python 3.10+ recommended
- A Google Cloud **Service Account** with access to the Google Sheets API
- Your Google Sheet (share it with the service account email)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Google Sheets setup

1. Go to Google Cloud Console → APIs & Services → **Enable APIs** → enable **Google Sheets API**.
2. Create **Credentials** → **Service account** → download the JSON key.
3. In Google Sheets, **share** your sheet with the service account's email (as Viewer).
4. Keep the JSON key safe; do **not** commit to git.

## 3) Run the app

```bash
streamlit run app.py
```

- In the sidebar, paste your **Google Sheets URL** or **spreadsheet ID**.
- Upload the **service account JSON** file.
- Pick a category or search tests; select what to visualize.

## 4) Optional: Offline mode

If you prefer to avoid Google APIs entirely, export the workbook as `.xlsx` and use the **Upload Excel** option.

## 4b) Wearable & scale data sources (2026 migration)

The Fitbit Web API sunsets in September 2026, and the Garmin scale stopped
reaching it in July 2026, so device data now comes from three places:

- **Weight** — `garmin_client.py` pulls weigh-ins from Garmin Connect
  (`.garmin_cache/`). One-time login: `.venv/bin/python garmin_login.py`.
  The app merges this with the frozen Fitbit weight history, which includes
  manually logged Fitbit entries (Garmin wins on shared dates).
- **Activity / sleep / HRV / RHR / breathing rate** — migrating from
  `fitbit_client.py` to `google_health_client.py` (Google Health API v4).
  One-time setup: create an OAuth client per the docstring in
  `google_health_login.py`, then run it. Before the app switches over, run
  `compare_fitbit_google.py` to diff both APIs day-by-day
  (report: `archive/fitbit_vs_google_report.md`).
- **Frozen archive** — `archive/fitbit_cache_2026-08-17/` is a complete
  snapshot of everything the Fitbit Web API ever returned (through the
  sunset). Do not delete; it is the only copy of pre-Garmin weight and any
  fields Google Health does not carry forward.

## 5) Notes

- The app infers date columns dynamically and handles common result formats like `"<9.00"`, `">130"`, and `4-9` ranges.
- Reference ranges are read from the **"Optimal Ranges"** sheet.
- Sample/lab-level notes from **"Labs and notes"** are displayed in hovers.

## 6) Customization ideas

- Add a YAML file of **personal events** (meds, supplements, training changes) and draw vertical annotations.
- Add **unit conversions** or harmonize lab differences where necessary.
- Compute derived markers (e.g., remnant cholesterol) in a separate module.

---

### Troubleshooting

- If you see "Expected sheets 'All Data' and 'Optimal Ranges' not found", double-check tab names.
- If a chart shows no band, we likely don't have both lower & upper bounds for that test in **Optimal Ranges**.
- For private Sheets, you **must** share with the service account email, or you'll get a 403.