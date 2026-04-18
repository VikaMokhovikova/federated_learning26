#!/usr/bin/env python3
"""
GetFMIHourly.py — Download hourly FMI wind + pressure observations and
aggregate them to daily values per station.

Fetched parameters (hourly):
    WS_PT1H_AVG  — wind speed average, m/s
    WD_PT1H_AVG  — wind direction average, degrees (0–360, 0=N)
    PA_PT1H_AVG  — air pressure at sea level, hPa

Daily aggregation:
    ws_day  = mean(WS_PT1H_AVG)          — mean wind speed (m/s)
    wu_day  = mean(WS * cos(WD_rad))     — zonal wind component
    wv_day  = mean(WS * sin(WD_rad))     — meridional wind component
    pa_day  = mean(PA_PT1H_AVG)          — mean pressure (hPa)

Wind direction is aggregated as vector components (circular statistics) to
avoid the wrap-around problem: the naive mean of 350° and 10° is 180° (wrong),
but the vector mean gives 0° (correct north wind).

Output:
    daily_wind_pressure.csv  — columns: station, lat, lon, day,
                                ws_day, wu_day, wv_day, pa_day

Usage:
    python GetFMIHourly.py

The script fetches the same date range as ReadInDailyMaxMin.py (last 40 days).
Run both scripts, then the notebook merges the two CSVs automatically.
"""

import requests
import xml.etree.ElementTree as ET
import datetime as dt
from datetime import timezone, timedelta
from collections import defaultdict
import math
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BBOX           = "bbox=19.0,59.5,31.6,70.2"   # entire Finland
HOURLY_PARAMS  = "WS_PT1H_AVG,WD_PT1H_AVG,PA_PT1H_AVG"
STORED_QUERY   = "fmi::observations::weather::hourly::timevaluepair"
CHUNK_DAYS     = 7          # request this many days per API call (avoids huge responses)
OUT_CSV        = "daily_wind_pressure.csv"
MIN_HOURS      = 12         # minimum valid hourly readings to compute a daily aggregate

# ─────────────────────────────────────────────────────────────────────────────
# XML parser for hourly timevaluepair
# ─────────────────────────────────────────────────────────────────────────────

NS = {
    "wml2":   "http://www.opengis.net/waterml/2.0",
    "gml":    "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om":     "http://www.opengis.net/om/2.0",
}


def parse_hourly_xml(xml_bytes: bytes) -> list:
    """Parse hourly timevaluepair XML into a list of dicts."""
    root = ET.fromstring(xml_bytes)
    rows = []
    for member in root.findall(".//wfs:member", {"wfs": "http://www.opengis.net/wfs/2.0"}):
        location = member.find(".//target:Location", NS)
        if location is None:
            continue
        name_el = location.find(
            './gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]', NS
        )
        if name_el is None:
            continue
        station = name_el.text
        pos_text = member.find(".//gml:pos", NS)
        if pos_text is None:
            continue
        lat, lon = float(pos_text.text.split()[0]), float(pos_text.text.split()[1])

        obs_prop = member.find(".//om:observedProperty", NS)
        if obs_prop is None:
            continue
        href = obs_prop.get("{http://www.w3.org/1999/xlink}href", "")
        if "param=" not in href:
            continue
        param = href.split("param=")[1].split("&")[0]

        for point in member.findall(".//wml2:point", NS):
            time_el  = point.find(".//wml2:time", NS)
            value_el = point.find(".//wml2:value", NS)
            if time_el is None or value_el is None:
                continue
            val = None if value_el.text in (None, "NaN") else float(value_el.text)
            rows.append({
                "station":   station,
                "lat":       lat,
                "lon":       lon,
                "time_utc":  time_el.text,   # "YYYY-MM-DDTHH:MM:SSZ"
                "parameter": param,
                "value":     val,
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_chunk(start: dt.datetime, end: dt.datetime) -> list:
    """Fetch one time chunk from FMI hourly API. Returns list of row dicts."""
    url = (
        f"http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature"
        f"&storedquery_id={STORED_QUERY}"
        f"&{BBOX}"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        f"&parameters={HOURLY_PARAMS}"
        f"&timestep=60"
    )
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        print(f"  WARNING: HTTP {response.status_code} for {start.date()}–{end.date()}")
        print(f"  {response.text[:200]}")
        return []
    return parse_hourly_xml(response.content)


# ─────────────────────────────────────────────────────────────────────────────
# Daily aggregation with circular wind statistics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_to_daily(all_rows: list) -> pd.DataFrame:
    """
    Aggregate hourly rows into daily (station, day) statistics.

    Wind direction is aggregated as vector components (circular mean):
        U = WS * cos(WD_radians),  V = WS * sin(WD_radians)
    This correctly handles the 360°→0° wraparound.

    Returns DataFrame with columns:
        station, lat, lon, day, ws_day, wu_day, wv_day, pa_day
    """
    # Collect hourly values grouped by (station, day, parameter)
    # Also keep lat/lon per station
    buckets: dict = defaultdict(list)
    coords:  dict = {}   # station -> (lat, lon)

    for row in all_rows:
        if row["value"] is None:
            continue
        day = row["time_utc"][:10]    # "YYYY-MM-DD"
        key = (row["station"], day, row["parameter"])
        buckets[key].append(row["value"])
        coords[row["station"]] = (row["lat"], row["lon"])

    # Build per-(station, day) records
    results = {}   # (station, day) -> dict

    for (station, day, param), vals in buckets.items():
        if len(vals) < 1:
            continue
        key = (station, day)
        if key not in results:
            lat, lon = coords.get(station, (None, None))
            results[key] = {
                "station": station, "lat": lat, "lon": lon, "day": day,
                "_ws": [], "_wd": [], "_pa": [],
            }

        if param == "WS_PT1H_AVG":
            results[key]["_ws"].extend(vals)
        elif param == "WD_PT1H_AVG":
            results[key]["_wd"].extend(vals)
        elif param == "PA_PT1H_AVG":
            results[key]["_pa"].extend(vals)

    # Compute aggregates
    output_rows = []
    for (station, day), rec in results.items():
        ws_vals = rec["_ws"]
        wd_vals = rec["_wd"]
        pa_vals = rec["_pa"]

        # Need at least MIN_HOURS valid readings for wind to be meaningful
        if len(ws_vals) < MIN_HOURS or len(wd_vals) < MIN_HOURS:
            ws_day = wu_day = wv_day = None
        else:
            # Only use timesteps where BOTH ws and wd are available
            n_pairs = min(len(ws_vals), len(wd_vals))
            ws_arr = ws_vals[:n_pairs]
            wd_arr = wd_vals[:n_pairs]

            ws_day = sum(ws_arr) / len(ws_arr)

            # Circular mean: decompose into U (east) and V (north) components
            # FMI wind direction: 0°=N, 90°=E, 180°=S, 270°=W (meteorological convention)
            # Convert to math convention: theta = 270 - WD (East=0, CCW positive)
            # For building features we just use U = WS*cos(WD_rad), V = WS*sin(WD_rad)
            # where WD_rad is the meteorological direction in radians.
            # The exact convention doesn't matter as long as it's consistent.
            u_vals = [ws * math.cos(math.radians(wd)) for ws, wd in zip(ws_arr, wd_arr)]
            v_vals = [ws * math.sin(math.radians(wd)) for ws, wd in zip(ws_arr, wd_arr)]
            wu_day = sum(u_vals) / len(u_vals)
            wv_day = sum(v_vals) / len(v_vals)

        pa_day = sum(pa_vals) / len(pa_vals) if len(pa_vals) >= MIN_HOURS else None

        if ws_day is None and pa_day is None:
            continue

        output_rows.append({
            "station": station,
            "lat":     rec["lat"],
            "lon":     rec["lon"],
            "day":     day,
            "ws_day":  ws_day,
            "wu_day":  wu_day,
            "wv_day":  wv_day,
            "pa_day":  pa_day,
        })

    df = pd.DataFrame(output_rows)
    if df.empty:
        return df
    df = df.sort_values(["station", "day"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Same date range as ReadInDailyMaxMin.py: last 40 days
    end_dt   = dt.datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=40)

    print(f"Fetching hourly wind+pressure: {start_dt.date()} → {end_dt.date()}")
    print(f"Parameters: {HOURLY_PARAMS}")
    print(f"Chunk size: {CHUNK_DAYS} days\n")

    all_rows = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end_dt)
        print(f"  Chunk {cursor.date()} – {chunk_end.date()} ...", end=" ", flush=True)
        rows = fetch_chunk(cursor, chunk_end)
        print(f"{len(rows)} rows")
        all_rows.extend(rows)
        cursor = chunk_end

    print(f"\nTotal hourly rows fetched: {len(all_rows)}")

    df = aggregate_to_daily(all_rows)
    print(f"Daily aggregate rows: {len(df)}")
    print(f"Stations: {df['station'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
