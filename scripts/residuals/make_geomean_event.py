#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geomean rows for ONE new event, in the exact format of geomean_pooled.csv,
so they can be pasted at the end of that file.

Input : the event's 2-D NLTH JSON
        ({building: {RSN: {"dir_1"|"dir_2": {"max_psd", "IM_data"}}}})
        and its station CSV (RSN or "Record Sequence Number", EQID,
        st_latitude, st_longitude, PGA).
Output: geomean_event_{eqid}.csv with one row per building-station:
        eqid, RSN, building, latitude, longitude, PGA, period, Say,
        EDP = sqrt(PSD_1 * PSD_2), IM_<name> = sqrt(IM_1 * IM_2).

Stations with PGA <= PGA_MIN are dropped, as in calculate_eps-geomean.py.
The rows are threshold- and fragility-independent: the main script computes
D and p from them for whichever damage state it runs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# ===========================================================================
# CONFIG -- edit for each new event
# ===========================================================================

RESULTS_JSON = Path("/path/to/event_results-2D.json")
STATIONS_CSV = Path("/path/to/stations_event.csv")
PGA_MIN      = 0.05     # keep stations with PGA > PGA_MIN (0.05 for all events so far)

HERE          = Path(__file__).parent
MODAL_JSON    = DATA / "portfolio" / "modal_results.json"
BILINEAR_JSON = DATA / "portfolio" / "bilinear_results.json"
POOLED_CSV    = DATA / "pooled" / "geomean_pooled.csv"   # only read, to check the format


def read_stations(path):
    st = pd.read_csv(path, encoding="utf-8-sig")
    st = st.rename(columns={"Record Sequence Number": "RSN", "EQID": "eqid",
                            "st_latitude": "latitude", "st_longitude": "longitude"})
    missing = {"RSN", "eqid", "latitude", "longitude", "PGA"} - set(st.columns)
    if missing:
        raise ValueError(f"{path}: missing station columns {sorted(missing)}")
    if st["eqid"].nunique() != 1:
        raise ValueError(f"{path}: expected one EQID, found {sorted(st['eqid'].unique())}")
    if st["RSN"].duplicated().any():
        raise ValueError(f"{path}: duplicated RSNs")
    return st[["eqid", "RSN", "latitude", "longitude", "PGA"]]


def geomean_rows(results_json, stations_csv, pga_min):
    stations = read_stations(stations_csv)
    with open(results_json) as f:
        results = json.load(f)

    rows = []
    for building, per_rsn in results.items():
        for rsn, rec in per_rsn.items():
            d1, d2 = rec["dir_1"], rec["dir_2"]
            row = {"RSN": int(rsn), "building": building,
                   "EDP": np.sqrt(d1["max_psd"] * d2["max_psd"])}
            for im in d1["IM_data"]:
                row[f"IM_{im}"] = np.sqrt(d1["IM_data"][im] * d2["IM_data"][im])
            rows.append(row)

    df = stations.merge(pd.DataFrame(rows), on="RSN", how="inner")
    n_json = len({int(r) for per_rsn in results.values() for r in per_rsn})
    print(f"{df['RSN'].nunique()} of {n_json} JSON records matched a station "
          f"({len(stations)} stations in CSV)")

    n0 = df["RSN"].nunique()
    df = df[df["PGA"] > pga_min]
    print(f"{df['RSN'].nunique()} of {n0} stations with PGA > {pga_min}")

    with open(MODAL_JSON) as f:
        modal = json.load(f)
    with open(BILINEAR_JSON) as f:
        bilinear = json.load(f)
    missing = sorted(b for b in set(df["building"]) if b not in modal or b not in bilinear)
    if missing:
        raise ValueError(f"no period / Say for buildings {missing[:5]}{'...' if len(missing) > 5 else ''}")
    df["period"] = df["building"].map(lambda b: modal[b]["MX"]["period"])
    df["Say"]    = df["building"].map(lambda b: bilinear[b]["X"]["force"][1])

    im_cols = [c for c in df.columns if c.startswith("IM_")]
    return df[["eqid", "RSN", "building", "latitude", "longitude", "PGA",
               "period", "Say", "EDP"] + im_cols].reset_index(drop=True)


if __name__ == "__main__":
    df = geomean_rows(RESULTS_JSON, STATIONS_CSV, PGA_MIN)
    eqid = int(df["eqid"].iloc[0])

    # Same columns, in the same order, as the pooled file it will be pasted into
    if POOLED_CSV.exists():
        pooled = pd.read_csv(POOLED_CSV, usecols=["eqid"])
        pooled_cols = list(pd.read_csv(POOLED_CSV, nrows=0).columns)
        if list(df.columns) != pooled_cols:
            raise ValueError(f"columns differ from {POOLED_CSV.name}:\n"
                             f"  new   : {list(df.columns)}\n  pooled: {pooled_cols}")
        if eqid in set(pooled["eqid"]):
            print(f"WARNING: eqid {eqid} is already in {POOLED_CSV.name} -- "
                  f"pasting these rows would duplicate it")

    out = HERE / f"geomean_event_{eqid}.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df['RSN'].nunique()} stations x "
          f"{df['building'].nunique()} buildings) for eqid {eqid} to {out}")
    print(f"Paste every line except the header at the end of {POOLED_CSV.name}.")
