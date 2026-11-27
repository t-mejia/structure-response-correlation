# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:17:07 2025

@author: Tomas Mejia
"""

from pathlib import Path
import numpy as np
import json
import csv
from scipy.stats import norm
from typing import List, Dict
import random
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def process_psd(edp, im, demand_intensity):
    
    slope = demand_intensity['slope']
    intercept = demand_intensity['intercept']
    std = demand_intensity['beta']

    median = slope * np.log(im) + intercept
    eps = (np.log(edp) - median) / std
    
    return eps


def process_damage(edp, im, fragility, damage_drift):
    mu = fragility['mu']
    sigma = fragility['beta']

    p = norm.cdf(np.log(im), loc=mu, scale=sigma)
    if edp < damage_drift:
        damage = 0
        eps = norm.ppf(1 - p)
    else:
        damage = 1
        eps = norm.ppf(p)
    
    return eps


def save_filtered_stations_to_csv(filtered_stations: List[Dict], output_path: Path) -> None:
    """
    Save a list of filtered station dictionaries to a CSV file.

    Parameters:
        filtered_stations (List[Dict]): List of station data dictionaries.
        output_path (Path): Full path (including filename) where the CSV will be saved.
    """
    if not filtered_stations:
        print("No data to save: 'filtered_stations' is empty.")
        return

    fieldnames = filtered_stations[0].keys()

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_stations)
    print(f"Filtered stations saved to: {output_path}")


directory = Path(__file__).parent

# Write "PSD" to correlate drifts or "Damage" to correlate damage directly
type_corr = "PSD"
collapse_drift = 5.0  # PSD at which the structure is assumed to have collapsed
damage_drift = 0.5  # Include PSD threshold to use in case of correlating damage

im_type = "AvgSa2"

station_csv_path = DATA / "events" / "chuetsu" / "stations_chuetsu.csv"
with open(station_csv_path, 'r', encoding='utf-8-sig', newline='') as csv_file:
    reader = csv.DictReader(csv_file)
    stations = list(reader)

# Filter by PGA > 0.05
filtered_stations = [row for row in stations if float(row["PGA"]) > 0.05]

#filtered_stations = stations   

edp_file = DATA / "events" / "chuetsu" / "chuetsu_results-2D.json"

with open(edp_file, 'r') as json_file:
    drift_dict = json.load(json_file)

buildings=list(drift_dict.keys())



outputs = []
for row in filtered_stations:
    rsn = row['RSN']
    lat = row['st_latitude']
    lon = row['st_longitude']
    
    output_row = {
        "RSN": rsn,
        "latitude": lat,
        "longitude": lon
    }
    
    for building in buildings:
        edp1 = drift_dict[building][str(rsn)]["dir_1"]['max_psd']
        edp2 = drift_dict[building][str(rsn)]["dir_2"]['max_psd']
        
        im1 = drift_dict[building][str(rsn)]["dir_1"]['IM_data'][im_type]
        im2 = drift_dict[building][str(rsn)]["dir_2"]['IM_data'][im_type]
        
        edp_i = (edp1*edp2)**0.5
        
        im_i = (im1*im2)**0.5
        
        if type_corr == "PSD":
            d_i_path = DATA / "models" / "downstream" / "demand_intensity.json"

            with open(d_i_path, 'r') as json_file:
                demand_intensity = json.load(json_file) 
            
            d_i = demand_intensity[im_type]    
            if edp_i > collapse_drift:
                eps = "NaN"
            else:
                eps = process_psd(edp_i, im_i, d_i)
            
        elif type_corr == "Damage":
             frag = load_json(DATA / "models" / "downstream" / f'fragility_{damage_drift}.json')
             fragility = frag[im_type]

             
             eps = process_damage(edp_i, im_i, fragility, damage_drift) 
             
        output_row[f"eps_{building}"] = eps
    
    outputs.append(output_row)
        
# ✅ Build fieldnames dynamically
fieldnames = ["RSN", "latitude", "longitude"] + [f"eps_{b}" for b in buildings]
outputs_dir = directory / f"epsilon-2D/{im_type}"
outputs_dir.mkdir(parents=True, exist_ok=True)

if type_corr == "PSD":
    output_path = outputs_dir / f"eps_{type_corr}-geomean.csv"
    
else:
    output_path = outputs_dir / f"eps_{type_corr}_{damage_drift}-geomean.csv"

# ✅ Write CSV
with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(outputs)

print(f"✅ CSV saved successfully: {output_path}")    


        
"""
with open(dir_stations, 'r', newline='', encoding='utf-8-sig') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        stations.append({
            'st_latitude': float(row['st_latitude']),
            'st_longitude': float(row['st_longitude']),
            'im': float(row['Sa_Avg']),
            'PGA': float(row['PGA']),
            'PSD': float(row['Drift_Value'])
        })


if type_corr == "PSD":
    dc = load_json(directory / "demand_capacity_GEM_PGA.json")
    filtered_stations = process_psd(stations, building, dc, collapse_drift)
    output_file = directory / "epsilon_EDP_tax" / 
        f'stations_drift_{building_number}.csv'
    save_filtered_stations_to_csv(filtered_stations, output_file)

elif type_corr == "Damage":
    frag = load_json(directory / f'fitted_params{damage_drift}.json')
    filtered_stations = process_damage(stations, building, frag, damage_drift)
    output_file = directory / "epsilon" / \
        f'stations_damage{damage_drift}_{building_number}.csv'
    save_filtered_stations_to_csv(filtered_stations, output_file)

else:
    raise ValueError('Variable type_corr must be equal to "PSD" or "Damage"')

"""