#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:52:36 2025

@author: Tomas Mejia
"""

import numpy as np
import re
import json
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from nlth_on_sdof import run_nlth_analysis_on_sdof_ops_py
import eqsig
import pandas as pd
from scipy.stats import gmean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def get_SaT(a, dt, damp, T):
	"""
	Details:
	Compute the spectral acceleration at period T with damping damp of a ground motion signal

	Information:
	Author: Gerard J. O'Reilly
	First Version: March 2021

	Notes:
	Requires eqsig

	References:

	Inputs:
	acc: Acceleration time series
	dt:	Time step of signal
	damp: Damping [0,1.0]
	T: Period to compute for [s]

	Returns:
	SaT: Value of IM

	"""


	_, _, SaT = eqsig.sdof.pseudo_response_spectra(a, dt, [T], xi=damp)

	return SaT

def get_AvgSa2(a, dt, damp, T):

    T_min = 0.2 * T
    T_max = 2.0 * T
    
    T_list = np.linspace(T_min, T_max, num=10)
    
    Sa_list = []
    
    _, _, SaT = eqsig.sdof.pseudo_response_spectra(a, dt, T_list, xi=damp)
    
    AvgSa2 = gmean(SaT)
    return AvgSa2

def extract_rsn(filename: str) -> int:
    match = re.search(r"RSN(\d+)_", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No RSN number found in filename: {filename}")

def analyze_building(building, path, parent_dir, records_dir, modal_properties, bilinear_properties, st_heights):
    """Run nonlinear time history analysis for a single building."""
    
    PSD = {}
    DISP_SDOF = {}

    force = bilinear_properties[building]['X']['force']
    disp = bilinear_properties[building]['X']['displacement']
    capacity_curve = np.column_stack((disp, force))

    gamma = modal_properties[building]['MX']['factor']
    
    eigenvector = np.array(modal_properties[building]['MX']['eigenvector'])
    storey_heights = st_heights[building]
    interstory_heights = np.diff(storey_heights, prepend=0)

    rec_names1 = np.loadtxt(DATA / "records" / "chuetsu" / "names_dir1.txt", dtype=str)
    rec_names2 = np.loadtxt(DATA / "records" / "chuetsu" / "names_dir2.txt", dtype=str)
    rec_dt = np.loadtxt(DATA / "records" / "chuetsu" / "dts.txt")

    psd = {}
    disp_SDOF = {}
    
    for i, record in enumerate(rec_names1):
        rec_dir = os.path.join(records_dir, record)
        
        rsn = extract_rsn(record)
        
        psd[rsn] = {}
        disp_SDOF[rsn] = {}
        
        acc = np.loadtxt(rec_dir)
        dt = rec_dt[i]
        time = np.arange(len(acc)) * dt
        
        gmr = np.column_stack((time, acc)) 

        outputs = run_nlth_analysis_on_sdof_ops_py(capacity_curve, gmr, 0.05, degradation=False)
        disps = outputs["rel_disp"]

        max_disp = np.max(np.abs(disps))
        disp_shape = max_disp * gamma * eigenvector
        rel_disp = np.diff(disp_shape, prepend=0)
        drift = rel_disp / interstory_heights * 100
        max_drift = np.max(drift)
        psd[rsn]["dir_1"] = max_drift
        disp_SDOF[rsn]["dir_1"]=max_disp
        

    # Double the record amplitude
    
    for j, record in enumerate(rec_names2):
        rec_dir = os.path.join(records_dir, record)
        
        rsn = extract_rsn(record)
        
        acc = np.loadtxt(rec_dir)
        dt = rec_dt[j]
        time = np.arange(len(acc)) * dt
        
        gmr = np.column_stack((time, acc)) 
    

        outputs = run_nlth_analysis_on_sdof_ops_py(capacity_curve, gmr, 0.05, degradation=False)
        disps = outputs["rel_disp"]

        max_disp = np.max(np.abs(disps))
        disp_shape = max_disp * gamma * eigenvector
        rel_disp = np.diff(disp_shape, prepend=0)
        drift = rel_disp / interstory_heights * 100
        max_drift = np.max(drift)
        psd[rsn]["dir_2"] = max_drift
        disp_SDOF[rsn]["dir_2"]=max_disp
        

    PSD[building] = psd
    DISP_SDOF[building] = disp_SDOF

    print(f'Analysis for {building} done.')
    return PSD, DISP_SDOF


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(path, "portfolio")
    records_dir = os.path.join(path, "rec_Chuetsu")
    buildings_dir = DATA / "portfolio" / "buildings.txt"
    
    buildings = np.loadtxt(buildings_dir, dtype=str)

    # Load inputs
    with open(DATA / "portfolio" / "modal_results.json", "r") as f:
        modal_properties = json.load(f)

    with open(DATA / "portfolio" / "bilinear_results.json", "r") as f:
        bilinear_properties = json.load(f)

    with open(DATA / "portfolio" / "storey_heights.json", "r") as f:
        st_heights = json.load(f)


    # Run analyses in parallel
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(
            analyze_building,
            [(b, path, parent_dir, records_dir, modal_properties, bilinear_properties, st_heights)
             for b in buildings]
        )

    # Combine all results
    PSD = {}
    DISP_SDOF = {}
    for psd_res, disp_res in results:
        PSD.update(psd_res)
        DISP_SDOF.update(disp_res)

    # Save results
    with open(os.path.join(path, "chuetsu_psd.json"), "w") as f:
        json.dump(PSD, f, indent=4)

    with open(os.path.join(path, "chuetsu_sdof_disp.json"), "w") as f:
        json.dump(DISP_SDOF, f, indent=4)

    print("All analyses completed successfully!")
    
    #Get IMs
    
    rec_names1 = np.loadtxt(DATA / "records" / "chuetsu" / "names_dir1.txt", dtype=str)
    rec_names2 = np.loadtxt(DATA / "records" / "chuetsu" / "names_dir2.txt", dtype=str)
    rec_dt = np.loadtxt(DATA / "records" / "chuetsu" / "dts.txt")


    # Output containers
    above_threshold_dir1 = []
    above_threshold_dir2 = []
    above_threshold_dts = []

    T = 0.665
    T_ = [T]
    Sa_avg_1 = []
    Sa_T_1 = []
    PGA_1 = []
    Sa_01_1 =[]
    Sa_06_1 = []
    Sa_1_1 = []
    
    Sa_avg_2 = []
    Sa_T_2 = []
    PGA_2 = []
    Sa_01_2 =[]
    Sa_06_2 = []
    Sa_1_2 = []
    
    rsn1 = []
    rsn2 = []
    

    n=[]
    
    intensities = {}
    for i in range(len(rec_names1)):
        
        path1 = os.path.join(records_dir, rec_names1[i])
        ag=np.loadtxt(path1)
        rsn = extract_rsn(rec_names1[i])
        
        rsn1.append(rsn)
        
        dt_i = rec_dt[i]

        sa_avg = get_AvgSa2(ag, dt_i, 0.05, T)
        #sa_T = get_sa(ag, dt_i, T, 0.0)
        sa_T = get_SaT(ag, dt_i, 0.05, T)
        sa_Ti=sa_T[0]
        sa_01 = get_SaT(ag, dt_i, 0.05, 0.1)
        sa_01i=sa_01[0]
        sa_06 = get_SaT(ag, dt_i, 0.05, 0.6)
        sa_06i=sa_06[0]
        sa_1 = get_SaT(ag, dt_i, 0.05, 1.0)
        sa_1i=sa_1[0]
        
        pga = np.max(ag)
        
        Sa_avg_1.append(sa_avg)
        Sa_T_1.append(sa_Ti)
        Sa_01_1.append(sa_01i)
        Sa_06_1.append(sa_06i)
        Sa_1_1.append(sa_1i)
        PGA_1.append(pga)

    for i in range(len(rec_names2)):
        path2 = os.path.join(records_dir, rec_names2[i])
        ag=np.loadtxt(path2)
        rsn = extract_rsn(rec_names2[i])
        
        rsn2.append(rsn)
        
        dt_i = rec_dt[i]

        sa_avg = get_AvgSa2(ag, dt_i, 0.05, T)
        #sa_T = get_sa(ag, dt_i, T, 0.0)
        sa_T = get_SaT(ag, dt_i, 0.05, T)
        sa_Ti=sa_T[0]
        sa_01 = get_SaT(ag, dt_i, 0.05, 0.1)
        sa_01i=sa_01[0]
        sa_06 = get_SaT(ag, dt_i, 0.05, 0.6)
        sa_06i=sa_06[0]
        sa_1 = get_SaT(ag, dt_i, 0.05, 1.0)
        sa_1i=sa_1[0]
        
        pga = np.max(ag)
        
        Sa_avg_2.append(sa_avg)
        Sa_T_2.append(sa_Ti)
        Sa_01_2.append(sa_01i)
        Sa_06_2.append(sa_06i)
        Sa_1_2.append(sa_1i)
        PGA_2.append(pga)

    df_dir1= pd.DataFrame({
        "RSN": rsn1,
        "PGA": PGA_1,
        "AvgSa2": Sa_avg_1,
        f"Sa(T={T:.3f}s)": Sa_T_1,
        "Sa(0.1s)": Sa_01_1,
        "Sa(0.6s)": Sa_06_1,
        "Sa(1.0s)": Sa_1_1
    })
    
    df_dir2= pd.DataFrame({
        "RSN": rsn2,
        "PGA": PGA_2,
        "AvgSa2": Sa_avg_2,
        f"Sa(T={T:.3f}s)": Sa_T_2,
        "Sa(0.1s)": Sa_01_2,
        "Sa(0.6s)": Sa_06_2,
        "Sa(1.0s)": Sa_1_2
    })
    
    combined_results = {}
    
    for building, building_psd in PSD.items():
        combined_results[building] = {}
    
        for rsn, directions in building_psd.items():
            combined_results[building][rsn] = {}
    
            # Loop both directions
            for direction in ["dir_1", "dir_2"]:
                psd_value = directions.get(direction, -np.inf)
    
                # Select the dataframe depending on direction
                if direction == "dir_1":
                    row = df_dir1[df_dir1["RSN"] == rsn]
                else:
                    row = df_dir2[df_dir2["RSN"] == rsn]
    
                if row.empty:
                    print(f"⚠️ Warning: RSN {rsn} not found in IM dataset for {direction}")
                    continue
    
                # Convert IM row to dict
                im_data = row.iloc[0].drop("RSN").to_dict()
    
                # Save result
                combined_results[building][rsn][direction] = {
                    "max_psd": psd_value,
                    "IM_data": im_data
                }
        
        with open(os.path.join(path, "chuetsu_results-2D.json"), "w") as f:
            json.dump(combined_results, f, indent=4)

