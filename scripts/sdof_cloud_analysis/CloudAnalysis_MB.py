#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:52:36 2025

@author: Tomas Mejia
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from pathlib import Path
from nlth_on_sdof import run_nlth_analysis_on_sdof_ops_py

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


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

    rec_names = np.loadtxt(DATA / "cloud_analysis" / "names.txt", dtype=str)

    psd = []
    disp_SDOF = []
    for record in rec_names:
        rec_dir = os.path.join(records_dir, record)
        gmr = np.loadtxt(rec_dir)

        outputs = run_nlth_analysis_on_sdof_ops_py(capacity_curve, gmr, 0.05, degradation=False)
        disps = outputs["rel_disp"]

        max_disp = np.max(np.abs(disps))
        disp_shape = max_disp * gamma * eigenvector
        rel_disp = np.diff(disp_shape, prepend=0)
        drift = rel_disp / interstory_heights * 100
        psd.append(np.max(drift))
        disp_SDOF.append(max_disp)

    # Double the record amplitude
    for record in rec_names:
        rec_dir = os.path.join(records_dir, record)
        gmr = np.loadtxt(rec_dir)
        gmr[:,1] = 2 * gmr[:,1]

        outputs = run_nlth_analysis_on_sdof_ops_py(capacity_curve, gmr, 0.05, degradation=False)
        disps = outputs["rel_disp"]

        max_disp = np.max(np.abs(disps))
        disp_shape = max_disp * gamma * eigenvector
        rel_disp = np.diff(disp_shape, prepend=0)
        drift = rel_disp / interstory_heights * 100
        psd.append(np.max(drift))
        disp_SDOF.append(max_disp)
        
    #Increase the record amplitude again
    for record in rec_names:
        rec_dir = os.path.join(records_dir, record)
        gmr = np.loadtxt(rec_dir)
        gmr[:,1] = 6 * gmr[:,1]

        outputs = run_nlth_analysis_on_sdof_ops_py(capacity_curve, gmr, 0.05, degradation=False)
        disps = outputs["rel_disp"]

        max_disp = np.max(np.abs(disps))
        disp_shape = max_disp * gamma * eigenvector
        rel_disp = np.diff(disp_shape, prepend=0)
        drift = rel_disp / interstory_heights * 100
        psd.append(np.max(drift))
        disp_SDOF.append(max_disp)
    
    

    PSD[building] = psd
    DISP_SDOF[building] = disp_SDOF

    print(f'Analysis for {building} done.')
    return PSD, DISP_SDOF


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(path, "portfolio")
    records_dir = os.path.join(path, "records-tomas")

    # Load inputs
    with open(DATA / "portfolio" / "modal_results.json", "r") as f:
        modal_properties = json.load(f)

    with open(DATA / "portfolio" / "bilinear_results.json", "r") as f:
        bilinear_properties = json.load(f)

    with open(DATA / "portfolio" / "storey_heights.json", "r") as f:
        st_heights = json.load(f)

    buildings = list(bilinear_properties.keys())  # or your custom subset

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
    with open(os.path.join(path, "cloud_analysis_psd.json"), "w") as f:
        json.dump(PSD, f, indent=4)

    with open(os.path.join(path, "cloud_analysis_sdof_disp.json"), "w") as f:
        json.dump(DISP_SDOF, f, indent=4)

    print("All analyses completed successfully!")
