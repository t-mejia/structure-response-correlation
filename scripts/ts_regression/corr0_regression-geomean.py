#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 11:27:28 2025

@author: Tomas Mejia
"""

import csv
import math
from pathlib import Path
import math
from itertools import combinations
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from scipy.optimize import minimize_scalar
import numpy as np
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def figure_with_axes_size(axes_width_in, axes_height_in,
                          margins=(0.8, 0.6, 0.2, 0.2), dpi=400):
    """
    Create a matplotlib figure with a given inner (axes) size in inches.
    margins = (left, bottom, right, top) in inches.
    """
    left_margin, bottom_margin, right_margin, top_margin = margins
    fig_width = axes_width_in + left_margin + right_margin
    fig_height = axes_height_in + bottom_margin + top_margin
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    fig.subplots_adjust(
        left=left_margin / fig_width,
        right=1 - right_margin / fig_width,
        bottom=bottom_margin / fig_height,
        top=1 - top_margin / fig_height
    )
    return fig, ax

def model1(X, w, a, b):
    dT, dS = X
    return w * np.exp(-dT / a) + (1 - w) * np.exp(-dS / b)

def model2(X, w, a, b):
    dT, dS = X
    return w * np.exp(-dT / a) + (1 - w) * np.cos(b * dS)

def modelTSm(X, a, b):
    """Model TSm: multiplicative form, high correlation only if similar in BOTH T and Say."""
    dT, dS = X
    return np.exp(-dT / a) * np.exp(-dS / b)

# Models to fit: name -> (function, parameter names, initial guess, bounds)
# Model TS keeps the original output file names; the others get a "-<name>" suffix.
MODELS = {
    "TS":  (model1,   ["w", "a", "b"], [0.5, 0.5, 0.5],
            ([0.0, 1e-6, 1e-6], [1.0, np.inf, np.inf])),
    "TSm": (modelTSm, ["a", "b"],      [0.5, 0.5],
            ([1e-6, 1e-6], [np.inf, np.inf])),
}
MODELS_TO_FIT = ["TS", "TSm"]
# Load station data
directory = Path(__file__).parent


type_corr = "PSD"
output_dir = directory / "AvgSa2"
output_dir.mkdir(parents=True, exist_ok=True)
buildings = np.loadtxt(DATA / "portfolio" / "buildings.txt", dtype=str)

with open(DATA / "portfolio" / "bilinear_results.json", "r") as f:
    bilinear_properties = json.load(f)
    
with open(DATA / "portfolio" / "modal_results.json", "r") as f:
    modal_properties = json.load(f)


plt.rcParams.update({
    "font.family": "serif",           
    "font.serif": ["Times New Roman"], 
})

deltaT_list = []
deltaSay_list = []
corr_list = []

for i, b1 in enumerate(buildings):
    T1 = modal_properties[b1]["MX"]["period"]    
    Say1 = bilinear_properties[b1]["X"]["force"][1]

    for j, b2 in enumerate(buildings):
        T2 = modal_properties[b2]["MX"]["period"]
        Say2 = bilinear_properties[b2]["X"]["force"][1]
        
        deltaT = abs(T2 - T1)
        deltaSay = abs(Say2 - Say1)

        # Load CSV paths
        if type_corr == "PSD":
            csv_path1 = DATA / "events" / "chichi" / "epsilon-2D" / "AvgSa2" / "eps_PSD-geomean.csv"
            csv_path2 = DATA / "events" / "chichi" / "epsilon-2D" / "AvgSa2" / "eps_PSD-geomean.csv"
        else:
            raise ValueError("Only PSD handled in this version")

        # Read eps for both buildings — optimized matching logic
        eps1, eps2 = [], []
        with open(csv_path1, 'r', newline='', encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    e1 = float(row[f'eps_{b1}'])
                    e2 = float(row[f'eps_{b2}'])
                    if np.isfinite(e1) and np.isfinite(e2):
                        eps1.append(e1)
                        eps2.append(e2)
                except:
                    continue

        if len(eps1) > 1 and len(eps2) > 1:
            corr_ij = np.corrcoef(eps1, eps2)[0, 1]
            if np.isfinite(corr_ij):
                deltaT_list.append(deltaT)
                deltaSay_list.append(deltaSay)
                corr_list.append(corr_ij)
                
        
# Convert to arrays
deltaT_arr = np.array(deltaT_list)
corr_obs = np.array(corr_list)
deltaSay_arr = np.array(deltaSay_list)

# === prepare input and mask invalid entries ===
mask = np.isfinite(deltaT_arr) & np.isfinite(deltaSay_arr) & np.isfinite(corr_obs)
xdata = np.vstack((deltaT_arr[mask], deltaSay_arr[mask]))
y = corr_obs[mask]

# Define number of bins for averaging (slice plots)
n_bins = 10
colors = cm.magma(np.linspace(0, 1, 5))
#colors = np.flip(colors)

fits = {}
for model_name in MODELS_TO_FIT:
    model, par_names, p0, bounds = MODELS[model_name]
    suffix = "" if model_name == "TS" else f"-{model_name}"

    popt, pcov = curve_fit(model, xdata, y, p0=p0, bounds=bounds, maxfev=10000)

    # parameter std dev (1-sigma)
    perr = np.sqrt(np.diag(pcov))

    print(f"\n=== Model {model_name} ===")
    print("Fitted parameters:")
    for name, v, e in zip(par_names, popt, perr):
        print(f"  {name} = {v:.4f} ± {e:.4f}")

    # === Diagnostics: predicted vs observed and RMSE ===
    y_pred = model(xdata, *popt)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    print(f"RMSE = {rmse:.4e}")

    ss_res = np.sum((y - y_pred)**2)          # residual sum of squares
    ss_tot = np.sum((y - np.mean(y))**2)      # total sum of squares
    r2 = 1 - ss_res / ss_tot
    print(f"R² = {r2:.4f}")

    # AIC for least squares (Gaussian residuals): n ln(SS_res / n) + 2k
    aic = len(y) * np.log(ss_res / len(y)) + 2 * len(popt)
    print(f"AIC = {aic:.1f}")
    fits[model_name] = dict(popt=popt, perr=perr, rmse=rmse, r2=r2, aic=aic)

    plt.figure(figsize=(6,4))
    plt.scatter(y, y_pred, alpha=0.6)
    lims = (min(y.min(), y_pred.min()), max(y.max(), y_pred.max()))
    plt.plot(lims, lims, 'k--', linewidth=0.8)
    plt.xlabel("Observed correlation")
    plt.ylabel("Predicted correlation")
    plt.title(f"Observed vs Predicted (Model {model_name})")
    plt.grid(True)
    plt.show()

    # ---- 1️⃣ Slices: correlation vs ΔT for fixed ΔSay (with fitted model) ----
    plt.figure(figsize=(4,3))
    ds_levels = [0.0, np.percentile(deltaSay_arr, 25),
                 np.percentile(deltaSay_arr, 50), np.percentile(deltaSay_arr, 75)]

    for i, ds in enumerate(ds_levels):
        color = colors[i % len(colors)]
        tol = 0.05 * np.max(deltaSay_arr)
        mask = (deltaSay_arr >= ds - tol) & (deltaSay_arr <= ds + tol)
        if not np.any(mask):
            continue

        df = pd.DataFrame({"ΔT": deltaT_arr[mask], "corr": corr_obs[mask]})
        df = df.sort_values("ΔT")
        bins = np.linspace(df["ΔT"].min(), df["ΔT"].max(), n_bins + 1)
        df["bin"] = np.digitize(df["ΔT"], bins)
        df["bin"] = np.clip(df["bin"], 1, len(bins)-1)

        avg_corr = df.groupby("bin")["corr"].mean()
        bin_indices = avg_corr.index.values
        bin_centers_valid = 0.5 * (bins[bin_indices - 1] + bins[bin_indices])

        # Plot binned data
        plt.plot(bin_centers_valid, avg_corr.values, 'o', color=color, label=f"ΔSay ≈ {ds:.3f}")

        # Plot fitted model
        dT_array = bin_centers_valid
        dS_array = np.full_like(dT_array, ds)
        model_vals = model((dT_array, dS_array), *popt)
        plt.plot(bin_centers_valid, model_vals, '-', color=color)

    plt.xlabel("ΔT")
    plt.xlim((0,0.20))
    plt.ylabel("Correlation")
    plt.ylim((0,1))
    #plt.title("Slices: correlation vs ΔT (for fixed ΔSay)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir/f'T-slices-geomean{suffix}.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    # ---- 2️⃣ Slices: correlation vs ΔSay for fixed ΔT (with fitted model) ----
    plt.figure(figsize=(4,3))
    dt_levels = [0.0, np.percentile(deltaT_arr, 25),
                 np.percentile(deltaT_arr, 50), np.percentile(deltaT_arr, 75)]

    for i, dt in enumerate(dt_levels):
        color = colors[i % len(colors)]
        tol = 0.05 * np.max(deltaT_arr)
        mask = (deltaT_arr >= dt - tol) & (deltaT_arr <= dt + tol)
        if not np.any(mask):
            continue

        df = pd.DataFrame({"ΔSay": deltaSay_arr[mask], "corr": corr_obs[mask]})
        df = df.sort_values("ΔSay")
        bins = np.linspace(df["ΔSay"].min(), df["ΔSay"].max(), n_bins + 1)
        df["bin"] = np.digitize(df["ΔSay"], bins)
        df["bin"] = np.clip(df["bin"], 1, len(bins)-1)

        avg_corr = df.groupby("bin")["corr"].mean()
        bin_indices = avg_corr.index.values
        bin_centers_valid = 0.5 * (bins[bin_indices - 1] + bins[bin_indices])

        # Plot binned data
        plt.plot(bin_centers_valid, avg_corr.values, 'o', color=color, label=f"ΔT ≈ {dt:.3f}")

        # Plot fitted model
        dS_array = bin_centers_valid
        dT_array = np.full_like(dS_array, dt)
        model_vals = model((dT_array, dS_array), *popt)
        plt.plot(bin_centers_valid, model_vals, '-', color=color)

    plt.xlabel("ΔSay")
    plt.ylabel("Correlation")
    plt.ylim((0,1))
    #plt.title("Slices: correlation vs ΔSay (for fixed ΔT)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir/f'Sa_y-slices-geomean{suffix}.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    ######################3D PLOT ####################
    fig = plt.figure(figsize=(6, 6))  # rectangular window
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 0.8), zoom=0.90)

    dT_grid, dS_grid = np.meshgrid(
        np.linspace(min(deltaT_list), max(deltaT_list), 50),
        np.linspace(min(deltaSay_list), max(deltaSay_list), 50)
    )

    corr_grid = model((dT_grid, dS_grid), *popt)
    # Scatter observed data

    ax.scatter(deltaT_arr, deltaSay_arr, corr_obs, color='gray', label='Observed', alpha=0.8)

    # Model surface
    ax.plot_surface(dT_grid, dS_grid, corr_grid, cmap='magma_r', alpha=0.8, edgecolor='none')

    ax.set_xlabel(r'$\Delta T$')
    ax.set_ylabel(r'$\Delta S_{ay}$')
    ax.set_zlabel('Correlation', labelpad=1)


    plt.subplots_adjust(left=0.05, right=0.75, bottom=0.1, top=0.95)
    plt.savefig(output_dir/f'3d_plot-geomean{suffix}.pdf', dpi=400, bbox_inches='tight')
    plt.show()

# === Comparison of the fitted models ===
print("\n=== Model comparison (lower AIC is better) ===")
for model_name, f in fits.items():
    pars = ", ".join(f"{n}={v:.4f}" for n, v in zip(MODELS[model_name][1], f["popt"]))
    print(f"  {model_name:4s}: {pars} | RMSE={f['rmse']:.4e}, R²={f['r2']:.4f}, AIC={f['aic']:.1f}")
