# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:50:01 2025

@author: Tomas Mejia
"""
from pathlib import Path
from scipy import stats
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, curve_fit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

def total_exceedance_prob_logspace(x, edp_k, a, b, beta_edp, b0, b1):
    """
    P(EDP > edp_k | IM = im), computed with x = ln(im)
    P(C|IM) now uses a logistic model: P(C|IM) = 1 / (1 + exp(b0 - b1*x))
    """
    mu_lnEDP = a + b * x
    P_exceed_NC = 1 - norm.cdf((np.log(edp_k) - mu_lnEDP) / beta_edp)
    P_C = 1 / (1 + np.exp(b0 - b1 * x))
    return P_exceed_NC * (1 - P_C) + P_C

def solve_lnim_for_p(p, edp_k, a, b, beta_edp, b0, b1,
                      im_lo=1e-4, im_hi=30.0):
    """
    Find x = ln(im) such that P(EDP > edp_k | IM=im) = p.
    Returns x_star = ln(im_star) directly (NOT converted back to linear im).
    """
    x_lo, x_hi = np.log(im_lo), np.log(im_hi)

    f = lambda x: total_exceedance_prob_logspace(
        x, edp_k, a, b, beta_edp, b0, b1
    ) - p

    f_lo, f_hi = f(x_lo), f(x_hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            f"No sign change in bracket [ln(im_lo)={x_lo:.4f}, ln(im_hi)={x_hi:.4f}] "
            f"(f_lo={f_lo:.4f}, f_hi={f_hi:.4f}). Widen the bracket."
        )

    x_star = brentq(f, x_lo, x_hi, xtol=1e-10, rtol=1e-12)
    return x_star

def read_txt_to_list(file_path):
    """
    Reads a .txt file from the given directory and converts its contents into a list.

    :param file_path: Path to the .txt file
    :return: List containing lines from the file
    """
    try:
        with open(file_path, 'r') as file:
            # Read all lines from the file and strip newline characters
            lines = [line.strip() for line in file.readlines()]
        return lines
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []


def save_plot(filename):
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()


def regression_model(ln_im, b0, b1):
    return 1/(1+np.exp(b0-b1*ln_im))


directory = Path(__file__).parent
edp_dir = DATA / "cloud_analysis" / "cloud_analysis_psd.json"
im_dir = DATA / "cloud_analysis" / "ground_motion_IMs.csv"
buildings_dir = DATA / "portfolio" / "buildings.txt"
output_dir = directory / "figs_tax"

collapse_drift = 5.0

# Three predefined (non-collapse) damage thresholds -- matching DS1-DS3 in
# validation_Comparison-Geomean-multiDS.py. DS4 is collapse (collapse_drift).
damage_threshold_1 = 0.5
damage_threshold_2 = 1.0
damage_threshold_3 = 2.0
damage_thresholds = [damage_threshold_1, damage_threshold_2, damage_threshold_3]

building_names = read_txt_to_list(buildings_dir)
selected_building = building_names[49]  # set to the building to plot

# Same per-building folder used for the demand-intensity plots.
building_dir = output_dir / selected_building
building_dir.mkdir(parents=True, exist_ok=True)

with open(edp_dir, 'r') as file:
    edps = json.load(file)

#import IMS
intensity = pd.read_csv(im_dir, sep=",")

im_types = intensity.columns.tolist()

max_edp = edps[selected_building]

for im_type in im_types:

    ims = intensity[im_type]

    #filter for collapse drift (non-collapse demand model)
    edp_filtered = []
    im_filtered = []

    for xi, yi in zip(max_edp, ims):
        if xi <= collapse_drift:
            edp_filtered.append(xi)
            im_filtered.append(yi)

    #Estimate logarithm
    ln_edp = np.log(edp_filtered)
    ln_im = np.log(im_filtered)

    #Linear regression (demand model)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_im, ln_edp)

    #Calculate vertical dispersion
    regression = slope*ln_im + intercept
    error = (regression-ln_edp)**2
    beta_edp = ((1/(len(error)-1))*np.sum(error))**0.5

    # Collapse indicator (1 if EDP > collapse_drift, else 0) vs ln(IM), over
    # all records -- this logistic fit is used both to correct the DS1-DS3
    # exceedance probabilities below for collapsed cases, and (copied from
    # collapse_fragility.py) to get DS4's own fragility directly.
    collapse_bin = [0 if xi <= collapse_drift else 1 for xi in max_edp]
    ln_im_all = np.log(ims)

    params, _ = curve_fit(regression_model, ln_im_all, collapse_bin, p0=[0.0, 1.0])
    b0_fit, b1_fit = params

    # DS4 (collapse) fragility -- from collapse_fragility.py.
    mu4 = b0_fit / b1_fit
    percent_16 = (b0_fit - np.log((1-0.16)/0.16)) / b1_fit
    percent_84 = (b0_fit - np.log((1-0.84)/0.84)) / b1_fit
    beta4 = (percent_84 - percent_16) / 2

    # DS1-DS3 fragility -- median/dispersion of ln(IM) at which the combined
    # (non-collapse EDP + collapse) exceedance probability of each threshold
    # reaches 0.5/0.16/0.84.
    fragility_params = []
    for edp_k in damage_thresholds:
        lnim_50 = solve_lnim_for_p(0.50, edp_k, intercept, slope, beta_edp, b0_fit, b1_fit)
        lnim_16 = solve_lnim_for_p(0.16, edp_k, intercept, slope, beta_edp, b0_fit, b1_fit)
        lnim_84 = solve_lnim_for_p(0.84, edp_k, intercept, slope, beta_edp, b0_fit, b1_fit)
        fragility_params.append((lnim_50, (lnim_84-lnim_16)/2))
    fragility_params.append((mu4, beta4))  # DS4 = collapse

    # --- plotting ---
    # Sized for three plots per line in the paper.
    im_grid = np.linspace(ims.min()*0.5, ims.max()*1.1, 300)
    ln_im_grid = np.log(im_grid)

    labels = [r"$DS_1$", r"$DS_2$", r"$DS_3$", r"$DS_4$"]
    colors = ["tab:green", "tab:orange", "tab:red", "black"]

    plt.figure(figsize=(2.4, 2.2), dpi=500)
    for (mu, beta), label, color in zip(fragility_params, labels, colors):
        prob = norm.cdf((ln_im_grid - mu) / beta)
        plt.plot(im_grid, prob, label=label, color=color)

    plt.xlabel(rf"${im_type}$", fontsize=8)
    plt.ylabel("Probability of exceedance", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, 1)
    plt.xlim(0, 3)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    save_plot(building_dir / f'fragility_{im_type}.pdf')
