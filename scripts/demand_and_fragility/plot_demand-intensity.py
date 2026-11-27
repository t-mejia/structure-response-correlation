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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

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


directory = Path(__file__).parent
edp_dir = DATA / "cloud_analysis" / "cloud_analysis_psd.json"
im_dir = DATA / "cloud_analysis" / "ground_motion_IMs.csv"
buildings_dir = DATA / "portfolio" / "buildings.txt"
output_dir = directory / "figs_tax"

collapse_drift = 5.0

building_names = read_txt_to_list(buildings_dir)
selected_building = building_names[49]  # set to the building to plot

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

    #filter for collapse drift
    edp_filtered = []
    im_filtered = []

    for xi, yi in zip(max_edp, ims):
        if xi <= collapse_drift:
            edp_filtered.append(xi)
            im_filtered.append(yi)

    #Estimate logarithm
    ln_edp = np.log(edp_filtered)
    ln_im = np.log(im_filtered)

    #Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(ln_im, ln_edp)

    #Calculate vertical dispersion
    regression = slope*ln_im + intercept
    error = (regression-ln_edp)**2
    beta = ((1/(len(error)-1))*np.sum(error))**0.5

    edp_plusbeta = regression + beta
    edp_minusbeta = regression - beta

    textstr = (
             rf'$\alpha_0 = {intercept:.2f}$'
            + '\n'
            + rf'$\alpha_1 = {slope:.2f}$'
            + '\n'
            + rf'$R^2 = {r_value:.3f}$'
            + '\n'
            + rf'$\beta_{{EDP}} = {beta:.3f}$'
        )

    # --- plotting ---
    # Sized for three plots per line in the paper.
    plt.figure(figsize=(2.4, 2.2), dpi=500)
    plt.scatter(ln_edp, ln_im, color="grey", s=8)
    plt.plot(regression, ln_im, color="black")
    plt.plot(edp_plusbeta, ln_im, color="red", linestyle="--")
    plt.plot(edp_minusbeta, ln_im, color="red", linestyle="--")
    plt.xlabel(r"Demand, $\ln EDP$", fontsize=8)
    plt.grid()
    plt.ylabel(rf"Intensity, $\ln {im_type} $", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    # place text in axes coordinates
    plt.text(
        0.02, 0.98,
        textstr,
        fontsize=7,
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )

    plt.tight_layout()
    save_plot(building_dir / f'demand_capacity_{im_type}.pdf')
