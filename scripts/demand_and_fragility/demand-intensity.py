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
import os

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


def estimate_histogram(data: np.ndarray, bins: int = 50, title: str = "Histogram of Residuals", density: bool = False, output_dir: str = None):
    """
    Estimates and plots the histogram of an array of Monte Carlo simulation results.

    Args:
        data (np.ndarray): The array of simulation results (e.g., final outcomes).
        bins (int): The number of bins to use for the histogram. Default is 50.
        title (str): The title for the resulting plot.
        density (bool): If True, the histogram is normalized such that the integral 
                        over the range is 1 (approximating a Probability Density Function).
                        If False (default), the counts in each bin are plotted.
    """
    if data.size == 0:
        print("Error: Input data array is empty.")
        return
    
    data = data[~np.isnan(data)]

    mean_val = np.mean(data)
    std_val = np.std(data)

    data = data[(data >= mean_val - 4.5*std_val) & (data <= mean_val + 4.5*std_val)]

    # Calculate the histogram data
    # hist, bin_edges = np.histogram(data, bins=bins, density=density)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Use plt.hist directly for easier visualization and normalized bar plots
    plt.hist(
        data, 
        bins=bins, 
        density=density, 
        alpha=0.7, 
        color='#4A90E2', # Nice blue color
        edgecolor='black'
    )
    
    # Add descriptive labels and title
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Outcome Value", fontsize=14)
    if density:
        plt.ylabel("Probability Density", fontsize=14)
    else:
        plt.ylabel("Frequency (Counts)", fontsize=14)
        
    # Add a grid for better readability
    plt.grid(axis='y', alpha=0.5, linestyle='--')
    
    # Calculate and display basic statistics
    mean_val = np.mean(data)
    std_dev = np.std(data)
    
    # Add a vertical line for the mean
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
    
    # Add a text box with statistics
    stats_text = (
        f"Simulations: {len(data)}\n"
        f"Mean: {mean_val:.4f}\n"
        f"Std Dev: {std_dev:.4f}"
    )
    plt.text(
        0.05, 0.95, stats_text, 
        transform=plt.gca().transAxes, 
        verticalalignment='top', 
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6, ec="#333333")
    )

    plt.legend()
    

    
    # --- Save or show the plot ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"{title.replace(' ', '_')}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Histogram saved at: {filename}")
        plt.close()
    else:
        plt.show()
    
directory=Path(__file__).parent
edp_dir=DATA/"cloud_analysis"/"cloud_analysis_psd.json"
im_dir=DATA/"cloud_analysis"/"ground_motion_IMs.csv"
buildings_dir= DATA/"portfolio"/"buildings.txt"
output_dir =directory /"figs_tax"



collapse_drift=5.0

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
with open(edp_dir, 'r') as file:
    edps= json.load(file)

#import IMS
intensity = pd.read_csv(im_dir, sep=",")

im_types = intensity.columns.tolist()

# Labels for each building
building_names = read_txt_to_list(buildings_dir)

tax_dem_in = {}
bspecif_dem_in = {}

for im_type in im_types:

 
    #define demand-capscity dictionary
    demand_capacity={}
    combined_demand = []
    combined_capacity = []
    bspecif_dem_in[im_type]={}
    
    for building in building_names:
        demand_capacity[building]={}
        ims=intensity[im_type]
        max_edp=edps[building]
    
        #filter for collapse drift
        edp_filtered = []
        im_filtered = []
        
        for xi, yi in zip(max_edp, ims):
            if xi <= collapse_drift:
                edp_filtered.append(xi)
                im_filtered.append(yi)
        
        #Estimate logartithm
        ln_edp=np.log(edp_filtered)
        ln_im=np.log(im_filtered)
        
        #Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(ln_im, ln_edp)
        
        #Calculate vertical dispersion
        regression=slope*ln_im+intercept
        error=(regression-ln_edp)**2
        beta=((1/(len(error)-1))*np.sum(error))**0.5
        
        #Fill dictionary
        demand_capacity[building]["slope"]=slope
        demand_capacity[building]["intercept"]=intercept
        demand_capacity[building]["beta"]=beta
        
        output_dir_i = output_dir / im_type
        output_dir_i.mkdir(parents=True, exist_ok=True)
        
        edp_plusbeta = regression + beta 
        edp_minusbeta = regression - beta
        
        textstr = (
                r'$\ln \mathrm{EDP} = '
                + rf'{slope:.2f}\,\ln {im_type} + {intercept:.2f}$'
                + '\n'
                + rf'$R^2 = {r_value:.3f}$'
                + '\n'
                + rf'$\beta_{{EDP}} = {beta:.3f}$'
            )
                
    
        # --- plotting (your code) ---
        plt.figure(figsize=(3.5, 3), dpi=500)
        plt.scatter(ln_edp, ln_im, color="grey", s=10)
        plt.plot(regression, ln_im, color="black")
        plt.plot(edp_plusbeta, ln_im , color="red", linestyle="--")
        plt.plot(edp_minusbeta, ln_im, color="red", linestyle="--")
        plt.xlabel("Demand, $\ln EDP$")
        plt.ylabel("Intensity, $\ln IM$")

        
        # place text in axes coordinates
        plt.text(
            0.02, 0.98,
            textstr,
            fontsize=9,
            transform=plt.gca().transAxes,
            verticalalignment='top'#,
            #bbox=dict(
            #    facecolor='white',
            #    edgecolor='black',
            #    boxstyle='round,pad=0.3',
            #    alpha = 0.3
            #)
        )
        
        plt.tight_layout()
        save_plot(output_dir_i / f'demand_capacity_{building}.pdf')
        
        
        bspecif_dem_in[im_type][building] = {
        "slope": slope,
        "intercept": intercept,
        "beta": beta
        }
    
    
    slopes = [demand_capacity[b]["slope"] for b in demand_capacity]
    intercepts = [demand_capacity[b]["intercept"] for b in demand_capacity]
    betas = [demand_capacity[b]["beta"] for b in demand_capacity]
    
    
        
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)
    mean_beta = (np.mean(np.square(betas)))**0.5
    
    global_min_lnIM = np.min(ln_im)
    global_max_lnIM = np.max(ln_im)
    

    ln_im_grid = np.linspace(global_min_lnIM, global_max_lnIM, 100)
    
    
    #Estimate intra building variability
    
    beta_inter = []
    
    var_inter_list = []

    for building in building_names:
        slope = demand_capacity[building]["slope"]
        intercept = demand_capacity[building]["intercept"]
    
        regression = slope * ln_im_grid + intercept
        regression_tax = mean_slope * ln_im_grid + mean_intercept
    
        error = (regression - regression_tax)**2
        var_i = np.mean(error)
        var_inter_list.append(var_i)
    
    inter_beta = np.sqrt(np.mean(var_inter_list))
    total_beta = np.sqrt(inter_beta**2+mean_beta**2)
    
    tax_dem_in[im_type]={}
    tax_dem_in[im_type] = {
    "slope": mean_slope,
    "intercept": mean_intercept,
    "intra_beta": mean_beta,
    "inter_beta": inter_beta,
    "beta": total_beta
    }
    

    print("Mean slope:", mean_slope)
    print("Mean intercept:", mean_intercept)
    print("inter-beta:", mean_beta)  
    print("intra-beta:", inter_beta)
    print("total beta:", total_beta)
     
    
    # Common ln(edp) range for plotting
    ln_edp_range = np.linspace(min(ln_edp), max(ln_edp), 100)
    
    plt.figure(figsize=(7, 5))
    
    # Plot each building regression
    for building in demand_capacity:
        slope = demand_capacity[building]["slope"]
        intercept = demand_capacity[building]["intercept"]
        regression = slope * ln_edp_range + intercept
        plt.plot(ln_edp_range, regression, color='grey', alpha=0.5, linewidth=1)
    
    # Plot mean regression and ±beta bounds
    mean_regression = mean_slope * ln_edp_range + mean_intercept
    plt.plot(ln_edp_range, mean_regression, color='blue', linewidth=2.5, label='Mean regression')
    plt.plot(ln_edp_range, mean_regression + mean_beta, color='blue', linestyle='--', linewidth=1)
    plt.plot(ln_edp_range, mean_regression - mean_beta, color='blue', linestyle='--', linewidth=1)
    plt.plot(ln_edp_range, mean_regression + inter_beta, color='red', linestyle='--', linewidth=1)
    plt.plot(ln_edp_range, mean_regression - inter_beta, color='red', linestyle='--', linewidth=1)
    
    plt.xlabel("ln(edp)")
    plt.ylabel("ln(im)")
    plt.title("Demand-capacity regression models for all buildings")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.text(
        0.05, 0.95,
        f'Mean: ln(edp) = {mean_slope:.3f} ln(im) + {mean_intercept:.3f}\n'
        f'Mean β = {mean_beta:.3f}',
        fontsize=11,
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )
    plt.show
    


filename = directory / "demand_intensity.json"
with open(filename, 'w') as json_file:
    json.dump(tax_dem_in, json_file, indent=4)
 
filename = directory / "demand_intensity_bs.json"
with open(filename, 'w') as json_file:
    json.dump(bspecif_dem_in, json_file, indent=4)

"""  
#process combined results
filt_comb_demand=[]
filt_comb_capacity = []

for xi, yi in zip(combined_capacity, combined_demand):
    if xi <= collapse_drift:
        filt_comb_capacity.append(xi)
        filt_comb_demand.append(yi)

 #Estimate logartithm
ln_edp_comb=np.log(filt_comb_capacity)
ln_im_comb=np.log(filt_comb_demand)
 
 #Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(ln_im_comb, ln_edp_comb)
 
 #Calculate vertical dispersion
regression=slope*ln_im_comb+intercept
error=(regression-ln_edp_comb)**2
beta=((1/(len(error)-1))*np.sum(error))**0.5
 
 #Fill dictionary
demand_capacity_combined={}
demand_capacity_combined["slope"]=slope
demand_capacity_combined["intercept"]=intercept
demand_capacity_combined["beta"]=beta

# Plot results
edp_plusbeta = regression + beta
edp_minusbeta = regression - beta
plt.figure()
plt.scatter(ln_edp_comb, ln_im_comb, color="grey")
plt.plot(regression, ln_im_comb, color="black")
plt.plot(edp_plusbeta, ln_im_comb, color="red", linestyle="--")
plt.plot(edp_minusbeta, ln_im_comb, color="red", linestyle="--")
plt.xlabel("ln(edp)")
plt.ylabel("ln(im)")
plt.title(f'Demand capacity model taxonomy')
# Format variables and create text
plt.text(
    0.05, 0.95,
    f'ln(edp)={slope:.3f} ln(im) {intercept:.3f} \n'
    f'R={r_value:.3f} \n'
    f'beta={beta:.3f}',
    fontsize=12,
    transform=plt.gca().transAxes,  # Ensure text is positioned relative to plot area
    verticalalignment='top'
)
save_plot(output_dir / f'demand_capacity_combined.png')
filename = directory / "demand_capacity_tax.json"
with open(filename, 'w') as json_file:
    json.dump(demand_capacity_combined, json_file, indent=4)
    
"""