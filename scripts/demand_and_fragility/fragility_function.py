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
                      im_lo=1e-4, im_hi=50.0):
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
    
directory=Path(__file__).parent
edp_dir=DATA/"cloud_analysis"/"cloud_analysis_psd.json"
im_dir=DATA/"cloud_analysis"/"ground_motion_IMs.csv"
buildings_dir= DATA/"portfolio"/"buildings.txt"
output_dir =directory /"figs_tax_FF"
collapse_ff_dir=DATA/"models"/"fragility_5.0.json"



collapse_drift=5.0
damage_threshold = 2.0

# Create the output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)
with open(edp_dir, 'r') as file:
    edps= json.load(file)

#import IMS
intensity = pd.read_csv(im_dir, sep=",")

im_types = intensity.columns.tolist()


# Labels for each building
building_names = read_txt_to_list(buildings_dir)

tax_frag = {}

for im_type in im_types:
 
    #define demand-capscity dictionary
    demand_capacity={}
    fragility_function = {}
    combined_demand = []
    combined_capacity = []
    for building in building_names:
        demand_capacity[building]={}
        fragility_function[building]={}
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
        
        #collapse parameters

        edp_c_filtered = []
        im_c_filtered = []
        
        for xci, yci in zip(max_edp, ims):
            if xci <= collapse_drift:
                edp_c_filtered.append(0)
                im_c_filtered.append(yci)
                
            else:
                edp_c_filtered.append(1)
                im_c_filtered.append(yci)
        
        #Estimate logartithm
        collapse=edp_c_filtered
        ln_im_collapse=np.log(im_c_filtered)
        
        #Linear regression
        # Initial guess for b0 and b1
        initial_guess = [0.0, 1.0]
        
        # Fit logistic regression
        params, covariance = curve_fit(regression_model, ln_im_collapse, collapse, p0=initial_guess)
        
        b0_fit, b1_fit = params

        #median fragility function
        lnim_50 = solve_lnim_for_p(p=0.50, edp_k=damage_threshold, a=intercept, b=slope, beta_edp=beta, b0=b0_fit, b1=b1_fit, im_lo=1e-4, im_hi=10.0)
        #Find percentiles
        lnim_16= solve_lnim_for_p(p=0.16, edp_k=damage_threshold, a=intercept, b=slope, beta_edp=beta, b0=b0_fit, b1=b1_fit, im_lo=1e-4, im_hi=10.0)
        #Find percentiles
        lnim_84= solve_lnim_for_p(p=0.84, edp_k=damage_threshold, a=intercept, b=slope, beta_edp=beta, b0=b0_fit, b1=b1_fit, im_lo=1e-4, im_hi=10.0)
        
        #Find dispersion of FF
        
        beta_FF = (lnim_84-lnim_16)/2
        
        fragility_function[building]["mu"] = lnim_50
        fragility_function[building]["beta"] = beta_FF
        
        output_dir_i = output_dir / im_type
        output_dir_i.mkdir(parents=True, exist_ok=True)
        
        # Plot results
        #edp_plusbeta = regression + beta
        #edp_minusbeta = regression - beta
        plt.figure()
        plt.scatter(ln_im, ln_edp, color="grey")
        plt.plot(regression, ln_edp, color="black")
        #plt.plot(edp_plusbeta, ln_im , color="red", linestyle="--")
        #plt.plot(edp_minusbeta, ln_im, color="red", linestyle="--")
        plt.xlabel("ln(im)")
        plt.ylabel("ln(edp)")
        plt.title(f'Demand capacity model {building}')
        # Format variables and create text
        plt.text(
            0.05, 0.95,
            f'ln(im)={slope:.3f} ln(edp) + {intercept:.3f} \n'
            f'R={r_value:.3f} \n'
            f'beta={beta:.3f}',
            fontsize=12,
            transform=plt.gca().transAxes,  # Ensure text is positioned relative to plot area
            verticalalignment='top'
        )
        save_plot(output_dir_i / f'demand_capacity_{building}.png')
    
    
    slopes = [demand_capacity[b]["slope"] for b in demand_capacity]
    intercepts = [demand_capacity[b]["intercept"] for b in demand_capacity]
    betas = [demand_capacity[b]["beta"] for b in demand_capacity]
    mu = [fragility_function[b]["mu"] for b in fragility_function]
           
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)
    mean_beta = np.mean(betas)
    mean_mu = np.mean(mu)

        
    #Estimate intra building variability
    
    beta_intra = []
    
    error = 0
    
    for building in building_names:
        median = fragility_function[building]["mu"]
        #Calculate vertical dispersion
        error += (median -  mean_mu)**2

    inter_beta= np.sqrt(error/len(building_names)) 
    total_beta = np.sqrt(inter_beta**2+mean_beta**2)
    
    tax_frag[im_type]={}
    tax_frag[im_type] = {
    "mu": mean_mu,
    "intra_beta": mean_beta,
    "inter_beta": inter_beta,
    "beta": total_beta
    }
    

    print("Mean slope:", mean_mu)
    print("Mean intercept:", mean_intercept)
    print("intra-beta:", mean_beta)  
    print("inter-beta:", inter_beta)
    print("total beta:", total_beta)
     

    

filename = directory/f'fragility_{damage_threshold}.json'
with open(filename, 'w') as json_file:
    json.dump(tax_frag, json_file, indent=4)
 


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
filename = "demand_capacity_tax.json"
with open(filename, 'w') as json_file:
    json.dump(demand_capacity_combined, json_file, indent=4)
    
"""