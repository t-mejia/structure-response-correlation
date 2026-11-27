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
from scipy.optimize import curve_fit

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

def regression_model(im, b0, b1):
    return 1/(1+np.exp(b0-b1*im))
    
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

tax_frag = {}
fragility={}
for im_type in im_types:
 
    #define demand-capscity dictionary
    fragility[im_type] = {}

    combined_demand = []
    combined_capacity = []
    for building in building_names:
        fragility[im_type][building]={}
        ims=intensity[im_type]
        max_edp=edps[building]
    
        #filter for collapse drift
        edp_filtered = []
        im_filtered = []
        
        
        for xi, yi in zip(max_edp, ims):
            if xi <= collapse_drift:
                edp_filtered.append(0)
                im_filtered.append(yi)
                
            else:
                edp_filtered.append(1)
                im_filtered.append(yi)
        
        #Estimate logartithm
        collapse=edp_filtered
        ln_im=np.log(im_filtered)
        
        #Linear regression
        # Initial guess for b0 and b1
        initial_guess = [0.0, 1.0]
        
        # Fit logistic regression
        params, covariance = curve_fit(regression_model, ln_im, collapse, p0=initial_guess)
        
        b0_fit, b1_fit = params
        
        median = b0_fit/b1_fit
        
        percent_16 = (b0_fit-np.log((1-0.16)/0.16))/b1_fit
        percent_84 = (b0_fit-np.log((1-0.84)/0.84))/b1_fit
        
        std = (percent_84 - percent_16)/2
        
        
        
        #Fill dictionary
        fragility[im_type][building]["mu"]=median
        fragility[im_type][building]["sigma"]=std

    
    mus = [fragility[im_type][b]["mu"] for b in fragility[im_type]]
    betas = [fragility[im_type][b]["sigma"] for b in fragility[im_type]]
    
    
        
    mean_mus = np.mean(mus)
    mean_beta = (np.mean(np.square(betas)))**0.5

        
    #Estimate inter building variability
    
    beta_inter = []
    
    for building in building_names:
        mu = fragility[im_type][building]["mu"]
        error=(mu-mean_mus)**2
        beta_inter.append(error)
    
    inter_beta=(np.sum(beta_inter)/len(beta_inter))**0.5
    total_beta = np.sqrt(inter_beta**2+mean_beta**2)
    
    tax_frag[im_type]={}
    tax_frag[im_type] = {
    "mu": mean_mus,
    "intra_beta": mean_beta,
    "inter_beta": inter_beta,
    "beta": total_beta
    }
    

    print("Mu:", mean_mus)
    print("intra-beta:", mean_beta)  
    print("inter-beta:", inter_beta)
    print("total beta:", total_beta)
     

    

    

filename = directory / f'fragility_{collapse_drift}.json'
with open(filename, 'w') as json_file:
    json.dump(tax_frag, json_file, indent=4)
 


filename = directory / f'fragility_{collapse_drift}_BuildingSpecific.json'
with open(filename, 'w') as json_file:
    json.dump(fragility, json_file, indent=4)
    