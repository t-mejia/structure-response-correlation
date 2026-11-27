# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:34:18 2025

@author: Tomas Mejia
"""

# from utils import *   (not required)
from pathlib import Path
import numpy as np
import os
import re
import eqsig
import pandas as pd
from scipy.stats import gmean

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


def extract_rsn_number(filename: str) -> int:
    """
    Extract the integer after 'RSN' in a filename.
    
    Args:
        filename (str): filename string (e.g., 'RSN1000_NORTHR_PIC180.txt')
    
    Returns:
        int: extracted RSN number
    """
    match = re.search(r'RSN(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No RSN number found in filename: {filename}")

def txt_to_list(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]
    return lines

def save_list_to_file(items, file_path):
    with open(file_path, 'w') as f:
        for item in items:
            f.write(f"{item}\n")

og_path=Path(__file__).parent
path=Path(__file__).parent/"records-tomas"
names1=txt_to_list(DATA/"cloud_analysis"/"names.txt")


# Output containers
above_threshold_dir1 = []
above_threshold_dir2 = []
above_threshold_dts = []

T = 0.665
T_ = [T]
Sa_avg = []
Sa_T = []
PGA = []
Sa_01 =[]
Sa_06 = []
Sa_1 = []

n=[]
for i in range(len(names1)):
    path1 = path / names1[i]
    gm=np.loadtxt(path1)
    ag = gm[:, 1]
    time = gm[:, 0]
    
    dt_i = time[1]-time[0]

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
    
    Sa_avg.append(sa_avg)
    Sa_T.append(sa_Ti)
    Sa_01.append(sa_01i)
    Sa_06.append(sa_06i)
    Sa_1.append(sa_1i)
    PGA.append(pga)

# Convert to numpy arrays for safer scaling
Sa_avg = np.array(Sa_avg)
Sa_T = np.array(Sa_T)
Sa_01 = np.array(Sa_01)
Sa_06 = np.array(Sa_06)
Sa_1 = np.array(Sa_1)
PGA = np.array(PGA)

# Append scaled copies
Sa_avg = np.concatenate([Sa_avg, 2*Sa_avg, 6*Sa_avg])
Sa_T  = np.concatenate([Sa_T,  2*Sa_T,  6*Sa_T])
Sa_01 = np.concatenate([Sa_01, 2*Sa_01, 6*Sa_01])
Sa_06 = np.concatenate([Sa_06, 2*Sa_06, 6*Sa_06])
Sa_1  = np.concatenate([Sa_1,  2*Sa_1,  6*Sa_1])
PGA   = np.concatenate([PGA,   2*PGA,   6*PGA])

df = pd.DataFrame({
    #"Record": record_ids,
    "PGA": PGA,
    "AvgSa2": Sa_avg,
    f"Sa(T={T:.3f}s)": Sa_T,
    "Sa(0.1s)": Sa_01,
    "Sa(0.6s)": Sa_06,
    "Sa(1.0s)": Sa_1
})

output_csv = os.path.join(og_path, "ground_motion_IMs.csv")
df.to_csv(output_csv, index=False)

print(f"✅ CSV saved at: {output_csv}")
print(df.head())