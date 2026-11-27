#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy of validation_Comparison-Geomean-multiDS-DSallpanel-Uncertainty.py in
which the DS-based correlated case is no longer a single Model H curve: any of
the DS-based models

    H   : (1 - tau2) * exp(-(d / c2)^c1) + tau2 * delta_ij
    HT  : exp(-(d / c2)^c1) * exp(-|T_i - T_j| / c4)                      (no nugget)
    HTS : exp(-(d / c2)^c1)
          * [c3 * exp(-|T_i - T_j| / c4) + (1 - c3) * exp(-|Say_i - Say_j| / c5)]  (no nugget)

can be shown, side by side, in each damage-state panel and in the loss panel
(PANEL_CURVES below), next to the EDP-based models (Independent, HT, HTS, TS).

Each DS-based model is simulated as its own full DS1 -> DS4 chain (the
sequential conditional sampling needs a correlation matrix at every damage
state, and so do the losses), with the parameters of that model at each damage
state taken from DS_PARAMS. Parameters still marked dummy=True are placeholders
until the model is fitted for that damage state; they print a warning.

Colors follow the model, not the approach: the same model (e.g. HT) has the
same color in every panel, solid for DS-based and dotted for EDP-based.

The nested Monte Carlo (outer building-assignment draws, inner N-iteration
simulation, curves averaged across draws, real-scenario mean +/- 1, 2 sigma
bands) is the same as in the original script.

Direct-sampling check (this copy of ...-Uncertainty-HT-HTS.py): in every DS
panel above DS1, each DS-based correlated model is also simulated by direct
sampling, without the sequential conditional approach,

    X ~ MVN(0, rho_k),   building i >= DS_k  iff  Phi(X_i) < p_k,

with rho_k that model's DS_k correlation used as is, and drawn faintly behind
the sequential curve (same color, alpha DIRECT_ALPHA) to show the impact of
the methodology. DS1 is left out (the first stage of the sequential sampling
is already a direct draw), and so are the losses (they need one joint DS
assignment across all levels, which per-level direct draws do not give).
"""

import time
import pandas as pd
from pathlib import Path
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def cholesky_with_jitter(K, initial_jitter=1e-12, max_tries=10):
    K = (K + K.T) / 2  # enforce symmetry
    jitter = initial_jitter
    for _ in range(max_tries):
        try:
            L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter *= 10
    raise np.linalg.LinAlgError("Cholesky failed even after adding jitter.")

def nearest_correlation_matrix(A, tol=1e-8):
    A = (A + A.T) / 2  # enforce symmetry
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals[eigvals < tol] = tol
    # macOS Accelerate's BLAS matmul raises spurious divide/overflow/invalid
    # RuntimeWarnings on some inputs even though the result has no NaN/Inf;
    # verified separately, so they are suppressed here rather than masking
    # an assertion that would catch a genuine problem.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        B = eigvecs @ np.diag(eigvals) @ eigvecs.T
    D = np.sqrt(np.diag(B))
    C = B / np.outer(D, D)
    assert np.isfinite(C).all(), "nearest_correlation_matrix produced non-finite entries"
    return C

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
_GL_NODES = 0.5 * (_GL_NODES + 1.0)      # rescale [-1,1] -> [0,1]
_GL_WEIGHTS = 0.5 * _GL_WEIGHTS

def bivariate_normal_cdf(h, k, rho):
    """Phi_2(h,k;rho) via Plackett's identity d(Phi_2)/d(rho) = phi_2(h,k;rho),
    integrated from 0 to rho with fixed-node Gauss-Legendre quadrature.
    h, k, rho broadcast together (e.g. all same-shape matrices)."""
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    rho = np.clip(np.asarray(rho, dtype=float), -1 + 1e-12, 1 - 1e-12)

    t = rho[..., None] * _GL_NODES  # (..., Q)
    h_ = h[..., None]
    k_ = k[..., None]
    denom = 1.0 - t ** 2
    density = np.exp(-(h_ ** 2 + k_ ** 2 - 2 * t * h_ * k_) / (2 * denom)) / (2 * np.pi * np.sqrt(denom))
    integral = rho[..., None] * density  # dt = rho * ds, ds weight applied below via _GL_WEIGHTS
    correction = np.sum(integral * _GL_WEIGHTS, axis=-1)

    return norm.cdf(h) * norm.cdf(k) + correction

def solve_stage_correlation(x_a, x_b, target, feasible_bound, lo=-1 + 1e-9, hi=1 - 1e-9, n_iter=50):
    """Vectorized bisection solving Phi_2(x_a,x_b;rho)=target for rho, elementwise.
    Where target exceeds feasible_bound, rho is set to 1 instead (Eq. stage_feasibility)."""
    lo_arr = np.full_like(target, lo)
    hi_arr = np.full_like(target, hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo_arr + hi_arr)
        val = bivariate_normal_cdf(x_a, x_b, mid)
        below = val < target
        lo_arr = np.where(below, mid, lo_arr)
        hi_arr = np.where(below, hi_arr, mid)
    rho = 0.5 * (lo_arr + hi_arr)
    infeasible = target > feasible_bound
    return np.where(infeasible, 1.0, rho)

def pairwise_haversine(lat, lon, R=6371.0):
    """Matrix of great-circle distances (km) between all sites; same formula
    as the scalar haversine of the original script."""
    phi = np.radians(lat)
    lam = np.radians(lon)
    dphi = phi[None, :] - phi[:, None]
    dlam = lam[None, :] - lam[:, None]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2) ** 2
    d = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 0, None)))
    np.fill_diagonal(d, 0.0)
    return d


N = 10000                # inner Monte Carlo iterations per scenario draw (Part I: 10,000)
N_OUTER_DRAWS = 100     # outer scenario (building-assignment) draws
BASE_SEED = 1000         # rng seed for draw 0; draw d uses BASE_SEED + d (scenario and inner simulation)


IM = "AvgSa2"
n_bst = 1  # number of buildings randomly assigned per site

replacement_cost = 252612.0  # Obtained from GEM 2026 Exposure model for CR/LFM/CDM+ERM/H:4/RES

#Damage Threholds
damage_threshold_1 = 0.5
damage_threshold_2 = 1.0
damage_threshold_3 = 2.0
collapse_threshold = 5.0

#Loss ratios
LR_1 = 0.05
LR_2 = 0.15
LR_3 = 0.60
LR_collapse = 1.0
LR_map = np.array([0.0, LR_1, LR_2, LR_3, LR_collapse])


# ===========================================================================
# WHICH CURVES TO SHOW
# ===========================================================================
# Curve ids: "DS:Independent", "DS:HT", "DS:HTS",
#            "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"
# Per damage-state panel (1-4) and for the loss panel ("loss"). A DS-based model
# listed anywhere is simulated over the full DS1 -> DS4 chain, so it needs
# parameters (fitted or dummy) at every damage state in DS_PARAMS.
PANEL_CURVES = {
    1:      ["DS:Independent", "DS:HT", "DS:HTS", "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"],
    2:      ["DS:Independent", "DS:HT", "DS:HTS", "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"],
    3:      ["DS:Independent", "DS:HT", "DS:HTS", "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"],
    4:      ["DS:Independent", "DS:HT", "DS:HTS", "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"],
    "loss": ["DS:Independent", "DS:HT", "DS:HTS", "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"],
}


# ===========================================================================
# DS-BASED MODEL PARAMETERS (per damage state)
# ===========================================================================
# dummy=True marks placeholders until the model is fitted for that damage
# state: replace the values and delete dummy=True.
#   H   : c2 (km), c1, tau2
#   HT  : c2 (km), c1, c4 (s)
#   HTS : c2 (km), c1, c4 (s), c5 (g), c3
# Dummy defaults are the prior means (c2 15 km, c1 1, c4 0.179 s,
# c5 0.171 g, c3 0.75).


DS_PARAMS = {   # Part I Table 4 = Part II Table 6 (posterior means); DS4 uses the DS1 model (no DS4 fit)
    1: {"HT":  dict(c2=26.99, c1=0.151, c4=0.107, dummy=False),
        "HTS": dict(c2=30.07, c1=0.192, c4=0.157, c5=0.158, c3=0.691, dummy=False)},
    2: {"HT":  dict(c2=19.39, c1=0.619, c4=0.114, dummy=False),
        "HTS": dict(c2=23.72, c1=0.585, c4=0.156, c5=0.148, c3=0.687, dummy=False)},
    3: {"HT":  dict(c2=13.07, c1=0.673, c4=0.153, dummy=False),
        "HTS": dict(c2=19.88, c1=0.846, c4=0.200, c5=0.151, c3=0.654, dummy=False)},
    4: {"HT":  dict(c2=26.99, c1=0.151, c4=0.107, dummy=False),
        "HTS": dict(c2=30.07, c1=0.192, c4=0.157, c5=0.158, c3=0.691, dummy=False)},
}
# The EDP-based simulations draw collapse (DS4) from the DS-based DS4
# correlation of this model (the original script used Model H DS4).
EDP_COLLAPSE_DS_MODEL = "HT"


# ===========================================================================
# EDP-BASED MODEL PARAMETERS
# ===========================================================================
EDP_PARAMS = {   # Part I Table 4 = Part II Table 5 (posterior means); TS from the deterministic fit
    "HT":  dict(c2=41.87, c1=0.149, c4=0.127),
    "HTS": dict(c2=58.48, c1=0.252, c3=0.704, c4=0.159, c5=0.134),
    "TS":  dict(c3=0.855, c4=0.179, c5=0.171),
}


# ===========================================================================
# STYLE: one color per model, solid = DS-based, dotted = EDP-based
# ===========================================================================
MODEL_COLORS = {"Independent": "k", "H": "#B0527A", "HT": "#146c14", "HTS": "#ff7f0e", "TS": "#366693"}
# The EDP-based curves are all dotted with the same (1, 1) dot/gap period;
# staggering each model's offset by a quarter of the period keeps nearby
# curves' dots interleaved instead of coincident.
EDP_DOT_OFFSET = {"Independent": 0.0, "HT": 0.5, "HTS": 1.0, "TS": 1.5}
CURVE_ORDER = ["DS:Independent", "DS:HT", "DS:HTS",
               "EDP:Independent", "EDP:HT", "EDP:HTS", "EDP:TS"]

# Direct sampling (no sequential conditioning) of the DS-based correlated models:
# True = simulate it for DS2-DS4 and draw it behind the sequential curves (same color, alpha
# below); False = skip it (no extra run time). Never done for DS1, where it coincides with the
# sequential sampling (its first stage is already a direct draw).
DIRECT_SAMPLING = True
DIRECT_LEVELS = (2, 3, 4) if DIRECT_SAMPLING else ()
DIRECT_ALPHA = 0.35
DIRECT_LW = 1.0      # line width of the direct-sampling curves (sequential curves: 1.5)
DIRECT_CHUNK = 20000   # iterations per matrix product in the direct sampling (memory)

# x-axis LIMIT of each panel (the axes stay in # buildings and Million USD):
#   "data"     : around the real-scenario counts, or where the curves vanish (DS panels);
#                20 x replacement cost (loss panel) -- the original rule
#   "expected" : X_LIMIT_FACTOR x the expected value under independence (sum_i P(DS >= k), and the
#                expected loss), averaged over the allocations
#   "ref"      : where the curve XLIM_REF_CURVE falls to XLIM_REF_PROB (falls back to "data" if that
#                curve is not computed)
XLIM_MODE      = {1: "ref", 2: "ref", 3: "ref", 4: "data", "loss": "ref"}
XLIM_REF_CURVE = "EDP:TS"
XLIM_REF_PROB  = 1e-3
X_LIMIT_FACTOR = {1: 3.0,    2: 3.0,    3: 4.0,        4: 3.0,        "loss": 3.0}
X_LIMIT_MIN = 5        # "expected" mode, DS panels: at least this many buildings (DS4 expects < 1)


def curve_style(curve_id):
    approach, model = curve_id.split(":")
    linestyle = "-" if approach == "DS" else (EDP_DOT_OFFSET[model], (1, 1))
    label = f"{approach}-based: " + ("Independent" if model == "Independent" else f"Model {model}")
    return dict(label=label, linestyle=linestyle, color=MODEL_COLORS[model])


# --- validate the selection and warn about dummy parameters ---
_panels = [1, 2, 3, 4, "loss"]
for _p in _panels:
    for _cid in PANEL_CURVES[_p]:
        if _cid not in CURVE_ORDER:
            raise ValueError(f"panel {_p}: unknown curve {_cid!r}; choose from {CURVE_ORDER}")
SHOWN = [c for c in CURVE_ORDER if any(c in PANEL_CURVES[p] for p in _panels)]
DS_CHAIN_MODELS = [c.split(":")[1] for c in SHOWN if c.startswith("DS:") and c != "DS:Independent"]
EDP_CORR_MODELS = [c.split(":")[1] for c in SHOWN if c.startswith("EDP:") and c != "EDP:Independent"]

for _m in DS_CHAIN_MODELS + [EDP_COLLAPSE_DS_MODEL]:
    for _lvl in (1, 2, 3, 4):
        if DS_PARAMS[_lvl].get(_m) is None:
            raise ValueError(f"DS-based Model {_m} is needed at DS{_lvl} but DS_PARAMS[{_lvl}]['{_m}'] is missing")
_dummies = sorted({(_lvl, _m) for _m in DS_CHAIN_MODELS + [EDP_COLLAPSE_DS_MODEL]
                   for _lvl in (1, 2, 3, 4) if DS_PARAMS[_lvl][_m].get("dummy", False)})
for _lvl, _m in _dummies:
    print(f"WARNING: DS-based Model {_m} at DS{_lvl} uses DUMMY parameters {DS_PARAMS[_lvl][_m]}", flush=True)


def ds_correlation_matrix(model, level, H, dT, dS):
    """DS-based correlation matrix at damage state `level` for `model`, given
    the site distance (H), period-difference (dT) and Say-difference (dS) matrices."""
    prm = DS_PARAMS[level][model]
    if model == "H":
        cs = np.exp(-1.0 * (H / prm["c2"]) ** prm["c1"])
        return (1.0 - prm["tau2"]) * cs + prm["tau2"] * np.eye(len(H))
    spatial = np.exp(-1.0 * (H / prm["c2"]) ** prm["c1"])
    if model == "HT":
        return spatial * np.exp(-dT / prm["c4"])
    return spatial * (prm["c3"] * np.exp(-dT / prm["c4"]) + (1.0 - prm["c3"]) * np.exp(-dS / prm["c5"]))


def edp_correlation_matrix(model, H, dT, dS):
    prm = EDP_PARAMS[model]
    if model == "HT":
        return np.exp(-1.0 * (H / prm["c2"]) ** prm["c1"]) * np.exp(-dT / prm["c4"])
    if model == "HTS":
        return np.exp(-1.0 * (H / prm["c2"]) ** prm["c1"]) * (
            prm["c3"] * np.exp(-dT / prm["c4"]) + (1 - prm["c3"]) * np.exp(-dS / prm["c5"]))
    return prm["c3"] * np.exp(-dT / prm["c4"]) + (1 - prm["c3"]) * np.exp(-dS / prm["c5"])   # TS


# Load CSV
df = pd.read_csv(DATA/"events"/"northridge"/"epsilon-2D"/"AvgSa2"/"eps_PSD-geomean.csv")

with open(DATA/"portfolio"/"bilinear_results.json", "r") as f:
    bilinear = json.load(f)

with open(DATA/"portfolio"/"modal_results.json", "r") as f:
    modal = json.load(f)

with open(DATA/"events"/"northridge"/"northridge_results-2D.json", "r") as f:
    nlth = json.load(f)

with open(DATA/"models"/"downstream"/f'fragility_{damage_threshold_1}.json', "r") as f:
    frag1 = json.load(f)

with open(DATA/"models"/"downstream"/f'fragility_{damage_threshold_2}.json', "r") as f:
    frag2 = json.load(f)

with open(DATA/"models"/"downstream"/f'fragility_{damage_threshold_3}.json', "r") as f:
    frag3 = json.load(f)

with open(DATA/"models"/"downstream"/f'fragility_{collapse_threshold}.json', "r") as f:
    frag4 = json.load(f)

with open(DATA/"models"/"downstream"/"demand_intensity.json", "r") as f:
    d_i = json.load(f)

mu1, beta1 = frag1[IM]["mu"], frag1[IM]["beta"]
mu2, beta2 = frag2[IM]["mu"], frag2[IM]["beta"]
mu3, beta3 = frag3[IM]["mu"], frag3[IM]["beta"]
mu4, beta4 = frag4[IM]["mu"], frag4[IM]["beta"]

intercept = d_i[IM]["intercept"]
slope = d_i[IM]["slope"]
beta = d_i[IM]["beta"]

RSNs = [str(int(line['RSN'])) for _, line in df.iterrows()]
LATS = df['latitude'].to_numpy()
LONS = df['longitude'].to_numpy()


def build_scenario(rng):
    """Draw one building assignment (one per site) and return the resulting
    scenario DataFrame. `rng` is a random.Random instance so each outer draw
    can use its own independent, reproducible stream."""
    rows = {"RSN": [], "latitude": [], "longitude": [], "IM": [],
            "PSD_real": [], "Period": [], "Say": []}
    for RSN, lat, lon in zip(RSNs, LATS, LONS):
        for _ in range(n_bst):
            building_number = rng.randint(1, 80)
            building = "Building_{}".format(building_number)
            IM_1 = nlth[building][RSN]['dir_1']["IM_data"][IM]
            IM_2 = nlth[building][RSN]['dir_2']["IM_data"][IM]
            PSD_1 = nlth[building][RSN]['dir_1']["max_psd"]
            PSD_2 = nlth[building][RSN]['dir_2']["max_psd"]
            rows["RSN"].append(RSN)
            rows["latitude"].append(lat)
            rows["longitude"].append(lon)
            rows["IM"].append((IM_1 * IM_2) ** 0.5)
            rows["PSD_real"].append((PSD_1 * PSD_2) ** 0.5)
            rows["Say"].append(bilinear[building]["X"]["force"][1])
            rows["Period"].append(modal[building]["MX"]["period"])
    return pd.DataFrame(rows)


def edp_ds(edp):
    ds = np.zeros(edp.shape, dtype=int)
    ds[edp > damage_threshold_1] = 1
    ds[edp > damage_threshold_2] = 2
    ds[edp > damage_threshold_3] = 3
    ds[edp > collapse_threshold] = 4
    return ds

def edp_full_ds(edp, collapsed):
    return np.where(collapsed, 4, edp_ds(edp))

def edp_losses(edp, collapsed):
    return replacement_cost * LR_map[edp_full_ds(edp, collapsed)].sum(axis=0)

def edp_damaged_DS(edp, collapsed, level):
    return np.sum(edp_full_ds(edp, collapsed) >= level, axis=0)


_EPS = 1e-10
def _clip01(p): return np.clip(p, _EPS, 1 - _EPS)


def stage_sigmas(delta_levels, p_levels):
    """Latent correlation matrices Sigma_X[1..4] of the sequential (stage-wise)
    DS sampling, from the DS-level correlations delta_levels[1..4]."""
    Sigma_X = [None] * 5
    Sigma_X[1] = nearest_correlation_matrix(delta_levels[1])
    for k in (2, 3, 4):
        p_k, p_km1 = p_levels[k], p_levels[k - 1]
        delta_k, delta_km1 = delta_levels[k], delta_levels[k - 1]
        ppf_pk = norm.ppf(_clip01(p_k))
        ppf_pkm1 = norm.ppf(_clip01(p_km1))
        J_k = bivariate_normal_cdf(ppf_pk[:, None], ppf_pk[None, :], delta_k)
        J_km1 = bivariate_normal_cdf(ppf_pkm1[:, None], ppf_pkm1[None, :], delta_km1)
        target_ratio = J_k / np.maximum(J_km1, _EPS)
        ratio_a = _clip01(p_k / p_km1)
        x_a = norm.ppf(ratio_a)
        feasible_bound = np.minimum(ratio_a[:, None], ratio_a[None, :])
        tilde_delta = solve_stage_correlation(x_a[:, None], x_a[None, :], target_ratio, feasible_bound)
        np.fill_diagonal(tilde_delta, 1.0)
        Sigma_X[k] = nearest_correlation_matrix(tilde_delta)
    return Sigma_X


def simulate_ds_chain(Sigma_X, p_levels, n):
    """Correlated DS-based case: sequential conditional sampling over DS1..DS4.
    Returns the final damage state of every site in every iteration, (n, N)."""
    L1_full, _ = cholesky_with_jitter(Sigma_X[1])
    final_DS = np.zeros((n, N), dtype=int)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for it in range(N):
            current_DS = np.zeros(n, dtype=int)
            active_idx = np.arange(n)
            for k in (1, 2, 3, 4):
                if len(active_idx) == 0:
                    break
                if k == 1:
                    L_active = L1_full
                else:
                    Sigma_active = Sigma_X[k][np.ix_(active_idx, active_idx)]
                    L_active, _ = cholesky_with_jitter(Sigma_active)
                Xn = norm.ppf(np.random.rand(len(active_idx)))
                Xc = L_active @ Xn
                U = norm.cdf(Xc)
                ratio_k = p_levels[k][active_idx] / p_levels[k - 1][active_idx]
                advance = U <= ratio_k
                current_DS[active_idx[advance]] = k
                active_idx = active_idx[advance]
            final_DS[:, it] = current_DS
    return final_DS


def simulate_ds_direct(delta_k, p_k, n):
    """Direct sampling at one damage state, without the sequential conditional
    approach: X ~ MVN(0, delta_k), building i >= DS_k iff Phi(X_i) < p_k.
    Returns the number of buildings >= DS_k in every iteration, (N,)."""
    L, _ = cholesky_with_jitter(nearest_correlation_matrix(delta_k))
    counts = np.zeros(N, dtype=int)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for start in range(0, N, DIRECT_CHUNK):
            m = min(DIRECT_CHUNK, N - start)
            # own random stream (set per outer draw), so the sequential curves
            # are identical to those of ...-Uncertainty-HT-HTS.py
            Xc = L @ direct_rng.standard_normal((n, m))
            counts[start:start + m] = np.sum(norm.cdf(Xc) < p_k[:, None], axis=0)
    return counts


def run_pipeline(scenario, d_values, loss_values):
    """Everything that's conditional on one scenario draw: fragility
    probabilities, correlation matrices, and the full N-iteration Monte Carlo
    simulation for every curve in SHOWN. Curves are evaluated on the shared
    `d_values`/`loss_values` grids (fixed across draws) so they can be
    averaged directly across outer draws.
    Returns real values for this draw plus per-draw curves
    {level: {curve_id: curve}} and {curve_id: loss curve}."""
    n = len(scenario)
    lat = scenario["latitude"].to_numpy()
    lon = scenario["longitude"].to_numpy()
    T = scenario["Period"].to_numpy()
    Say = scenario["Say"].to_numpy()
    IM_vals = scenario["IM"].to_numpy()
    PSD_real = scenario["PSD_real"].to_numpy()

    im = np.log(IM_vals)
    p1 = norm.cdf(im, loc=mu1, scale=beta1)
    p2 = norm.cdf(im, loc=mu2, scale=beta2)
    p3 = norm.cdf(im, loc=mu3, scale=beta3)
    p4 = norm.cdf(im, loc=mu4, scale=beta4)
    p_levels = [np.ones(n), p1, p2, p3, p4]

    # "Real"/observed values for this draw
    real_D1 = np.sum((PSD_real > damage_threshold_1) & (PSD_real <= damage_threshold_2))
    real_D2 = np.sum((PSD_real > damage_threshold_2) & (PSD_real <= damage_threshold_3))
    real_D3 = np.sum((PSD_real > damage_threshold_3) & (PSD_real <= collapse_threshold))
    real_D4 = np.sum(PSD_real > collapse_threshold)
    real_loss = replacement_cost * (real_D1*LR_1 + real_D2*LR_2 + real_D3*LR_3 + real_D4*LR_collapse)
    real_DS_count = {
        1: real_D1 + real_D2 + real_D3 + real_D4,
        2: real_D2 + real_D3 + real_D4,
        3: real_D3 + real_D4,
        4: real_D4,
    }

    # Distance, period and Say differences between all sites
    H = pairwise_haversine(lat, lon)
    dT = np.abs(T[:, None] - T[None, :])
    dS = np.abs(Say[:, None] - Say[None, :])

    damaged = {}   # curve_id -> {level: (N,) count of buildings >= level}
    losses = {}    # curve_id -> (N,) total loss

    # ---- independent DS-based case ----
    results_ind_DS0 = np.zeros((n, N)); results_ind_DS1 = np.zeros((n, N))
    results_ind_DS2 = np.zeros((n, N)); results_ind_DS3 = np.zeros((n, N))
    for it in range(N):
        U_ind = norm.cdf(norm.ppf(np.random.rand(n)))
        results_ind_DS0[:, it] = (U_ind < p1).astype(int)
        results_ind_DS1[:, it] = (U_ind < p2).astype(int)
        results_ind_DS2[:, it] = (U_ind < p3).astype(int)
        results_ind_DS3[:, it] = (U_ind < p4).astype(int)

    exclusive_DS1 = results_ind_DS0 - results_ind_DS1
    exclusive_DS2 = results_ind_DS1 - results_ind_DS2
    exclusive_DS3 = results_ind_DS2 - results_ind_DS3
    exclusive_collapse = results_ind_DS3

    losses["DS:Independent"] = replacement_cost * (
        LR_1*np.sum(exclusive_DS1, axis=0) + LR_2*np.sum(exclusive_DS2, axis=0)
        + LR_3*np.sum(exclusive_DS3, axis=0) + LR_collapse*np.sum(exclusive_collapse, axis=0)
    )
    damaged["DS:Independent"] = {
        1: np.sum(results_ind_DS0, axis=0), 2: np.sum(results_ind_DS1, axis=0),
        3: np.sum(results_ind_DS2, axis=0), 4: np.sum(results_ind_DS3, axis=0),
    }

    # ---- correlated DS-based cases: one full DS1 -> DS4 chain per model ----
    damaged_direct = {}   # model -> {level: (N,) count of buildings >= level}, direct sampling
    for model in DS_CHAIN_MODELS:
        delta_levels = [None] + [ds_correlation_matrix(model, k, H, dT, dS) for k in (1, 2, 3, 4)]
        final_DS = simulate_ds_chain(stage_sigmas(delta_levels, p_levels), p_levels, n)
        losses[f"DS:{model}"] = replacement_cost * LR_map[final_DS].sum(axis=0)
        damaged[f"DS:{model}"] = {level: np.sum(final_DS >= level, axis=0) for level in (1, 2, 3, 4)}
        damaged_direct[model] = {k: simulate_ds_direct(delta_levels[k], p_levels[k], n) for k in DIRECT_LEVELS}

    # ---- EDP-based cases ----
    L_edp = {model: cholesky_with_jitter(edp_correlation_matrix(model, H, dT, dS))[0] for model in EDP_CORR_MODELS}
    Sigma_collapse = nearest_correlation_matrix(ds_correlation_matrix(EDP_COLLAPSE_DS_MODEL, 4, H, dT, dS))
    L_collapse, _ = cholesky_with_jitter(Sigma_collapse)

    mean_ln_edp = intercept + slope * im

    results_ind_edp = np.zeros((n, N))
    results_corr_edp = {model: np.zeros((n, N)) for model in EDP_CORR_MODELS}
    collapse_ind = np.zeros((n, N), dtype=bool); collapse_corr = np.zeros((n, N), dtype=bool)

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for it in range(N):
            eps_ind = norm.ppf(np.random.rand(n)) * beta
            results_ind_edp[:, it] = np.exp(mean_ln_edp + eps_ind)
            # same standard-normal draw for every EDP model (common random numbers)
            for model in EDP_CORR_MODELS:
                results_corr_edp[model][:, it] = np.exp(mean_ln_edp + L_edp[model] @ eps_ind)

            Xc = norm.ppf(np.random.rand(n))
            collapse_ind[:, it] = norm.cdf(Xc) < p4
            collapse_corr[:, it] = norm.cdf(L_collapse @ Xc) < p4

    losses["EDP:Independent"] = edp_losses(results_ind_edp, collapse_ind)
    damaged["EDP:Independent"] = {level: edp_damaged_DS(results_ind_edp, collapse_ind, level) for level in (1, 2, 3, 4)}
    for model in EDP_CORR_MODELS:
        losses[f"EDP:{model}"] = edp_losses(results_corr_edp[model], collapse_corr)
        damaged[f"EDP:{model}"] = {level: edp_damaged_DS(results_corr_edp[model], collapse_corr, level)
                                   for level in (1, 2, 3, 4)}

    # ---- curves on the shared grids (so different draws can be averaged) ----
    curves_by_level = {level: {cid: np.array([np.mean(damaged[cid][level] > d) for d in d_values])
                               for cid in SHOWN}
                       for level in (1, 2, 3, 4)}
    loss_curves = {cid: np.array([np.mean(losses[cid] > l) for l in loss_values]) for cid in SHOWN}
    direct_curves_by_level = {level: {model: np.array([np.mean(damaged_direct[model][level] > d) for d in d_values])
                                      for model in DS_CHAIN_MODELS}
                              for level in DIRECT_LEVELS}

    # expected values under independence for this allocation (used by XLIM_MODE = "expected")
    expected_DS_count = {k: float(p_levels[k].sum()) for k in (1, 2, 3, 4)}
    expected_loss = float(replacement_cost * np.sum(LR_1 * (p1 - p2) + LR_2 * (p2 - p3)
                                                    + LR_3 * (p3 - p4) + LR_collapse * p4))

    return dict(real_loss=real_loss, real_DS_count=real_DS_count,
                expected_DS_count=expected_DS_count, expected_loss=expected_loss,
                curves_by_level=curves_by_level, loss_curves=loss_curves,
                direct_curves_by_level=direct_curves_by_level)


# ------------------------------------------------------------------
# Fixed shared grids, computed once so every outer draw's curves line up
# and can be averaged elementwise into a mixture distribution.
# ------------------------------------------------------------------
scenario0 = build_scenario(random.Random(BASE_SEED))
n_total = len(scenario0)
d_values = np.arange(0, n_total + 1)
up_xlim = 20 * replacement_cost
loss_values = np.linspace(0, 3 * up_xlim, 600)   # same resolution as before, wider so any x-limit fits

print(f"Curves: {SHOWN}")
print(f"DS-based chains simulated: {DS_CHAIN_MODELS}; EDP collapse correlation: DS4 Model {EDP_COLLAPSE_DS_MODEL}")

# ------------------------------------------------------------------
# Nested Monte Carlo: redraw the scenario N_OUTER_DRAWS times, rerunning
# the full correlation-matrix + N-iteration simulation pipeline each time.
# ------------------------------------------------------------------
real_loss_draws = np.zeros(N_OUTER_DRAWS)
expected_loss_draws = np.zeros(N_OUTER_DRAWS)
expected_DS_count_draws = {level: np.zeros(N_OUTER_DRAWS) for level in (1, 2, 3, 4)}
real_DS_count_draws = {level: np.zeros(N_OUTER_DRAWS) for level in (1, 2, 3, 4)}
curves_by_level_draws = {level: {cid: [] for cid in SHOWN} for level in (1, 2, 3, 4)}
loss_curves_draws = {cid: [] for cid in SHOWN}
direct_curves_draws = {level: {model: [] for model in DS_CHAIN_MODELS} for level in DIRECT_LEVELS}

t_start = time.time()
for draw in range(N_OUTER_DRAWS):
    np.random.seed(BASE_SEED + draw)   # inner simulation reproducible per draw
    direct_rng = np.random.default_rng([BASE_SEED, draw])   # direct-sampling stream
    scenario_d = scenario0 if draw == 0 else build_scenario(random.Random(BASE_SEED + draw))
    result = run_pipeline(scenario_d, d_values, loss_values)

    real_loss_draws[draw] = result["real_loss"]
    expected_loss_draws[draw] = result["expected_loss"]
    for level in (1, 2, 3, 4):
        expected_DS_count_draws[level][draw] = result["expected_DS_count"][level]
    for level in (1, 2, 3, 4):
        real_DS_count_draws[level][draw] = result["real_DS_count"][level]
        for cid in SHOWN:
            curves_by_level_draws[level][cid].append(result["curves_by_level"][level][cid])
    for cid in SHOWN:
        loss_curves_draws[cid].append(result["loss_curves"][cid])
    for level in DIRECT_LEVELS:
        for model in DS_CHAIN_MODELS:
            direct_curves_draws[level][model].append(result["direct_curves_by_level"][level][model])

    print(f'draw {draw+1}/{N_OUTER_DRAWS} done ({time.time()-t_start:.0f}s elapsed, '
          f'real_loss={result["real_loss"]:,.0f})', flush=True)

curves_by_level_avg = {level: {cid: np.mean(np.stack(curves_by_level_draws[level][cid]), axis=0) for cid in SHOWN}
                       for level in (1, 2, 3, 4)}
loss_curves_avg = {cid: np.mean(np.stack(loss_curves_draws[cid]), axis=0) for cid in SHOWN}
direct_curves_avg = {level: {model: np.mean(np.stack(direct_curves_draws[level][model]), axis=0)
                             for model in DS_CHAIN_MODELS}
                     for level in DIRECT_LEVELS}

print(f'Real loss over {N_OUTER_DRAWS} nested draws: '
      f'mean={np.mean(real_loss_draws):,.0f}  std={np.std(real_loss_draws):,.0f}  '
      f'[{np.min(real_loss_draws):,.0f}, {np.max(real_loss_draws):,.0f}]')
print(f'Real # buildings >= DS1..DS4 (mean over draws): '
      f'{[round(np.mean(real_DS_count_draws[l]), 1) for l in (1,2,3,4)]}')

# ------------------------------------------------------------------
# Panel x-axis zoom: around the real-count draws when informative; fall
# back to the simulated max otherwise (e.g. DS4/collapse draws are mostly
# 0 with only rare non-zero counts, so their mean is a small fraction --
# 2x that would collapse the axis to a sliver rather than showing the
# actual, integer-valued spread of the draws).
# ------------------------------------------------------------------
def curve_support_max(curve_set, grid, threshold=1e-4):
    """Largest grid point where any of the shown averaged curves is still
    above `threshold` -- stand-in for "simulated max" now that only the
    averaged exceedance curves are kept, not each draw's raw per-iteration counts."""
    above = np.any(curve_set > threshold, axis=0)
    return grid[np.max(np.nonzero(above))] if np.any(above) else grid[-1]

xlim_by_level = {}
for level in (1, 2, 3, 4):
    mean_real_DS_count = np.mean(real_DS_count_draws[level])
    max_real_DS_count = np.max(real_DS_count_draws[level])
    if mean_real_DS_count >= 1:
        xlim_by_level[level] = max(2 * mean_real_DS_count, 1.2 * max_real_DS_count)
    else:
        shown = np.stack([curves_by_level_avg[level][cid] for cid in PANEL_CURVES[level]])
        xlim_by_level[level] = max(1, int(curve_support_max(shown, d_values)))
xlim_loss = up_xlim


def ref_limit(curve, grid):
    """First grid value where the reference curve is <= XLIM_REF_PROB (grid max if never)."""
    below = np.nonzero(curve <= XLIM_REF_PROB)[0]
    return grid[below[0]] if len(below) else grid[-1]


for level in (1, 2, 3, 4):
    if XLIM_MODE[level] == "ref" and XLIM_REF_CURVE in SHOWN:
        xlim_by_level[level] = max(1, float(ref_limit(curves_by_level_avg[level][XLIM_REF_CURVE], d_values)))
if XLIM_MODE["loss"] == "ref" and XLIM_REF_CURVE in SHOWN:
    xlim_loss = float(ref_limit(loss_curves_avg[XLIM_REF_CURVE], loss_values))
for level in (1, 2, 3, 4):
    if XLIM_MODE[level] == "expected":
        xlim_by_level[level] = max(X_LIMIT_FACTOR[level] * np.mean(expected_DS_count_draws[level]), X_LIMIT_MIN)
if XLIM_MODE["loss"] == "expected":
    xlim_loss = X_LIMIT_FACTOR["loss"] * np.mean(expected_loss_draws)
print(f"x-limits (ref = {XLIM_REF_CURVE} at P = {XLIM_REF_PROB:g}): "
      + ", ".join(f"DS{l} {xlim_by_level[l]:.1f} ({XLIM_MODE[l]})" for l in (1, 2, 3, 4))
      + f", loss {xlim_loss / 1e6:.2f}M ({XLIM_MODE['loss']}) | expected under independence: "
      + ", ".join(f"DS{l} {np.mean(expected_DS_count_draws[l]):.2f}" for l in (1, 2, 3, 4))
      + f", loss {np.mean(expected_loss_draws) / 1e6:.2f}M")

# ------------------------------------------------------------------
# Merged 2x3 A4-sized figure:
#   [ DS1 ] [ DS2 ] [ DS3 ]
#   [ DS4 ] [ Loss ] [ legend ]
#
# The "real scenario" reference uncertainty is drawn as a classic mean
# +/- 1,2 std fan (add_std_bands) -- preferred over an earlier
# density/KDE-based band for being simpler to read at a glance, even
# though it assumes symmetry and hides the actual shape of the draws.
# ------------------------------------------------------------------

mpl.rcParams['font.family'] = 'Helvetica'

# One consistent reference-line style/label across all 5 panels (its x
# position differs per panel, but the shared legend just needs one entry).
ref_style = dict(color='red', linestyle='--', linewidth=2, label="Real Scenario: Mean")


def add_std_bands(ax, draws, color='red', alpha_2sigma=0.15, alpha_1sigma=0.30):
    """Mean +/- 1,2 sigma fan: two flat-shaded spans (2 sigma lighter, 1
    sigma darker), using the sample standard deviation (ddof=1) since
    these are estimated from a finite sample of N_OUTER_DRAWS Monte Carlo
    draws. Skips drawing anything if the draws have (numerically) no
    spread."""
    draws = np.asarray(draws, dtype=float)
    sigma = np.std(draws, ddof=1)
    if sigma < 1e-9:
        return
    mean = np.mean(draws)
    ax.axvspan(mean - 2*sigma, mean + 2*sigma, color=color, alpha=alpha_2sigma, zorder=0, lw=0)
    ax.axvspan(mean - sigma, mean + sigma, color=color, alpha=alpha_1sigma, zorder=0, lw=0)


def render_figure(output_path):
    """Reuses the nested-Monte-Carlo results already computed above."""
    PANEL_SIZE = 2.1
    fig, axes = plt.subplots(2, 3, figsize=(3*PANEL_SIZE, 2*PANEL_SIZE), dpi=500, sharey=True)
    ax_ds1, ax_ds2, ax_ds3 = axes[0]
    ax_ds4, ax_loss, ax_legend = axes[1]

    ds_axes = [ax_ds1, ax_ds2, ax_ds3, ax_ds4]
    for level, ax in zip((1, 2, 3, 4), ds_axes):
        add_std_bands(ax, real_DS_count_draws[level])

        mean_DS_count = np.mean(real_DS_count_draws[level])
        ax.axvline(mean_DS_count, **ref_style)
        # Actual value labeled just above each panel, at the line's x position
        # (axes-fraction y so it sits consistently above the plot regardless of
        # the log-scale y-limits).
        ax.text(mean_DS_count, 1.02, f"{mean_DS_count:.1f}",
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=8, color='red', clip_on=False)
        for cid in PANEL_CURVES[level]:
            ax.plot(d_values, curves_by_level_avg[level][cid], zorder=3, **curve_style(cid))
            # direct sampling (no sequential conditioning), faint, behind the sequential curve
            model = cid.split(":")[1]
            if level in DIRECT_LEVELS and cid.startswith("DS:") and model in DS_CHAIN_MODELS:
                ax.plot(d_values, direct_curves_avg[level][model], color=MODEL_COLORS[model],
                        linestyle="-", linewidth=DIRECT_LW, alpha=DIRECT_ALPHA, zorder=2)

        ax.set_xlabel(f"# Buildings ≥ DS{level}", fontsize=9)
        ax.set_ylim((0.0001, 1))
        ax.set_xlim(0, xlim_by_level[level])
        ax.tick_params(labelsize=9)
        ax.set_yscale('log')
        ax.grid(True, which='major', alpha=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle='--')

    # Loss in millions of USD reads far more naturally than raw dollars.
    loss_values_M = loss_values / 1e6
    real_loss_draws_M = real_loss_draws / 1e6
    up_xlim_M = xlim_loss / 1e6

    add_std_bands(ax_loss, real_loss_draws_M)

    mean_loss_M = np.mean(real_loss_draws_M)
    ax_loss.axvline(mean_loss_M, **ref_style)
    ax_loss.text(mean_loss_M, 1.02, f"{mean_loss_M:.1f}M",
                 transform=ax_loss.get_xaxis_transform(), ha='center', va='bottom',
                 fontsize=8, color='red', clip_on=False)
    for cid in PANEL_CURVES["loss"]:
        ax_loss.plot(loss_values_M, loss_curves_avg[cid], zorder=3, **curve_style(cid))

    ax_loss.set_xlabel("Total Loss (Million USD)", fontsize=9)
    ax_loss.set_ylim((0.0001, 1))
    ax_loss.set_xlim(0, up_xlim_M)
    ax_loss.tick_params(labelsize=9)
    ax_loss.set_yscale('log')
    ax_loss.grid(True, which='major', alpha=0.5)
    ax_loss.grid(True, which='minor', alpha=0.15, linestyle='--')

    # Separate y-axis label per row (leftmost panel of each), instead of one
    # label shared across the whole figure.
    ax_ds1.set_ylabel("Probability of Exceedance", fontsize=9)
    ax_ds4.set_ylabel("Probability of Exceedance", fontsize=9)

    # Legend occupies the sixth grid cell: mean line, its sigma bands, then
    # every curve shown in at least one panel, in CURVE_ORDER (built from
    # proxies so it does not depend on which panel shows which curve).
    ax_legend.axis('off')
    handles = [Line2D([], [], **ref_style),
               Patch(color='red', alpha=0.30, lw=0, label="Real Scenario: Mean ± 1σ"),
               Patch(color='red', alpha=0.15, lw=0, label="Real Scenario: Mean ± 2σ")]
    handles += [Line2D([], [], **curve_style(cid)) for cid in SHOWN]
    if DS_CHAIN_MODELS and DIRECT_LEVELS:
        handles.append(Line2D([], [], color="0.4", linestyle="-", linewidth=DIRECT_LW, alpha=DIRECT_ALPHA,
                              label="DS-based, direct sampling"))
    ax_legend.legend(handles, [h.get_label() for h in handles], fontsize=7, loc='center', frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f'wrote {output_path}')


_tag = "_".join(c.replace(":", "") for c in SHOWN if c.split(":")[1] != "Independent")
render_figure(Path(__file__).parent/f"comparison_results-multiDS-DSallpanel-Uncertainty-{_tag}-direct.pdf")

# ------------------------------------------------------------------
# Probability, per model, of exceeding the mean real scenario -- i.e. each
# case's curve evaluated (via interpolation on its shared grid) at the
# mean reference value plotted as the red dashed line in each panel.
# ------------------------------------------------------------------
print("\nProbability of exceeding the mean real scenario, by model:")
for level in (1, 2, 3, 4):
    mean_DS_count = np.mean(real_DS_count_draws[level])
    print(f"  DS{level} (mean real = {mean_DS_count:.1f} buildings):")
    for cid in PANEL_CURVES[level]:
        p = np.interp(mean_DS_count, d_values, curves_by_level_avg[level][cid])
        line = f"    {curve_style(cid)['label']:28s}: {p:.4f}"
        model = cid.split(":")[1]
        if level in DIRECT_LEVELS and cid.startswith("DS:") and model in DS_CHAIN_MODELS:
            p_dir = np.interp(mean_DS_count, d_values, direct_curves_avg[level][model])
            line += f"   (direct sampling: {p_dir:.4f})"
        print(line)

mean_loss_M = np.mean(real_loss_draws) / 1e6
print(f"  Loss (mean real = {mean_loss_M:.1f}M USD):")
for cid in PANEL_CURVES["loss"]:
    p = np.interp(mean_loss_M, loss_values / 1e6, loss_curves_avg[cid])
    print(f"    {curve_style(cid)['label']:28s}: {p:.4f}")
