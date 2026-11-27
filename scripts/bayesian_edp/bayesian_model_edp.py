#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from pathlib import Path
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd
import numpy as np
import jax
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import hashlib
import os
import pickle
import shutil
import signal
from datetime import datetime
import sys
from collections import defaultdict
from numpyro.diagnostics import gelman_rubin, effective_sample_size

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


# Bumped whenever the building-to-station sampling changes, so the per-iteration
# cache of an older sampling is never reused (it is part of the fingerprint).
# v2: buildings drawn independently at each station, without replacement
# (v1 gave every station the same set, with replacement). Used for Part II.
SAMPLING_VERSION = "per-station-v2"


# ===========================================================================
# DATA HELPERS
# ===========================================================================

def data_fingerprint(df, *settings):
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(repr(settings).encode())
    return h.hexdigest()[:10]


def sample_buildings(df, n_bst, random_state):
    """n_bst distinct buildings per station, drawn independently at each station.
    Deterministic for a given random_state and input table."""
    rng = np.random.default_rng(random_state)
    return (df.assign(_k=rng.random(len(df)))
              .sort_values("_k", kind="stable")
              .groupby(["eqid", "RSN"], sort=False).head(n_bst)
              .drop(columns="_k")
              .sort_values(["eqid", "RSN", "building"], kind="stable")
              .reset_index(drop=True))


# ===========================================================================
# IN-PROCESS TIMEOUT (SIGALRM-BASED)
# ===========================================================================
# Each worker process runs MCMC in-process and uses signal.alarm to enforce
# a wall-clock budget. When the alarm fires, Python receives a SIGALRM and
# raises TimeoutError on the next bytecode boundary. JAX/XLA C++ kernels
# may delay delivery until they return to Python, but in practice this
# fires reliably between MCMC iterations. This avoids the doubled memory
# / process overhead of running each task inside a grandchild subprocess.

class TaskTimeout(Exception):
    """Raised when a task exceeds its wall-clock budget."""
    pass


def _alarm_handler(signum, frame):
    raise TaskTimeout("MCMC exceeded wall-clock budget")


# ===========================================================================
# CONVERGENCE CHECK
# ===========================================================================

def _check_convergence(res_chains, rhat_threshold=1.1, ess_threshold=200):
    skip_keys = {"c1", "c2_pow_c1"}

    for key, chains in res_chains.items():
        if key in skip_keys or key.startswith("z_"):
            continue

        arr = np.array(chains)

        if arr.ndim == 0 or arr.size == 0:
            print(f"⚠️  '{key}': empty or scalar array — discarding.", flush=True)
            return False

        rhat     = gelman_rubin(arr)
        max_rhat = float(np.max(rhat))
        if max_rhat > rhat_threshold:
            print(f"⚠️  R-hat={max_rhat:.3f} for '{key}' (threshold {rhat_threshold}) — discarding.", flush=True)
            return False

        ess     = effective_sample_size(arr)
        min_ess = float(np.min(ess))
        if min_ess < ess_threshold:
            print(f"⚠️  ESS={min_ess:.1f} for '{key}' (threshold {ess_threshold}) — discarding.", flush=True)
            return False

        print(f"✅  '{key}': R-hat={max_rhat:.3f}, ESS={min_ess:.1f}", flush=True)

    return True


# ===========================================================================
# MODEL RUNNER
# ===========================================================================

def run_model(m, seed, train_data, n_chains):
    print(f"  Running {m.__name__} | seed={seed} | chains={n_chains} ...", flush=True)
    kernel = NUTS(m)
    mcmc   = MCMC(kernel, num_warmup=2000, num_samples=1000,
                  num_chains=n_chains, chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), **train_data)

    samples_flat   = mcmc.get_samples(group_by_chain=False)
    samples_chains = mcmc.get_samples(group_by_chain=True)

    print(f"  Finished {m.__name__}", flush=True)
    return samples_flat, samples_chains


# ===========================================================================
# TIMED MCMC (IN-PROCESS, SIGALRM)
# ===========================================================================
# Runs MCMC in the current process, with signal.alarm enforcing a
# wall-clock timeout. Returns a (status, payload) tuple matching the
# previous subprocess-based interface so the rest of the code is unchanged.

def run_mcmc_with_timeout(m, seed, train_data, n_chains,
                          rhat_threshold, ess_threshold, timeout_s):
    # Install our SIGALRM handler and start the countdown.
    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout_s))
    try:
        samples_flat, samples_chains = run_model(m, seed, train_data, n_chains)
        converged = _check_convergence(samples_chains, rhat_threshold, ess_threshold)
        if converged:
            return ("ok", samples_flat)
        return ("not_converged", None)
    except TaskTimeout:
        return ("timeout", None)
    except Exception as e:
        return ("error", repr(e))
    finally:
        # Always cancel the alarm and restore the previous handler so the
        # alarm can never fire after we've returned.
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


# ===========================================================================
# SINGLE TASK  — one (iteration × model) pair, runs inside a worker process
# ===========================================================================

def run_single(it, m_idx, m, seed, df_all, eqids, n_bst, n_chains,
               base_seed, rhat_threshold, ess_threshold,
               per_iter_dir, timeout_s):
    """
    Run one model for one bootstrap iteration.

    All models within the same iteration share the same bootstrap sample
    (same data_seed) so results are comparable across models.

    Saves the result to `per_iter_dir/it{it:03d}_m{m_idx}.pkl` immediately
    on completion (atomic write). If a result file already exists, the
    task is skipped (resume-from-disk).

    Returns
    -------
    (it, m_idx, samples_flat) or (it, m_idx, None) if failed/not converged/timeout
    """
    import numpyro

    numpyro.set_platform("cpu")  # must be called before any JAX computation

    out_file = Path(per_iter_dir) / f"it{it:03d}_m{m_idx}.pkl"

    # --- Resume: skip if already done ---
    if out_file.exists():
        try:
            with open(out_file, "rb") as f:
                cached = pickle.load(f)
            print(f"[it={it+1}, {m.__name__}] ⏭  found cached result, skipping.", flush=True)
            return cached
        except Exception as e:
            print(f"[it={it+1}, {m.__name__}] cached file unreadable ({e}) — recomputing.", flush=True)

    print(f"[it={it+1}, {m.__name__}] Starting ...", flush=True)

    # --- Bootstrap sample: distinct buildings per station ---
    # (eps already scaled once in main, with the same factor as the test set)
    data_seed = base_seed + it * 1000
    df = sample_buildings(df_all, n_bst, random_state=data_seed)

    # --- Prepare model inputs ---
    Xtot, ztot = [], []
    for eqid in eqids:
        group = df[df.eqid == eqid].copy()
        Xtot.append(group[['latitude', 'longitude', 'period', 'Say']].values)
        ztot.append(group['eps'].values)
    train_data = dict(X=Xtot, eqids=eqids, z=ztot)

    # --- Run MCMC with hard timeout ---
    status, payload = run_mcmc_with_timeout(
        m, seed, train_data, n_chains,
        rhat_threshold, ess_threshold, timeout_s,
    )

    if status == "ok":
        samples_flat = payload
        print(f"[it={it+1}, {m.__name__}] ✅ passed.", flush=True)
        result = (it, m_idx, samples_flat)
    elif status == "not_converged":
        print(f"[it={it+1}, {m.__name__}] ⚠️  failed convergence — discarding.", flush=True)
        result = (it, m_idx, None)
    elif status == "timeout":
        print(f"[it={it+1}, {m.__name__}] ⏱  hit {timeout_s/3600:.1f}h timeout — discarding.", flush=True)
        result = (it, m_idx, None)
    else:  # "error"
        print(f"[it={it+1}, {m.__name__}] ❌ crashed — {payload}", flush=True)
        result = (it, m_idx, None)

    # --- Atomic write of per-iteration result ---
    try:
        tmp = out_file.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(result, f)
        os.replace(tmp, out_file)
    except Exception as e:
        print(f"[it={it+1}, {m.__name__}] ⚠️ could not save per-iter result: {e}", flush=True)

    return result


# Wrapper for imap_unordered (single-arg) — unpacks the task tuple.
def _run_single_unpack(args):
    return run_single(*args)


# ===========================================================================
# PLOTTING
# ===========================================================================

def estimate_histogram(data, bins=50, title="Histogram of Monte Carlo Results",
                       density=False, output_dir=None):
    if data.size == 0:
        print("Error: Input data array is empty.")
        return

    data     = data[~np.isnan(data)]
    if data.size == 0:
        return
    mean_val = np.mean(data)
    std_val  = np.std(data)
    data     = data[(data >= mean_val - 4.5 * std_val) & (data <= mean_val + 4.5 * std_val)]
    if data.size == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, density=density,
             alpha=0.7, color='#4A90E2', edgecolor='black')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Outcome Value", fontsize=14)
    plt.ylabel("Probability Density" if density else "Frequency (Counts)", fontsize=14)
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    mean_val = np.mean(data)
    std_dev  = np.std(data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2,
                label=f'Mean: {mean_val:.4f}')
    stats_text = (f"Simulations: {len(data)}\n"
                  f"Mean: {mean_val:.4f}\n"
                  f"Std Dev: {std_dev:.4f}")
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.6, ec="#333333"))
    plt.legend()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename_final = os.path.join(output_dir, f"{title.replace(' ', '_')}.pdf")
        # Use a temp name with a real .pdf extension so matplotlib picks
        # the right format, then atomically rename to the final name.
        filename_tmp   = filename_final + ".tmp.pdf"
        try:
            plt.savefig(filename_tmp, dpi=300, bbox_inches="tight")
            os.replace(filename_tmp, filename_final)
        finally:
            plt.close('all')
    else:
        plt.show()


def summarize_samples(df_samples, ci=95, round_digits=4):
    results   = {}
    lower_pct = (100 - ci) / 2
    upper_pct = 100 - lower_pct
    for col in df_samples.columns:
        samples  = df_samples[col].values
        mean_val = np.mean(samples)
        ci_lower = np.percentile(samples, lower_pct)
        ci_upper = np.percentile(samples, upper_pct)
        results[col] = {
            "mean": round(mean_val, round_digits),
            f"{ci}% CI": (round(ci_lower, round_digits), round(ci_upper, round_digits))
        }
    results_df = pd.DataFrame(results).T
    print(results_df)
    return results_df


# ===========================================================================
# INCREMENTAL AGGREGATION + REFRESH
# ===========================================================================

def aggregate_and_refresh(all_results, models, vars_per_model,
                          npz_path, plot_dir, label=""):
    """
    Build per-model arrays from all results received so far and atomically
    refresh the aggregate .npz file and histogram PDFs.
    """
    results_accum = [defaultdict(list) for _ in range(len(models))]
    valid_counts  = [0] * len(models)

    for it, m_idx, samples in all_results:
        if samples is None:
            continue
        for k, v in samples.items():
            results_accum[m_idx][k].append(v)
        valid_counts[m_idx] += 1

    print(f"\n[refresh{(' ' + label) if label else ''}] Convergence pass rates so far:", flush=True)
    for m_idx, (m, count) in enumerate(zip(models, valid_counts)):
        print(f"  Model {m_idx} ({m.__name__}): {count} iterations passed", flush=True)

    results_all      = []
    df_param_samples = []

    for m_idx, (res_acc, var_list) in enumerate(zip(results_accum, vars_per_model)):
        arrays_dict = {}
        for k, v_list in res_acc.items():
            arrs = [jnp.array(v) for v in v_list if v is not None and len(v) > 0]
            if arrs:
                arrays_dict[k] = jnp.concatenate(arrs, axis=0)

        if not arrays_dict:
            df_param_samples.append(pd.DataFrame())
            results_all.append({})
            continue

        df_temp = pd.DataFrame(arrays_dict)
        for v in var_list:
            if v not in df_temp.columns:
                df_temp[v] = np.array([])
        df_param_samples.append(df_temp[var_list])
        results_all.append(arrays_dict)

    # --- Atomic .npz refresh ---
    try:
        npz_tmp = str(npz_path) + ".tmp.npz"
        np.savez(
            npz_tmp,
            **{f"model_{i}_{k}": np.asarray(v)
               for i, model_dict in enumerate(results_all)
               for k, v in model_dict.items()}
        )
        os.replace(npz_tmp, npz_path)
    except Exception as e:
        print(f"[refresh] ⚠️ could not save npz: {e}", flush=True)

    # --- Histogram refresh ---
    try:
        for m_idx, (m, dfp) in enumerate(zip(models, df_param_samples)):
            if dfp.empty:
                continue
            tag = "HT" if m.__name__ == "modelHT" else ("HTS" if m.__name__ == "modelHTS" else m.__name__)
            for var in dfp.columns:
                arr = np.asarray(dfp[var])
                if arr.size == 0:
                    continue
                estimate_histogram(arr, bins=30,
                                   title=f'{tag} - Histogram for {var}',
                                   density=False, output_dir=plot_dir)
    except Exception as e:
        print(f"[refresh] ⚠️ histogram refresh failed: {e}", flush=True)

    return results_all, df_param_samples, valid_counts


# ===========================================================================
# KERNEL & DISTANCE HELPERS
# ===========================================================================

def getHaversineDistance(X, R=6371.0):
    lat  = jnp.radians(X[:, 0]);  lon  = jnp.radians(X[:, 1])
    lat1 = lat[:, None];           lat2 = lat[None, :]
    lon1 = lon[:, None];           lon2 = lon[None, :]
    dlat = lat2 - lat1;            dlon = lon2 - lon1
    a    = jnp.sin(dlat / 2.0)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon / 2.0)**2
    c    = 2.0 * jnp.arcsin(jnp.sqrt(a))
    d    = R * c
    return d.at[jnp.diag_indices_from(d)].set(0.0)

def getPeriodDissimilarity(X):
    sq = jnp.square(jnp.reshape(X[:, 2], [-1, 1]) - jnp.reshape(X[:, 2], [1, -1]))
    return jnp.sqrt(jnp.clip(sq, 0, np.inf))

def getSayDissimilarity(X):
    sq = jnp.square(jnp.reshape(X[:, 3], [-1, 1]) - jnp.reshape(X[:, 3], [1, -1]))
    return jnp.sqrt(jnp.clip(sq, 0, np.inf))


# ===========================================================================
# COVARIANCE KERNELS
# ===========================================================================

def rhoH(X, c2_pow_c1, c1, nugget=1e-4):
    K = jnp.exp(-jnp.multiply(jnp.power(getHaversineDistance(X), c1), 1.0 / c2_pow_c1))
    return K.at[jnp.diag_indices(X.shape[0])].add(nugget)

def rhoHT(X, c2_pow_c1, c1, c4, nugget=1e-4):
    KH = jnp.exp(-jnp.multiply(jnp.power(getHaversineDistance(X), c1), 1.0 / c2_pow_c1))
    KT = jnp.exp(-getPeriodDissimilarity(X) / c4)
    K  = KH * KT
    return K.at[jnp.diag_indices(X.shape[0])].add(nugget)

def rhoHTS(X, c2_pow_c1, c1, c4, c5, c3, nugget=1e-4):
    KH = jnp.exp(-jnp.multiply(jnp.power(getHaversineDistance(X), c1), 1.0 / c2_pow_c1))
    KT = jnp.exp(-getPeriodDissimilarity(X) / c4)
    KS = jnp.exp(-getSayDissimilarity(X) / c5)
    K  = KH * (c3 * KT + (1 - c3) * KS)
    return K.at[jnp.diag_indices(X.shape[0])].add(nugget)


# ===========================================================================
# PROBABILISTIC MODELS
# ===========================================================================

def modelH(X, eqids, z):
    
    c2 = numpyro.sample("c2",dist.TruncatedNormal(loc=15.0, scale=15.0, low=0.01, high=100.0),)
    
    #c2     = numpyro.sample("c2",     dist.InverseGamma(concentration=2, rate=30))
    c1_half = numpyro.sample("c1_half", dist.Beta(2, 2))
    c1 = numpyro.deterministic("c1", 2.0 * c1_half)
    c2_pow_c1    = numpyro.deterministic("c2_pow_c1",    jnp.power(c2, c1))
    z = [numpyro.sample(f"z_{eqid}",
                        dist.MultivariateNormal(0, rhoH(X[i], c2_pow_c1, c1)),
                        obs=z[i])
         for i, eqid in enumerate(eqids)]

def modelHT(X, eqids, z):
    c2 = numpyro.sample("c2",dist.TruncatedNormal(loc=15.0, scale=15.0, low=0.01, high=100.0),)
    #c2     = numpyro.sample("c2",     dist.InverseGamma(concentration=2, rate=8))
    c1_half = numpyro.sample("c1_half", dist.Beta(2, 2))
    c1 = numpyro.deterministic("c1", 2.0 * c1_half)
    c2_pow_c1    = numpyro.deterministic("c2_pow_c1",    jnp.power(c2, c1))

    mean_c4 = 0.1794;  sd_c4 = 0.0346*2.5
    var_c4   = sd_c4**2
    sigma_c4 = jnp.sqrt(jnp.log(1 + var_c4 / mean_c4**2))
    mu_c4    = jnp.log(mean_c4) - 0.5 * sigma_c4**2
    c4 = numpyro.sample("c4", dist.LogNormal(mu_c4, sigma_c4))

    z = [numpyro.sample(f"z_{eqid}",
                        dist.MultivariateNormal(0, rhoHT(X[i], c2_pow_c1, c1, c4)),
                        obs=z[i])
         for i, eqid in enumerate(eqids)]

def modelHTS(X, eqids, z):
    c2 = numpyro.sample("c2",dist.TruncatedNormal(loc=15.0, scale=15.0, low=0.01, high=100.0),)
    #c2     = numpyro.sample("c2",     dist.InverseGamma(concentration=2, rate=4))
    c1_half = numpyro.sample("c1_half", dist.Beta(2, 2))
    c3_raw     = numpyro.sample("c3_raw",     dist.Beta(2, 2))

    c3_min = 0.500
    c3_max = 0.999

    c1 = numpyro.deterministic("c1", 2.0 * c1_half)
    c2_pow_c1    = numpyro.deterministic("c2_pow_c1",    jnp.power(c2, c1))
    c3      = numpyro.deterministic("c3",      c3_min + (c3_max - c3_min) * c3_raw)

    mean_c4 = 0.1794;  sd_c4 = 0.0346*2.5
    var_c4  = sd_c4**2
    sigma_c4 = jnp.sqrt(jnp.log(1 + var_c4 / mean_c4**2))
    mu_c4    = jnp.log(mean_c4) - 0.5 * sigma_c4**2
    c4 = numpyro.sample("c4", dist.LogNormal(mu_c4, sigma_c4))

    mean_c5 = 0.1713;  sd_c5 = 0.0296*2.5
    var_c5   = sd_c5**2
    sigma_c5 = jnp.sqrt(jnp.log(1 + var_c5 / mean_c5**2))
    mu_c5    = jnp.log(mean_c5) - 0.5 * sigma_c5**2
    c5 = numpyro.sample("c5", dist.LogNormal(mu_c5, sigma_c5))

    z = [numpyro.sample(f"z_{eqid}",
                        dist.MultivariateNormal(0, rhoHTS(X[i], c2_pow_c1, c1, c4, c5, c3)),
                        obs=z[i])
         for i, eqid in enumerate(eqids)]


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CONFIG
    # -----------------------------------------------------------------------
    N              = 50    # bootstrap iterations
    n_bst          = 3     # buildings sampled per station per iteration
    n_chains       = 3     # MCMC chains per model run (run sequentially)
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count())) - 1


    rhat_threshold = 1.1
    ess_threshold  = 200

    eqids          = [137, 278]   # Chi-Chi and Chuetsu (fitting events, Part II Table 1)
    models         = [modelHT, modelHTS]
    seeds          = [0, 23]   # one MCMC seed per model, fixed across iterations
    base_seed      = 24        # controls bootstrap sampling per iteration

    # Hard wall-clock timeout per (iteration × model) task. If MCMC for a
    # single task does not finish within this, the task is terminated and
    # treated as a failure. 5h * 3600 = 18000s (covers 2.5h * 2 models).
    timeout_s      = 10 * 3600

    # Refresh aggregated .npz + histograms every this many completed tasks.
    refresh_every  = 10

    eqids_test     = [127] # events used for the LPPD
    lppd_n_alloc   = 20    # fixed test allocations, seeds base_seed + k, k = 0..lppd_n_alloc-1
    lppd_n_bst     = 3     # distinct buildings per station in each test allocation

    vars_per_model = [
        ["c2", "c1", "c4"],
        ["c2", "c1", "c4", "c5", "c3"]
    ]
    # -----------------------------------------------------------------------

    import functools
    print = functools.partial(print, flush=True)

    # --- Load data ---
    stations  = pd.read_csv(DATA / "pooled" / "stations.csv")
    eps       = pd.read_csv(DATA / "pooled" / "eps_PSD.csv")
    buildings = pd.read_csv(DATA / "portfolio" / "buildings.txt", names=["building"])

    with open(DATA / "portfolio" / "modal_results.json")    as f: build_modal    = json.load(f)
    with open(DATA / "portfolio" / "bilinear_results.json") as f: build_bilinear = json.load(f)

    T, Say = [], []
    for b in buildings['building']:
        T.append(build_modal[b]['MX']['period'])
        Say.append(build_bilinear[b]['X']['force'][1])
    buildings["period"] = T
    buildings["Say"]    = Say

    eps      = eps.merge(stations[["RSN", "EQID"]], on="RSN", how="inner")
    eps_long = eps.melt(id_vars=["EQID", "RSN", "latitude", "longitude"],
                        var_name="building", value_name="eps")
    eps_long["building"] = eps_long["building"].str.replace("^eps_", "", regex=True)

    df = eps_long.merge(buildings, on="building", how="left")
    df = df[["EQID", "RSN", "building", "eps", "latitude", "longitude", "period", "Say"]]
    df = df.rename(columns={"EQID": "eqid"})
    df = df.dropna()
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

    # One fixed scale for eps, applied to both the training samples and the
    # LPPD test sets (previously each bootstrap sample was divided by its own
    # std while the test eps stayed raw).
    # Scale = std of the residuals used for the calibration (fitting events `eqids`, stations in eps_PSD.csv),
    # applied to the training samples and to the LPPD test sets alike.
    eps_scale = df.loc[df.eqid.isin(eqids), "eps"].std()
    df["eps"] = df["eps"] / eps_scale

    df_all = df.reset_index(drop=True)

    fingerprint = data_fingerprint(df_all, eqids, n_bst, n_chains, base_seed, seeds,
                                   [m.__name__ for m in models], SAMPLING_VERSION)

    output_dir   = Path(__file__).parent / "results_iterations-exp"
    per_iter_dir = output_dir / f"per_iteration-{fingerprint}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(per_iter_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir  = str(output_dir / f"plots_{timestamp}")
    npz_path  = Path(__file__).parent / f'results_all_{timestamp}.npz'

    print(f"eps scale (std of the fitting-event residuals): {eps_scale:.4f}")
    print(f"Per-iter dir: {per_iter_dir}")

    # --- Build flattened task list: one entry per (iteration × model) ---
    tasks = [
        (it, m_idx, m, seeds[m_idx], df_all, eqids, n_bst, n_chains,
         base_seed, rhat_threshold, ess_threshold,
         str(per_iter_dir), timeout_s)
        for it in range(N)
        for m_idx, m in enumerate(models)
    ]

    n_tasks = len(tasks)
    print(f"\nLaunching {n_tasks} tasks ({N} iterations × {len(models)} models) "
          f"across {n_cpus} workers ...")
    print(f"Per-task timeout: {timeout_s/3600:.1f}h | "
          f"Refresh every: {refresh_every} completions | "
          f"Per-iter dir: {per_iter_dir}")

    all_results = []
    completed   = 0
    success     = False

    try:
        with mp.get_context("spawn").Pool(processes=n_cpus) as pool:
            for result in pool.imap_unordered(_run_single_unpack, tasks):
                all_results.append(result)
                completed += 1
                it, m_idx, samples = result
                status = "ok" if samples is not None else "failed"
                print(f"  → received it={it+1} m_idx={m_idx} ({status}) "
                      f"[{completed}/{n_tasks}]", flush=True)

                # Periodic refresh of aggregate .npz + histograms
                if completed % refresh_every == 0:
                    aggregate_and_refresh(
                        all_results, models, vars_per_model,
                        npz_path, plot_dir,
                        label=f"after {completed}/{n_tasks}",
                    )

        print("\nAll tasks complete!")
        success = True
    finally:
        # Always do a final refresh so latest state is on disk, even if we
        # crashed or were interrupted partway through.
        try:
            aggregate_and_refresh(
                all_results, models, vars_per_model,
                npz_path, plot_dir,
                label="final",
            )
        except Exception as e:
            print(f"⚠️ final refresh failed: {e}", flush=True)

    # --- Log results to file ---
    original_stdout = sys.stdout
    log_file  = os.path.join(output_dir, f"run_log_{timestamp}.txt")
    sys.stdout = open(log_file, "w")

    print(f"Logging to {log_file}")
    print(f"N={N}, n_bst={n_bst}, n_chains={n_chains}, n_cpus={n_cpus}")
    print(f"R-hat threshold={rhat_threshold}, ESS threshold={ess_threshold}")
    print(f"Sampling: {SAMPLING_VERSION}, per-iter dir {per_iter_dir}, eps scale {eps_scale:.4f}")
    print(f"Per-task timeout: {timeout_s/3600:.1f}h, refresh every {refresh_every} tasks\n")

    # --- Accumulate results (final) ---
    results_accum_global = [defaultdict(list) for _ in range(len(models))]
    valid_counts_global  = [0] * len(models)

    for it, m_idx, samples in all_results:
        if samples is None:
            continue
        for k, v in samples.items():
            results_accum_global[m_idx][k].append(v)
        valid_counts_global[m_idx] += 1

    print("Convergence pass rates:")
    for m_idx, (m, count) in enumerate(zip(models, valid_counts_global)):
        print(f"  Model {m_idx} ({m.__name__}): {count}/{N} iterations passed "
              f"({100 * count / N:.1f}%)")

    # --- Build per-model DataFrames ---
    results_all      = []
    df_param_samples = []
    models_ok        = []   # models with valid results, aligned with results_all

    for m_idx, (res_acc, var_list) in enumerate(zip(results_accum_global, vars_per_model)):
        arrays_dict = {}
        for k, v_list in res_acc.items():
            arrs = [jnp.array(v) for v in v_list if v is not None and len(v) > 0]
            if arrs:
                arrays_dict[k] = jnp.concatenate(arrs, axis=0)

        if not arrays_dict:
            print(f"Model {m_idx} produced no valid results — skipping.")
            continue

        df_temp = pd.DataFrame(arrays_dict)
        missing = [v for v in var_list if v not in df_temp.columns]
        if missing:
            print(f"Model {m_idx} missing vars {missing} — skipping.")
            continue

        df_param_samples.append(df_temp[var_list])
        results_all.append(arrays_dict)
        models_ok.append(models[m_idx])

    params_1 = pd.concat([df_param_samples[0]], ignore_index=True) if len(df_param_samples) > 0 else pd.DataFrame()
    params_2 = pd.concat([df_param_samples[1]], ignore_index=True) if len(df_param_samples) > 1 else pd.DataFrame()

    # --- Summaries ---
    print("\n--- Model HT ---")
    mean1 = summarize_samples(params_1)
    print("\n--- Model HTS ---")
    mean2 = summarize_samples(params_2)

    # --- Histograms (final) ---
    for var in params_1.columns:
        estimate_histogram(np.array(params_1[var]), bins=30,
                           title=f'HT - Histogram for {var}',
                           density=False, output_dir=plot_dir)
    for var in params_2.columns:
        estimate_histogram(np.array(params_2[var]), bins=30,
                           title=f'HTS - Histogram for {var}',
                           density=False, output_dir=plot_dir)

    # --- Save all samples (final, atomic) ---
    final_npz = str(Path(__file__).parent / f'results_all_{timestamp}.npz')
    final_tmp = final_npz + ".tmp.npz"
    np.savez(final_tmp,
             **{f"model_{i}_{k}": v
                for i, model_dict in enumerate(results_all)
                for k, v in model_dict.items()})
    os.replace(final_tmp, final_npz)

    # --- LPPD model comparison ---
    # lppd_n_alloc fixed test allocations (lppd_n_bst distinct buildings per
    # station, seeds base_seed + k), the same for every model. eps of df_all
    # are already scaled by eps_scale, like the training samples.
    from numpyro.infer import log_likelihood
    from scipy.special import logsumexp

    df_test_all = df_all[df_all.eqid.isin(eqids_test)]
    test_sets = []
    for k in range(lppd_n_alloc):
        df_test = sample_buildings(df_test_all, lppd_n_bst, random_state=base_seed + k)
        Xtot, ztot = [], []
        for eqid in eqids_test:
            group = df_test[df_test.eqid == eqid]
            Xtot.append(group[['latitude', 'longitude', 'period', 'Say']].values)
            ztot.append(group['eps'].values)
        test_sets.append((base_seed + k, dict(X=Xtot, eqids=eqids_test, z=ztot)))

    for m, res in zip(models_ok, results_all):
        n_r = len(np.asarray(res["c2"]))   # this model's own number of posterior draws

        print(f"\nLPD per test allocation ({n_r} draws, {lppd_n_bst} buildings per station) — {m.__name__}")
        print(f"  {'alloc':>5} {'seed':>6} {'LPD_model':>11} {'LPD_ind':>11} {'diff [nats]':>12}")

        rows = []
        for k, (seed_k, test_data) in enumerate(test_sets):
            # numpyro log_likelihood with parallel=False (the default) returns wrong per-draw values for these MVN models; parallel=True, in chunks
            LL_CHUNK = 200
            loglik_k = np.concatenate([
                np.sum([np.asarray(v) for v in log_likelihood(
                    m, {kk: np.asarray(vv)[i:i + LL_CHUNK] for kk, vv in res.items()}, parallel=True, **test_data).values()], axis=0)
                for i in range(0, n_r, LL_CHUNK)])

            LPD      = -np.log(n_r) + logsumexp(loglik_k)
            ztot_arr = np.hstack(test_data['z'])
            LPD_ind  = -0.5 * (len(ztot_arr) * np.log(2 * np.pi) + np.sum(ztot_arr**2))
            rows.append((k, seed_k, LPD, LPD_ind, LPD - LPD_ind))
            print(f"  {k:>5} {seed_k:>6} {LPD:>11.3f} {LPD_ind:>11.3f} {LPD - LPD_ind:>12.3f}")

            if k == 0:
                estimate_histogram(loglik_k, bins=30,
                                   title=f'Histogram of LogLikelihood of {m.__name__} (test allocation 0)',
                                   density=False, output_dir=plot_dir)

        lpd = pd.DataFrame(rows, columns=["alloc", "seed", "LPD_model", "LPD_ind", "diff_nats"])
        diff_sd = lpd["diff_nats"].std(ddof=1)
        print(f"  {'mean':>5} {'':>6} {lpd['LPD_model'].mean():>11.3f} {lpd['LPD_ind'].mean():>11.3f} "
              f"{lpd['diff_nats'].mean():>12.3f}   (sd of diff {diff_sd:.3f}, "
              f"se {diff_sd / np.sqrt(len(lpd)):.3f})")
        lpd.to_csv(os.path.join(output_dir, f"lpd_{m.__name__}_{timestamp}.csv"), index=False)

    sys.stdout.close()
    sys.stdout = original_stdout
    print("Logging finished.")

    # --- Cleanup: only remove per-iteration .pkl cache on full success ---
    if success:
        try:
            shutil.rmtree(per_iter_dir)
            print(f"Cleaned up per-iteration cache at {per_iter_dir}")
        except Exception as e:
            print(f"⚠️ could not remove per-iteration cache: {e}")
    else:
        print(f"Run did not complete cleanly — keeping per-iteration cache at {per_iter_dir} "
              f"so you can resume.")
