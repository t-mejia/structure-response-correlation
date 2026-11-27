#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model HT for binary damage, sampled with Albert & Chib (1993) data
augmentation, fitted on the pooled geomean table geomean_pooled.csv.

The latent field has a spatial term multiplied by an exponential decay in the
difference of the buildings' fundamental periods, with no nugget:

    u ~ MVN(0, R(theta)),
    R_ij = exp(-(d_ij / c2)^c1) * exp(-|T_i - T_j| / c4)        (rhoHT)
    D_n = 1  iff  u_n > t_n,   t_n = Phi^-1(1 - p_n)

so P(D_n = 1) = p_n exactly (R_ii = 1). theta = (c2, c1, c4) with
    c1 / 2 ~ Beta(2, 2)                      (c1 in (0, 2))
    c2         ~ TruncatedNormal(15, 15) on [0, 100] km
    c4         ~ LogNormal(-1.822, 0.457)        (seconds)
Each sweep: Metropolis on theta given u, then a Gibbs pass drawing every
u_i from its truncated normal given the other latents (see the sampler
section). Same sampler and runner as Bayesian_Damage_ModelE-AlbertChib.py.

geomean_pooled.csv has one row per building-station: eqid, RSN, building, latitude, longitude, PGA, period, Say,
EDP = sqrt(PSD_1 * PSD_2) and IM_<name> = sqrt(IM_1 * IM_2) for every IM.
To add an event, run make_geomean_event.py on its 2-D JSON + station CSV
and paste the rows it writes at the end of geomean_pooled.csv.

D and p for the chosen damage state are computed in memory from that table:
D = 1 iff EDP >= DAMAGE_THRESHOLD, p = P(D=1 | IM) from
fragility_{threshold}.json.

Run with --build-only to stop after computing D and p (no MCMC).

Runner (bootstrap over buildings, parallel tasks, timeout, resume, R-hat /
ESS check, .npz, histograms, log, LPPD) is the one of the PAR scripts.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from pathlib import Path
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import log_ndtr, logsumexp, ndtri
from scipy.special import ndtri as np_ndtri
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import hashlib
import json
import os
import pickle
import shutil
import signal
from datetime import datetime
import sys
from collections import defaultdict
from numpyro.diagnostics import gelman_rubin, effective_sample_size


# ===========================================================================
# DATA CONFIG
# ===========================================================================

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

GEOMEAN_CSV      = DATA / "pooled" / "geomean_pooled.csv"

DAMAGE_THRESHOLD = 0.5      # PSD [%]: DS1 0.5, DS2 1.0, DS3 2.0 (Part II)
IM_TYPE          = "AvgSa2" # IM_<IM_TYPE> column of GEOMEAN_CSV
FRAGILITY_JSON   = DATA / "models" / "downstream" / f"fragility_{DAMAGE_THRESHOLD}.json"
FRAGILITY_KEY    = "AvgSa2" # IM key inside the fragility JSON

# Bumped whenever the building-to-station sampling changes, so the per-iteration
# cache of an older sampling is never reused (it is part of the fingerprint).
# v2: buildings drawn independently at each station (v1 gave every station the same set).
SAMPLING_VERSION = "per-station-v2"


# ===========================================================================
# SAMPLER CONFIG (Albert-Chib)
# ===========================================================================

AC_WARMUP  = 4000    # warmup sweeps per chain (proposal adapted every block)
AC_SAMPLES = 30000   # sampling sweeps per chain, kept every AC_THIN-th (Part II: 30000 for DS1, DS2; 45000 for DS3)
AC_THIN    = 10      # -> AC_SAMPLES / AC_THIN draws kept per chain
AC_BLOCK   = 500     # sweeps per jit-compiled block during sampling
AC_ADAPT_BLOCK = 100 # sweeps per warmup block (one proposal update per block)
AC_TARGET_ACC  = 0.234  # warmup tunes each chain's step size towards this
AC_N_MH    = 3       # Metropolis steps on theta per sweep
AC_PARALLEL_CHAINS = True  # True: the chains of an iteration run in parallel,
                           # one per core (jax.pmap over CPU devices; needs
                           # XLA_FLAGS=--xla_force_host_platform_device_count=<n_chains>,
                           # otherwise falls back to vectorized chains on one core)
AC_EVENT_MEAN = False  # True: one latent mean per event (flat prior), so a
                       # shift common to the whole event (more or less damage
                       # than the fragility predicts) is not read as correlation


# ===========================================================================
# DATA: geomean table -> (D, p) table
# ===========================================================================

def damage_table(pooled, threshold, im_type, fragility):
    """(eqid, RSN, building, latitude, longitude, period, Say, D, p) for one
    damage state: D = 1 iff geomean EDP >= threshold, p = P(D=1 | geomean IM)
    from that threshold's fragility."""
    df = pooled.copy()
    df["D"] = (df["EDP"] >= threshold).astype(int)
    df["p"] = norm.cdf(np.log(df[f"IM_{im_type}"]), loc=fragility["mu"], scale=fragility["beta"])

    # p in {0, 1} gives an infinite threshold on the latent (the old pipeline
    # dropped these as non-finite eps)
    ok = np.isfinite(df["EDP"]) & (df["p"] > 0) & (df["p"] < 1)
    if (~ok).any():
        print(f"  dropping {(~ok).sum()} rows with non-finite EDP or p in {{0, 1}}")
    df = df[ok]

    return df[["eqid", "RSN", "building", "latitude", "longitude",
               "period", "Say", "D", "p"]].reset_index(drop=True)


def data_fingerprint(df, *settings):
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(repr(settings).encode())
    return h.hexdigest()[:10]


# ===========================================================================
# MODEL E: kernel, observation bounds, truncated normal
# ===========================================================================

BOUND_CAP = 10.0            # replaces +-inf bounds; the latent is unit variance
# Floor so ndtri never sees an underflowed 0 (w_i >= ~-37). It only binds
# when an interval lies ~37+ sd from its conditional mean, where the log
# density is already below ~-690, so it never affects the posterior; it
# just keeps those rejected states finite instead of NaN.
_LOG_TINY = np.log(1e-300)
_LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# Correlation kernel (Model HT: spatial term x period term, no nugget)
# ---------------------------------------------------------------------------

def getHaversineDistance(X, R=6371.0):
    lat = jnp.radians(X[:, 0])
    lon = jnp.radians(X[:, 1])
    lat1, lat2 = lat[:, None], lat[None, :]
    lon1, lon2 = lon[:, None], lon[None, :]
    a = jnp.sin((lat2 - lat1) / 2.0) ** 2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin((lon2 - lon1) / 2.0) ** 2
    d = R * 2.0 * jnp.arcsin(jnp.sqrt(a))
    return d.at[jnp.diag_indices_from(d)].set(0.0)


def rhoHT(X, c2, c1, c4, jitter=1e-6):
    """exp(-(d / c2)^c1) * exp(-|T_i - T_j| / c4); X columns: lat, lon, period.
    No nugget: only a tiny diagonal jitter for the Cholesky factorization."""
    C_h = jnp.exp(-jnp.power(getHaversineDistance(X) / c2, c1))
    C_T = jnp.exp(-jnp.abs(X[:, 2][:, None] - X[:, 2][None, :]) / c4)
    K = C_h * C_T
    return K.at[jnp.diag_indices(X.shape[0])].add(jitter)


# ---------------------------------------------------------------------------
# Observation bounds
# ---------------------------------------------------------------------------

def outcome_bounds(D, p, cap=BOUND_CAP):
    """(lo, hi) on the latent u for binary outcomes D given P(D=1) = p.

    D=1 -> (t, +cap), D=0 -> (-cap, t), with t = Phi^-1(1 - p) computed as
    -Phi^-1(p) so it stays accurate for p near 0."""
    D = np.asarray(D)
    p = np.asarray(p, dtype=float)
    t = -np_ndtri(p)
    lo = np.where(D == 1, t, -cap)
    hi = np.where(D == 1, cap, t)
    bad = ~(lo < hi)
    if np.any(bad):
        raise ValueError(f"{bad.sum()} empty interval(s): |t| >= cap={cap} "
                         f"(t range [{t.min():.2f}, {t.max():.2f}]); raise cap.")
    return lo, hi


# ---------------------------------------------------------------------------
# Truncated normal (also the GHK core used for the LPPD)
# ---------------------------------------------------------------------------

def _log1mexp(x):
    """log(1 - exp(x)) for x < 0, accurate across the whole range."""
    near0 = x > -_LN2
    x_near0 = jnp.where(near0, x, -1.0)
    x_far = jnp.where(near0, -1.0, x)
    return jnp.where(near0, jnp.log(-jnp.expm1(x_near0)), jnp.log1p(-jnp.exp(x_far)))


def _truncnorm_icdf(alpha, beta, v):
    """w = Phi^-1(Phi(alpha) + v (Phi(beta) - Phi(alpha))) and
    log(Phi(beta) - Phi(alpha)), for alpha < beta.

    Intervals centred above zero are evaluated through the mirrored
    interval (-beta, -alpha) with v -> 1 - v, which is the same mapping
    but keeps both Phi values in the accurate lower tail. The branch is
    chosen on the inputs, so there is only one numerical path and no NaN
    leaks into the gradient from an unused branch."""
    flip = alpha + beta > 0
    lo_ = jnp.where(flip, -beta, alpha)
    hi_ = jnp.where(flip, -alpha, beta)
    vv = jnp.where(flip, 1.0 - v, v)

    log_a = log_ndtr(lo_)
    log_b = log_ndtr(hi_)
    log_width = log_b + _log1mexp(log_a - log_b)

    log_x = jnp.logaddexp(log_a, jnp.log(vv) + log_width)
    c3_raw = ndtri(jnp.exp(jnp.maximum(log_x, _LOG_TINY)))
    return jnp.where(flip, -c3_raw, c3_raw), log_width


def _ghk_scan(L, lo, hi, v):
    """Map v in [0,1]^n to u = L w inside lo < u < hi.

    Returns (u, log_widths) with log_widths[i] = log(b_i - a_i)."""
    diagL = jnp.diag(L)

    def step(w, i):
        mu = L[i] @ w                   # w[j >= i] are still 0
        alpha = (lo[i] - mu) / diagL[i]
        beta = (hi[i] - mu) / diagL[i]
        w_i, log_width = _truncnorm_icdf(alpha, beta, v[i])
        return w.at[i].set(w_i), log_width

    w, log_widths = jax.lax.scan(step, jnp.zeros(L.shape[0], dtype=L.dtype), jnp.arange(L.shape[0]))
    return L @ w, log_widths


# ---------------------------------------------------------------------------
# Albert-Chib sampler
# ---------------------------------------------------------------------------
#
# State per chain: x = (logit(c2 / LH_MAX), logit(c1 / 2), log c4), the latents u (one
# vector per event) and, if AC_EVENT_MEAN, one latent mean mu_e per event:
#     u_e ~ MVN(mu_e 1, R_e(x)),   mu_e flat prior.
# One sweep =
#   1. AC_N_MH random-walk Metropolis steps on x given u, target
#      log p(x) + sum_e log p(u_e | x). Without event means p(u_e | x) is
#      MVN(u_e; 0, R_e); with them it is the restricted likelihood with mu_e
#      integrated out, so a shift common to the whole event never looks
#      like spatial correlation;
#   2. (AC_EVENT_MEAN) mu_e | u_e, x ~ N(1'R^-1 u / 1'R^-1 1, 1 / 1'R^-1 1);
#   3. one Gibbs pass over every u_i | u_-i, mu, x, D: a normal truncated to
#      (lo_i, hi_i), with mean/variance from the precision matrix R^-1.
# During warmup, after every AC_ADAPT_BLOCK sweeps, each chain's proposal is
# reset to scale_c * chol(2.38^2/3 * Cov(recent draws of all chains, each
# centred on its own mean)) (Haario-style), with the per-chain scale_c nudged
# towards AC_TARGET_ACC acceptance (Robbins-Monro on log scale). Sampling keeps
# the final proposal fixed, so the chain targets the exact posterior. Chains are vectorized with jax.vmap.

LH_MAX = 100.0   # km, upper end of the c2 prior support
# Prior log-densities written out explicitly (no numpyro distribution objects:
# older numpyro versions cache TruncatedNormal's normalizer lazily, which leaks
# a JAX tracer when first evaluated inside jit).
_LOG_Z_LH = float(np.log(norm.cdf((LH_MAX - 15.0) / 15.0) - norm.cdf((0.0 - 15.0) / 15.0)))


def _logpdf_lh(x):
    """TruncatedNormal(15, 15) on [0, LH_MAX] (x is always inside the support)."""
    return -0.5 * ((x - 15.0) / 15.0) ** 2 - jnp.log(15.0) - 0.5 * jnp.log(2.0 * jnp.pi) - _LOG_Z_LH


def _logpdf_beta22(y):
    """Beta(2, 2) on (0, 1)."""
    return jnp.log(6.0) + jnp.log(y) + jnp.log1p(-y)


def _logpdf_lognormal(x, mu, s):
    lx = jnp.log(x)
    return -lx - jnp.log(s) - 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * ((lx - mu) / s) ** 2


def _theta(x):
    """(c2, c1 / 2, c4) from the unconstrained x:
    c2 = LH_MAX * sigmoid(x0), c1 / 2 = sigmoid(x1), c4 = exp(x2)."""
    return LH_MAX * jax.nn.sigmoid(x[0]), jax.nn.sigmoid(x[1]), jnp.exp(x[2])


def _log_prior_unconstrained(x):
    c2, g2, c4 = _theta(x)
    s0 = jax.nn.sigmoid(x[0])
    log_jac = (jnp.log(LH_MAX) + jnp.log(s0) + jnp.log1p(-s0)
               + jnp.log(g2) + jnp.log1p(-g2) + x[2])
    return _logpdf_lh(c2) + _logpdf_beta22(g2) + _logpdf_lognormal(c4, -1.822, 0.457) + log_jac


def _cholesky_R(X, x):
    c2, g2, c4 = _theta(x)
    return jnp.linalg.cholesky(rhoHT(X, c2, 2.0 * g2, c4))


def _mvn_logpdf(L, u):
    z = solve_triangular(L, u, lower=True)
    return -0.5 * z @ z - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * u.shape[0] * jnp.log(2.0 * jnp.pi)


def _restricted_logpdf(L, u):
    """log p(u | R) with u ~ MVN(mu 1, R) and mu integrated out (flat prior)."""
    z = solve_triangular(L, u, lower=True)
    w = solve_triangular(L, jnp.ones_like(u), lower=True)
    a, b = w @ w, w @ z
    return (-0.5 * z @ z + 0.5 * b * b / a - jnp.sum(jnp.log(jnp.diag(L)))
            - 0.5 * jnp.log(a) - 0.5 * (u.shape[0] - 1) * jnp.log(2.0 * jnp.pi))


def _draw_mu(L, u, key):
    """mu | u, R ~ N(1'R^-1 u / 1'R^-1 1, 1 / 1'R^-1 1)."""
    z = solve_triangular(L, u, lower=True)
    w = solve_triangular(L, jnp.ones_like(u), lower=True)
    a = w @ w
    return (w @ z) / a + jax.random.normal(key, dtype=u.dtype) / jnp.sqrt(a)


def _gibbs_sweep(L, u, mu, lo, hi, key):
    """One sequential pass of u_i | u_-i ~ N(m_i, s_i^2) truncated to (lo_i, hi_i),
    for u ~ MVN(mu 1, R)."""
    Q = cho_solve((L, True), jnp.eye(L.shape[0]))
    qd = jnp.diag(Q)
    V = jax.random.uniform(key, u.shape, dtype=u.dtype)

    def body(i, u):
        m = mu - (Q[i] @ (u - mu) - qd[i] * (u[i] - mu)) / qd[i]
        sd = 1.0 / jnp.sqrt(qd[i])
        w, _ = _truncnorm_icdf((lo[i] - m) / sd, (hi[i] - m) / sd, V[i])
        return u.at[i].set(m + sd * w)

    return jax.lax.fori_loop(0, u.shape[0], body, u)


def _make_sweeps(Xs, los, his, n_mh, event_mean, parallel=False):
    loglik = _restricted_logpdf if event_mean else _mvn_logpdf

    def log_post(x, us):
        return _log_prior_unconstrained(x) + sum(
            loglik(_cholesky_R(X, x), u) for X, u in zip(Xs, us))

    def sweep(state, key):
        x, us, mus, lp, chol_prop, n_acc = state
        keys = jax.random.split(key, 2 * n_mh + 2 * len(us))
        for k in range(n_mh):
            xp = x + chol_prop @ jax.random.normal(keys[2 * k], (3,), dtype=x.dtype)
            lpp = log_post(xp, us)
            acc = jnp.log(jax.random.uniform(keys[2 * k + 1], dtype=x.dtype)) < lpp - lp
            x = jnp.where(acc, xp, x)
            lp = jnp.where(acc, lpp, lp)
            n_acc = n_acc + acc
        Ls = [_cholesky_R(X, x) for X in Xs]
        if event_mean:
            mus = jnp.stack([_draw_mu(L, u, keys[2 * n_mh + e]) for e, (L, u) in enumerate(zip(Ls, us))])
        us = [_gibbs_sweep(L, u, mus[e], lo, hi, keys[2 * n_mh + len(us) + e])
              for e, (L, u, lo, hi) in enumerate(zip(Ls, us, los, his))]
        return (x, us, mus, log_post(x, us), chol_prop, n_acc), jnp.concatenate([x, mus])

    if parallel:
        # one chain per CPU device; keys: (n_chains, block, 2)
        run_block = jax.pmap(lambda state, keys: jax.lax.scan(sweep, state, keys))
    else:
        @jax.jit
        def run_block(state, keys):        # keys: (block, n_chains, 2); block length is static
            return jax.lax.scan(jax.vmap(sweep), state, keys)

    @jax.jit
    def initial_log_post(x, us):
        return jax.vmap(log_post)(x, us)

    return run_block, initial_log_post


def modelHT_albert_chib(seed, X, eqids, lo, hi, n_chains, start_index=0, label=""):
    """Albert-Chib sampler for Model HT. Returns {name: (n_chains, n_kept)}
    for c2, c1_half, c1, c4 and, with AC_EVENT_MEAN, mu_<eqid>
    (after warmup, thinned by AC_THIN). start_index offsets the chains'
    overdispersed starting points (to split chains across processes).
    label is printed in the progress lines (e.g. "it=7")."""
    Xs  = [jnp.asarray(Xe[:, :3]) for Xe in X]   # lat, lon, period
    los = [jnp.asarray(l) for l in lo]
    his = [jnp.asarray(h) for h in hi]
    n_ev = len(Xs)
    parallel = AC_PARALLEL_CHAINS and n_chains > 1 and jax.local_device_count() >= n_chains
    if AC_PARALLEL_CHAINS and not parallel and n_chains > 1:
        print(f"    [{label}] only {jax.local_device_count()} CPU device(s): running the "
              f"{n_chains} chains vectorized on one core", flush=True)
    run_block, initial_log_post = _make_sweeps(Xs, los, his, AC_N_MH, AC_EVENT_MEAN, parallel)

    # Overdispersed starts (spread around chain index 1.5); latents start inside their bounds
    c = start_index + jnp.arange(n_chains, dtype=jnp.float64)[:, None]
    # start around c2 = 15 km, c1 = 1, c4 = 0.1 s, overdispersed across chains
    x = jnp.asarray([np.log(0.15 / 0.85), 0.0, np.log(0.1)]) + jnp.array([0.5, 0.3, 0.4]) * (c - 1.5)
    us = [jnp.tile(jnp.where(l > -BOUND_CAP + 1, l + 0.5, h - 0.5), (n_chains, 1)) for l, h in zip(los, his)]
    mus = jnp.zeros((n_chains, n_ev))
    chol_prop = jnp.tile(0.1 * jnp.eye(3), (n_chains, 1, 1))
    state = (x, us, mus, initial_log_post(x, us), chol_prop, jnp.zeros(n_chains, dtype=jnp.int64))

    key = jax.random.PRNGKey(seed)
    t_start = datetime.now()
    n_warm_blocks = AC_WARMUP // AC_ADAPT_BLOCK
    log_scale = np.zeros(n_chains)
    warm_hist, kept, acc_hist = [], [], []
    for b in range(n_warm_blocks + AC_SAMPLES // AC_BLOCK):
        warm = b < n_warm_blocks
        blk = AC_ADAPT_BLOCK if warm else AC_BLOCK
        key, sub = jax.random.split(key)
        if parallel:
            state, xs = run_block(state, jax.random.split(sub, blk * n_chains).reshape(n_chains, blk, -1))
            xs = np.asarray(xs).transpose(1, 0, 2)
        else:
            state, xs = run_block(state, jax.random.split(sub, blk * n_chains).reshape(blk, n_chains, -1))
            xs = np.asarray(xs)                # (block, n_chains, 3 + n_events)
        x, us, mus, lp, chol_prop, n_acc = state
        acc_rate = np.asarray(n_acc) / (blk * AC_N_MH)
        if warm:
            warm_hist.append(xs[..., :3])
            # Robbins-Monro on the log step size, gain decaying over warmup
            log_scale += (acc_rate - AC_TARGET_ACC) * 3.0 / np.sqrt(1.0 + b / 4.0)
            H = np.concatenate(warm_hist[-max(1, 1000 // AC_ADAPT_BLOCK):])  # last ~1000 sweeps
            # Proposal shape from all chains' recent draws (each centred on its own
            # mean), so one chain's short history cannot leave it badly tuned;
            # the step size stays per chain.
            pooled_cov = np.cov((H - H.mean(axis=0, keepdims=True)).reshape(-1, 3).T) + 1e-6 * np.eye(3)
            shape = np.linalg.cholesky((2.38 ** 2 / 3) * pooled_cov)
            chol_prop = jnp.asarray(np.stack([np.exp(log_scale[ch]) * shape for ch in range(n_chains)]))
        else:
            kept.append(xs[::AC_THIN])
            acc_hist.append(acc_rate)
        state = (x, us, mus, lp, chol_prop, jnp.zeros(n_chains, dtype=jnp.int64))

        # Progress line every 1000 warmup / 2000 sampling sweeps
        done_w = min(b + 1, n_warm_blocks) * AC_ADAPT_BLOCK
        done_s = max(0, b + 1 - n_warm_blocks) * AC_BLOCK
        if (warm and done_w % 1000 == 0) or (not warm and done_s % 2000 == 0):
            phase = f"warmup {done_w}/{AC_WARMUP}" if warm else f"sampling {done_s}/{AC_SAMPLES}"
            hours = (datetime.now() - t_start).total_seconds() / 3600
            print(f"    [{label}] {phase} sweeps, {hours:.1f} h elapsed, "
                  f"MH acceptance {np.round(acc_rate, 2)}", flush=True)
    print(f"    MH acceptance per chain during sampling {np.round(np.mean(acc_hist, axis=0), 3)}", flush=True)

    draws = np.concatenate(kept).transpose(1, 0, 2)  # (n_chains, n_kept, 3 + n_events)
    c2 = LH_MAX / (1.0 + np.exp(-draws[..., 0]))
    c1_half = 1.0 / (1.0 + np.exp(-draws[..., 1]))
    c4 = np.exp(draws[..., 2])
    out = dict(c2=c2, c1_half=c1_half, c1=2.0 * c1_half, c4=c4)
    if AC_EVENT_MEAN:
        out.update({f"mu_{eqid}": draws[..., 3 + e] for e, eqid in enumerate(eqids)})
    return out


# ---------------------------------------------------------------------------
# Log-likelihood for LPPD
# ---------------------------------------------------------------------------

def ghk_loglik(X, lo, hi, c2, c1, c4, key, n_mc=1000):
    """GHK Monte Carlo estimate of log P(lo < u < hi) for one event and one
    parameter draw. The GHK estimator of P is unbiased; its log is slightly
    biased low (Jensen), shrinking as n_mc grows."""
    return _ghk_loglik_jit(jnp.asarray(X)[:, :3], jnp.asarray(lo), jnp.asarray(hi),
                           c2, c1, c4, key, n_mc)


@jax.jit
def _ghk_mc(L, lo, hi, V):
    log_w = jax.vmap(lambda v: jnp.sum(_ghk_scan(L, lo, hi, v)[1]))(V)
    return logsumexp(log_w) - jnp.log(V.shape[0])


def _ghk_loglik_jit(X, lo, hi, c2, c1, c4, key, n_mc):
    L = jnp.linalg.cholesky(rhoHT(X, c2, c1, c4))
    V = jax.random.uniform(key, (n_mc, lo.shape[0]), dtype=L.dtype)
    return _ghk_mc(L, lo, hi, V)


def independent_loglik(lo, hi):
    """Exact log P(lo < u < hi) for independent unit-variance latents,
    i.e. sum_n log P(D_n | p_n) -- the no-correlation baseline for LPPD."""
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)
    return jnp.sum(_truncnorm_icdf(lo, hi, jnp.full_like(lo, 0.5))[1])


# ===========================================================================
# IN-PROCESS TIMEOUT (SIGALRM-BASED)
# ===========================================================================
# Each worker process runs MCMC in-process and uses signal.alarm to enforce
# a wall-clock budget.  When the alarm fires, Python receives a SIGALRM and
# raises TaskTimeout on the next bytecode boundary.  JAX/XLA C++ kernels may
# delay delivery until they return to Python, but in practice this fires
# reliably between MCMC iterations.

class TaskTimeout(Exception):
    """Raised when a task exceeds its wall-clock budget."""
    pass


def _alarm_handler(signum, frame):
    raise TaskTimeout("MCMC exceeded wall-clock budget")


# ===========================================================================
# CONVERGENCE CHECK
# ===========================================================================

def _check_convergence(res_chains, rhat_threshold=1.1, ess_threshold=200):
    """
    Check R-hat and ESS on per-chain samples.

    Parameters
    ----------
    res_chains : dict  {param_name: array of shape (n_chains, n_samples, ...)}
        Per-chain sampler output (latents are never returned).
    """
    skip_keys = {"c1"}   # derived from c1_half

    for key, chains in res_chains.items():
        if key in skip_keys:
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
    """Run the sampler for one model and return (flat_samples, chain_samples)."""
    print(f"  Running {m.__name__} | seed={seed} | chains={n_chains} | "
          f"{AC_WARMUP} warmup + {AC_SAMPLES} sweeps (thin {AC_THIN}) ...", flush=True)
    samples_chains = m(seed, n_chains=n_chains, **train_data)
    samples_flat = {k: v.reshape(-1) for k, v in samples_chains.items()}
    print(f"  Finished {m.__name__}", flush=True)
    return samples_flat, samples_chains


# ===========================================================================
# TIMED MCMC (IN-PROCESS, SIGALRM)
# ===========================================================================

def run_mcmc_with_timeout(m, seed, train_data, n_chains,
                          rhat_threshold, ess_threshold, timeout_s):
    """
    Run MCMC inside the current process with a hard wall-clock timeout.

    Returns
    -------
    ("ok",            samples_flat)   — converged
    ("not_converged", samples_flat)   — R-hat / ESS failed (samples kept for inspection)
    ("timeout",       None)           — hit wall-clock budget
    ("error",         repr(e))        — unexpected exception
    """
    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(timeout_s))
    try:
        samples_flat, samples_chains = run_model(m, seed, train_data, n_chains)
        converged = _check_convergence(samples_chains, rhat_threshold, ess_threshold)
        if converged:
            return ("ok", samples_flat)
        return ("not_converged", samples_flat)
    except TaskTimeout:
        return ("timeout", None)
    except Exception as e:
        return ("error", repr(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


# ===========================================================================
# MODEL INPUTS
# ===========================================================================

def event_inputs(df, eqids):
    """Per-event (X, lo, hi) lists from a per building-station table."""
    Xtot, lotot, hitot = [], [], []
    for eqid in eqids:
        group = df[df.eqid == eqid]
        lo, hi = outcome_bounds(group['D'].values, group['p'].values)
        Xtot.append(group[['latitude', 'longitude', 'period', 'Say']].values)
        lotot.append(lo)
        hitot.append(hi)
    return Xtot, lotot, hitot


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
    on completion (atomic write).  If a result file already exists the task
    is skipped (resume-from-disk).

    Returns
    -------
    (it, m_idx, samples_flat)  or  (it, m_idx, None) if failed/timeout.
    """
    import numpyro as _numpyro

    _numpyro.set_platform("cpu")  # must be called before any JAX computation

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
    data_seed = base_seed + (it+1) * 1000
    df = sample_buildings(df_all, n_bst, random_state=data_seed)

    # --- Prepare model inputs ---
    Xtot, lotot, hitot = event_inputs(df, eqids)
    train_data = dict(X=Xtot, eqids=eqids, lo=lotot, hi=hitot, label=f"it={it+1}")

    # --- Run MCMC with hard timeout ---
    status, payload = run_mcmc_with_timeout(
        m, seed, train_data, n_chains,
        rhat_threshold, ess_threshold, timeout_s,
    )

    if status == "ok":
        print(f"[it={it+1}, {m.__name__}] ✅ passed.", flush=True)
        result = (it, m_idx, payload)
    elif status == "not_converged":
        # Not used in any result; kept outside per_iter_dir (deleted on a clean finish)
        # so the convergence threshold can be re-checked later
        failed_file = Path(per_iter_dir).parent / "failed_samples" / f"{out_file.stem}.pkl"
        print(f"[it={it+1}, {m.__name__}] ⚠️  failed convergence — discarding "
              f"(samples kept in {failed_file}).", flush=True)
        result = (it, m_idx, None)
        try:
            failed_file.parent.mkdir(exist_ok=True)
            with open(failed_file, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"[it={it+1}, {m.__name__}] ⚠️ could not save failed samples: {e}", flush=True)
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


def _pin_worker(cpu_queue):
    """Pool initializer: pin this worker to its own CPUs of the job's
    allocation (one per chain when chains run in parallel), so stray JAX/BLAS
    threads cannot spill onto other workers' cores.
    No-op where CPU affinity is unavailable (e.g. macOS)."""
    cpus = cpu_queue.get()
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpus))
        except OSError:
            pass


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

    data = data[~np.isnan(data)]
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
            for var in dfp.columns:
                arr = np.asarray(dfp[var])
                if arr.size == 0:
                    continue
                estimate_histogram(arr, bins=30,
                                   title=f'{m.__name__} - Histogram for {var}',
                                   density=False, output_dir=plot_dir)
    except Exception as e:
        print(f"[refresh] ⚠️ histogram refresh failed: {e}", flush=True)

    return results_all, df_param_samples, valid_counts



# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # RUN CONFIG
    # -----------------------------------------------------------------------
    N              = 50   # bootstrap iterations

    n_bst          = 3     # distinct buildings sampled per station per iteration
    n_chains       = 4     # chains per model run (vectorized inside one worker)
    # One worker per core: each worker runs its chains vectorized on one core
    # Workers: one per core, or one per n_chains cores when the chains run in parallel
    cores_per_worker = n_chains if AC_PARALLEL_CHAINS else 1
    n_cpus = max(1, (int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count())) - 1) // cores_per_worker)

    rhat_threshold = 1.1
    ess_threshold  = 200

    eqids          = [137, 278]  # events to fit; None = every event in GEOMEAN_CSV
    eqids_test     = [127] # events used for the LPPD
    models         = [modelHT_albert_chib]
    seeds          = [0]   # one sampler seed per model, fixed across iterations
    base_seed      = 24    # controls bootstrap sampling per iteration

    # Hard wall-clock timeout per (iteration × model) task.
    timeout_s      = 46 * 3600   # 46 hours (server: ~35 h per iteration at n_bst=5)

    # Refresh aggregated .npz + histograms every this many completed tasks.
    refresh_every  = 3

    # LPPD: GHK orthant probability on an evenly spaced subset of draws.
    lppd_n_mc      = 500   # GHK Monte Carlo draws per posterior draw
    lppd_max_draws = 500   # posterior draws used
    lppd_n_alloc   = 20    # fixed test allocations, seeds base_seed + k, k = 0..lppd_n_alloc-1
    lppd_n_bst     = 3     # distinct buildings per station in each test allocation

    vars_per_model = [
        ["c2", "c1", "c4"],
    ]
    # -----------------------------------------------------------------------

    import functools
    print = functools.partial(print, flush=True)

    # --- Pooled geomean table (one row per building-station) ---
    pooled = pd.read_csv(GEOMEAN_CSV)
    if pooled.duplicated(["eqid", "RSN", "building"]).any():
        raise ValueError(f"duplicated (eqid, RSN, building) rows in {GEOMEAN_CSV.name} "
                         f"-- an event pasted twice?")
    print(f"Read {len(pooled)} rows from {GEOMEAN_CSV}")

    # --- (D, p) for this damage state, in memory ---
    with open(FRAGILITY_JSON) as f:
        fragility = json.load(f)[FRAGILITY_KEY]
    df_all = damage_table(pooled, DAMAGE_THRESHOLD, IM_TYPE, fragility)
    print(f"Damage state: EDP >= {DAMAGE_THRESHOLD}, fragility {FRAGILITY_JSON.name} "
          f"(mu={fragility['mu']:.4f}, beta={fragility['beta']:.4f})")
    print(df_all.groupby("eqid").agg(n=("D", "size"), stations=("RSN", "nunique"),
                                     D_rate=("D", "mean"), p_mean=("p", "mean")))

    if "--build-only" in sys.argv:
        sys.exit(0)

    if eqids is None:
        eqids = sorted(df_all["eqid"].unique().tolist())
    for e in list(eqids) + list(eqids_test):
        if e not in set(df_all["eqid"]):
            raise ValueError(f"eqid {e} not in the damage table")

    fingerprint = data_fingerprint(df_all, eqids, n_bst, base_seed, seeds,
                                   [m.__name__ for m in models],
                                   AC_WARMUP, AC_SAMPLES, AC_THIN, AC_N_MH, AC_EVENT_MEAN,
                                   SAMPLING_VERSION)

    output_dir   = HERE / f"results_damage-albertchib-HT_{DAMAGE_THRESHOLD}"
    per_iter_dir = output_dir / f"per_iteration-{fingerprint}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(per_iter_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir  = str(output_dir / f"plots_{timestamp}")
    npz_path  = HERE / f'results_damage_albertchib-HT_{DAMAGE_THRESHOLD}_{timestamp}.npz'

    print(f"Events      : {eqids}")
    print(f"Output dir  : {output_dir}")
    print(f"Per-iter dir: {per_iter_dir}")
    print(f"NPZ path    : {npz_path}")

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
        # Hand each worker its own CPU (the main process keeps the first one)
        ctx = mp.get_context("spawn")
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        cpu_queue = ctx.Queue()
        blocks = [allowed[1 + w * cores_per_worker: 1 + (w + 1) * cores_per_worker] for w in range(n_cpus)]
        for blk_cpus in blocks:
            cpu_queue.put(blk_cpus if len(blk_cpus) == cores_per_worker else None)
        print(f"Pinning workers to CPUs: {blocks if allowed else 'not supported here'}")
        with ctx.Pool(processes=n_cpus, initializer=_pin_worker, initargs=(cpu_queue,)) as pool:
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
        # Always do a final refresh so the latest state is on disk even if
        # we were interrupted partway through.
        results_all, df_param_samples, valid_counts = [], [], [0] * len(models)
        try:
            results_all, df_param_samples, valid_counts = aggregate_and_refresh(
                all_results, models, vars_per_model,
                npz_path, plot_dir,
                label="final",
            )
        except Exception as e:
            print(f"⚠️ final refresh failed: {e}", flush=True)

    # --- Log results to file (after parallel work is done so live logs were visible) ---
    log_file        = os.path.join(output_dir, f"run_log_{timestamp}.txt")
    original_stdout = sys.stdout
    sys.stdout      = open(log_file, "w")

    print(f"Logging to {log_file}")
    print(f"Data: {GEOMEAN_CSV.name}, EDP >= {DAMAGE_THRESHOLD}, {FRAGILITY_JSON.name}, "
          f"events {eqids}, LPPD events {eqids_test}")
    print(f"N={N}, n_bst={n_bst}, n_chains={n_chains}, n_cpus={n_cpus}")
    print(f"Albert-Chib: {AC_WARMUP} warmup + {AC_SAMPLES} sweeps per chain, thin {AC_THIN}, "
          f"{AC_N_MH} MH steps per sweep, event means {'on' if AC_EVENT_MEAN else 'off'}, "
          f"chains {'in parallel (' + str(jax.local_device_count()) + ' CPU devices)' if AC_PARALLEL_CHAINS else 'vectorized'}")
    print(f"Sampling: {SAMPLING_VERSION} (n_bst distinct buildings per station, independent across stations)")
    print("Priors:")
    print(f"  c2         ~ TruncatedNormal(15, 15) on [0, {LH_MAX:g}] km")
    print(f"  c1 / 2 ~ Beta(2, 2)                      (c1 in (0, 2))")
    print(f"  c4         ~ LogNormal(-1.822, 0.457)        (seconds)")
    print(f"R-hat threshold={rhat_threshold}, ESS threshold={ess_threshold}")
    print(f"Per-task timeout: {timeout_s/3600:.1f}h, refresh every {refresh_every} tasks\n")

    print("Convergence pass rates:")
    for m, count in zip(models, valid_counts):
        print(f"  {m.__name__}: {count}/{N} iterations passed ({100 * count / N:.1f}%)")

    # --- Summaries ---
    for m, dfp in zip(models, df_param_samples):
        print(f"\n--- {m.__name__} ---")
        if dfp.empty:
            print("no converged iterations")
        else:
            summarize_samples(dfp)

    # --- LPPD (binary outcomes, GHK orthant probability) ---
    # lppd_n_alloc fixed test allocations (lppd_n_bst distinct buildings per
    # station, seeds base_seed + k), the same for every model.
    df_test_all = df_all[df_all.eqid.isin(eqids_test)]
    test_sets = []
    for k in range(lppd_n_alloc):
        df_test = sample_buildings(df_test_all, lppd_n_bst, random_state=base_seed + k)
        test_sets.append((base_seed + k, *event_inputs(df_test, eqids_test)))

    from scipy.special import logsumexp as np_logsumexp

    for m_idx, (m, post) in enumerate(zip(models, results_all)):
        if not post:
            continue
        n_post   = len(post["c4"])
        draw_idx = np.unique(np.linspace(0, n_post - 1, min(lppd_max_draws, n_post)).astype(int))

        print(f"\nLPD per test allocation ({len(draw_idx)} draws x {lppd_n_mc} GHK samples, "
              f"{lppd_n_bst} buildings per station) — binary outcomes, "
              f"NOT comparable with fixed-transform LPDs:")
        print(f"  Model {m_idx} ({m.__name__})")
        print(f"  {'alloc':>5} {'seed':>6} {'LPD_model':>11} {'LPD_ind':>11} {'diff [nats]':>12}")

        rows = []
        for k, (seed_k, Xtest, lotest, hitest) in enumerate(test_sets):
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(base_seed), k),
                                    len(draw_idx) * len(eqids_test))
            keys = keys.reshape(len(draw_idx), len(eqids_test), -1)

            loglik_tot = np.array([
                sum(float(ghk_loglik(Xtest[e], lotest[e], hitest[e],
                                     post["c2"][s], post["c1"][s], post["c4"][s],
                                     keys[j, e], n_mc=lppd_n_mc))
                    for e in range(len(eqids_test)))
                for j, s in enumerate(draw_idx)
            ])

            LPD     = -np.log(len(draw_idx)) + np_logsumexp(loglik_tot)
            LPD_ind = sum(float(independent_loglik(lo, hi)) for lo, hi in zip(lotest, hitest))
            rows.append((k, seed_k, LPD, LPD_ind, LPD - LPD_ind))
            print(f"  {k:>5} {seed_k:>6} {LPD:>11.3f} {LPD_ind:>11.3f} {LPD - LPD_ind:>12.3f}")

            if k == 0:
                estimate_histogram(loglik_tot, bins=30,
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
    print(f"Log written to {log_file}")

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
