"""
Prior-predictive LPD for DS1 (geomean PSD >= 0.5 %) on the Northridge test event (EQID 127), as the LPPD block of
Bayesian_Damage_ModelHT-AlbertChib.py / Bayesian_Damage_ModelHTS-AlbertChib.py (server, Latent-2D), but with the
parameters drawn from priors instead of the Albert-Chib posterior.

Unchanged (imported from the two scripts): damage_table (D = 1 iff EDP >= 0.5, p = Phi((ln AvgSa2 - mu) / beta) of
fragility_0.5.json["AvgSa2"]), sample_buildings (per-station-v2), event_inputs / outcome_bounds, the kernels rhoHT and
rhoHTS, the GHK orthant probability (500 GHK samples per draw), independent_loglik, the 20 test allocations
(3 buildings per station, seeds 24 ... 43) and the GHK keys (fold_in(PRNGKey(24), k), one per draw).
LPD = -ln n + logsumexp over the draws; LPD_ind = exact independent log P(D | p).

Priors (--priors):
  script : c2 ~ TruncatedNormal(15, 15) on [0, 100] km, c1 / 2 ~ Beta(2, 2), c4 ~ LogNormal(-1.822, 0.457) s,
           c5 ~ LogNormal(-1.848, 0.414) g, c3 = 0.5 + 0.5 Beta(2, 2)
  generic: c2 ~ Uniform(0.01, 200) km, c1 / 2 ~ Uniform(0, 1), c4, c5 ~ LogUniform(0.01, 10), c3 = 0.5 + 0.5 U(0, 1)

Per draw (as prior_lpd_northridge/per_draw_gain.py): each draw's own gain over independence, mean over the allocations,
saved with the parameters; the mixture LPD (logsumexp over draws) is computed from the same draws.
--event: test event (DS1 data has 127 Northridge, 137 Chi-Chi, 278 Chuetsu).

Usage: <phd python> prior_lpd_ds.py --priors script|generic [--draws 2000] [--batch 8] [--allocs 20] [--event 127]
Output: lpd_ds1_prior_<priors>_<model>_ev<event>.csv (mixture LPD per allocation), per_draw_ds1_<priors>_<model>_ev<event>.csv,
prior_lpd_ds1_<priors>_ev<event>_log.txt (previous mixture-only version: mixture_only_v1/)
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load(name, f):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "bayesian_damage" / f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


HT = load("dsHT", "Bayesian_Damage_ModelHT-AlbertChib.py")
HTS = load("dsHTS", "Bayesian_Damage_ModelHTS-AlbertChib.py")
import jax                      # x64 already enabled by the scripts
import jax.numpy as jnp
from scipy.special import logsumexp
from scipy.stats import truncnorm

THRESHOLD, IM_TYPE, FRAG_KEY = 0.5, "AvgSa2", "AvgSa2"
BASE_SEED, EQIDS_TEST, N_ALLOC, N_BST, N_MC = 24, [127], 20, 3, 500


def prior_draws(model, priors, n, seed):
    r = np.random.default_rng(seed)
    if priors == "script":
        d = dict(c2=truncnorm.rvs((0 - 15) / 15, (100 - 15) / 15, loc=15, scale=15, size=n, random_state=r),
                 g2=r.beta(2, 2, n), c4=r.lognormal(-1.822, 0.457, n))
        if model == "HTS":
            d.update(c5=r.lognormal(-1.848, 0.414, n), c3=0.5 + 0.5 * r.beta(2, 2, n))
    else:
        lu = lambda: np.exp(r.uniform(np.log(0.01), np.log(10.0), n))
        d = dict(c2=r.uniform(0.01, 200.0, n), g2=r.uniform(0, 1, n), c4=lu())
        if model == "HTS":
            d.update(c5=lu(), c3=0.5 + 0.5 * r.uniform(0, 1, n))
    d["c1"] = 2.0 * d.pop("g2")
    return d


def make_batched(model):
    """log P(lo < u < hi) for a batch of parameter draws (vmap of the scripts' GHK)."""
    mod = HT if model == "HT" else HTS

    def one(X, lo, hi, th, key):
        R = mod.rhoHT(X, th[0], th[1], th[2]) if model == "HT" else mod.rhoHTS(X, th[0], th[1], th[2], th[3], th[4])
        L = jnp.linalg.cholesky(R)
        V = jax.random.uniform(key, (N_MC, lo.shape[0]), dtype=L.dtype)
        log_w = jax.vmap(lambda v: jnp.sum(mod._ghk_scan(L, lo, hi, v)[1]))(V)
        return jax.scipy.special.logsumexp(log_w) - jnp.log(N_MC)

    return jax.jit(jax.vmap(one, in_axes=(None, None, None, 0, 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priors", choices=("script", "generic"), required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--allocs", type=int, default=N_ALLOC)
    ap.add_argument("--models", default="HT,HTS")
    ap.add_argument("--event", type=int, default=127)
    a = ap.parse_args()
    EQIDS_TEST[:] = [a.event]

    pooled = pd.read_csv(DATA / "pooled" / "geomean_pooled.csv")
    frag = json.loads((DATA / "models" / "downstream" / "fragility_0.5.json").read_text())[FRAG_KEY]
    df_all = HT.damage_table(pooled, THRESHOLD, IM_TYPE, frag)
    df_test_all = df_all[df_all.eqid.isin(EQIDS_TEST)]
    test_sets = []
    for k in range(a.allocs):
        df_test = HT.sample_buildings(df_test_all, N_BST, random_state=BASE_SEED + k)
        test_sets.append((BASE_SEED + k, *HT.event_inputs(df_test, EQIDS_TEST)))
    log = [f"DS1: EDP >= {THRESHOLD}, fragility AvgSa2 mu {frag['mu']:.4f} beta {frag['beta']:.4f}; "
           f"event {a.event}: {df_test_all.RSN.nunique()} stations, {len(test_sets[0][2][0])} outcomes per allocation "
           f"(D rate {df_test_all.D.mean():.2f}); priors {a.priors}, {a.draws} draws x {N_MC} GHK samples"]
    print(log[0], flush=True)

    for model, seed in (("HT", 0), ("HTS", 23)):
        if model not in a.models.split(","):
            continue
        th = prior_draws(model, a.priors, a.draws, seed)
        cols = ["c2", "c1", "c4"] + (["c5", "c3"] if model == "HTS" else [])
        TH = np.column_stack([th[c] for c in cols])
        f = make_batched(model)
        rows = []; G = np.full((a.draws, len(test_sets)), np.nan)
        for k, (seed_k, Xt, lot, hit) in enumerate(test_sets):
            t0 = time.time()
            X, lo, hi = jnp.asarray(Xt[0][:, :4]), jnp.asarray(lot[0]), jnp.asarray(hit[0])
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(BASE_SEED), k), a.draws)
            ll = np.concatenate([np.asarray(f(X, lo, hi, jnp.asarray(TH[i:i + a.batch]), keys[i:i + a.batch]))
                                 for i in range(0, a.draws, a.batch)])
            lpd_ind = float(HT.independent_loglik(lot[0], hit[0]))
            G[:, k] = ll - lpd_ind
            ll = ll[np.isfinite(ll)]; n = len(ll)
            lpd = -np.log(n) + logsumexp(ll)
            part = {f"LPD_model_{q}": -np.log(n // q) + logsumexp(ll[: n // q]) for q in (4, 2)}
            rows.append(dict(alloc=k, seed=seed_k, n_draws=n, LPD_model=lpd, LPD_ind=lpd_ind, diff_nats=lpd - lpd_ind,
                             **part, max_weight=float(np.exp(ll.max() - logsumexp(ll)))))
            print(f"  {model} alloc {k:2d}: LPD {lpd:9.3f} ind {lpd_ind:9.3f} diff {lpd - lpd_ind:8.3f} "
                  f"(n/4 {part['LPD_model_4'] - lpd_ind:8.3f}, n/2 {part['LPD_model_2'] - lpd_ind:8.3f}; "
                  f"largest weight {rows[-1]['max_weight']:.2f}) {time.time() - t0:.0f} s", flush=True)
        t = pd.DataFrame(rows); t.to_csv(HERE / f"lpd_ds1_prior_{a.priors}_{model}_ev{a.event}.csv", index=False)
        g = G.mean(axis=1); mix = t.diff_nats.mean()
        pdf = pd.DataFrame(TH, columns=cols); pdf["gain_mean"] = g
        for k in range(G.shape[1]):
            pdf[f"gain_alloc{k}"] = G[:, k]
        pdf.to_csv(HERE / f"per_draw_ds1_{a.priors}_{model}_ev{a.event}.csv", index=False, float_format="%.6g")
        fin = np.isfinite(g); q = np.percentile(g[fin], [5, 25, 50, 75, 95])
        pline = (f"  per draw: best {g[fin].max():.2f}, quantiles 5/25/50/75/95 % = {' / '.join(f'{x:.2f}' for x in q)} | "
                 f"share > 0: {np.mean(g > 0):.1%}, > 50 % of mixture: {np.mean(g > 0.5 * mix):.1%}, "
                 f"> 0 in every allocation: {np.mean((G > 0).all(axis=1)):.1%}")
        sd = t.diff_nats.std(ddof=1) if len(t) > 1 else np.nan
        line = (f"{model} ({a.priors} priors): mean LPD {t.LPD_model.mean():.3f}, LPD_ind {t.LPD_ind.mean():.3f}, "
                f"diff {t.diff_nats.mean():.3f} nats (sd {sd:.3f}, se {sd / np.sqrt(len(t)):.3f}); "
                f"n/4 {(t.LPD_model_4 - t.LPD_ind).mean():.3f}, n/2 {(t.LPD_model_2 - t.LPD_ind).mean():.3f}")
        print(line + "\n" + pline, flush=True); log += [line, pline]
    (HERE / f"prior_lpd_ds1_{a.priors}_ev{a.event}_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
