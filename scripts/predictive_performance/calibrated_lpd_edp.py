"""
Posterior (calibrated) LPD of the EDP Model HTS on held-out test events, as the LPPD block of bayesian_model-exp_n3.py
in the conditions of the original calibration (7 Oct 2026):
    posterior  results_all_20261007_165733.npz, model_1 (ETS = HTS kernel; 49 converged iterations, 147000 draws),
               fitted on Chi-Chi (137) and Chuetsu (278) only. Kernel exp(-d^c1 / c2^c1) * (c3 exp(-dT / c4)
               + (1 - c3) exp(-dSay / c5)), nugget 1e-4.
    eps scale  std of the residuals of the fitting events 137 and 278 (prior_lpd.load_df)
    sampling   per-station-v2 (the sampler of that time: n distinct buildings per station, independent across stations)
LPD = -ln n + logsumexp over ALL posterior draws of the Gaussian log-likelihood (numpyro log_likelihood of modelHTS);
LPD_ind = -1/2 (n ln 2 pi + sum z^2). 20 allocations, 3 buildings per station, seeds 24 + k.
--draws N uses N evenly spaced posterior draws (np.linspace over all draws) instead of all 147000 (cost).
Usage: <phd python> calibrated_lpd_edp.py [--events 127] [--allocs 20] [--chunk 200] [--draws 0]
Output: calibrated_HTS_edp_ev<event>_local.csv (alloc, seed, LPD_model, LPD_ind, diff_nats)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import prior_lpd as P

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
NPZ = DATA / "posteriors" / "edp_HT_HTS.npz"   # = results_all_20261007_165733.npz


def load_edp():
    """Residual table of prior_lpd.load_df (divided by the std of the fitting-event residuals)."""
    return P.load_df()


def sample_v2(df, n_bst, random_state):
    """per-station-v2 sampler (bayesian_model-exp_n3.py of 7 Oct; = sample_buildings of the DS scripts)."""
    rng = np.random.default_rng(random_state)
    return (df.assign(_k=rng.random(len(df))).sort_values("_k", kind="stable")
              .groupby(["eqid", "RSN"], sort=False).head(n_bst).drop(columns="_k")
              .sort_values(["eqid", "RSN", "building"], kind="stable").reset_index(drop=True))


def test_sets(df, ev, allocs=20):
    t = df[df.eqid == ev]
    out = []
    for k in range(allocs):
        g = sample_v2(t, 3, 24 + k)
        out.append(dict(X=[g[["latitude", "longitude", "period", "Say"]].values], eqids=[ev], z=[g["eps"].values]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", default="127"); ap.add_argument("--allocs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--draws", type=int, default=0, help="0 = all draws; else this many evenly spaced draws")
    a = ap.parse_args()
    import numpyro
    from numpyro.infer import log_likelihood
    from scipy.special import logsumexp
    numpyro.set_platform("cpu")
    d = np.load(NPZ)
    post = {k: d[f"model_1_{k}"] for k in ("c2", "c1_half", "c3_raw", "c4", "c5")}
    if a.draws:
        idx = np.unique(np.linspace(0, len(post["c2"]) - 1, a.draws).astype(int))
        post = {k: v[idx] for k, v in post.items()}
    n = len(post["c2"])
    df, scale = load_edp()
    print(f"posterior {NPZ.name} model_1 (HTS), {n} draws; eps scale {scale:.4f}", flush=True)
    for ev in (int(e) for e in a.events.split(",")):
        rows = []
        for k, t in enumerate(test_sets(df, ev, a.allocs)):
            t0 = time.time()
            # numpyro log_likelihood with parallel=False (the default) returns wrong per-draw values for these MVN models; parallel=True, in chunks
            ll = np.concatenate([np.sum([np.asarray(v) for v in log_likelihood(P.bm.modelHTS, {kk: v[i:i + a.chunk] for kk, v in post.items()}, parallel=True, **t).values()], axis=0)
                                 for i in range(0, n, a.chunk)])
            lpd = -np.log(n) + logsumexp(ll)
            z = t["z"][0]; ind = -0.5 * (len(z) * np.log(2 * np.pi) + np.sum(z ** 2))
            rows.append(dict(alloc=k, seed=24 + k, LPD_model=lpd, LPD_ind=ind, diff_nats=lpd - ind))
            print(f"HTS EDP event {ev} alloc {k:2d}: LPD {lpd:9.3f} ind {ind:9.3f} diff {lpd - ind:8.3f} ({time.time() - t0:.0f} s)", flush=True)
        r = pd.DataFrame(rows); r.to_csv(HERE / f"calibrated_HTS_edp_ev{ev}_local.csv", index=False)
        print(f"HTS EDP event {ev}: mean diff {r.diff_nats.mean():.3f} (sd {r.diff_nats.std(ddof=1):.3f}, "
              f"se {r.diff_nats.std(ddof=1) / np.sqrt(len(r)):.3f})", flush=True)


if __name__ == "__main__":
    main()
