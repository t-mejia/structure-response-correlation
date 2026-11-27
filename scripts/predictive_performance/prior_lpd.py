"""
LPD on the Northridge test event (EQID 127) exactly as in bayesian_model-exp_n3.py (server, Pooled_geomean/275564927),
but with the parameters drawn from the PRIORS of modelHT and modelHTS instead of the posterior (prior predictive LPD).

Unchanged (imported from bayesian_model-exp_n3.py): data preparation (eps / std of the fitting-event eps), sample_buildings (per-station-v2),
the 20 fixed test allocations (3 buildings per station, seeds base_seed + k = 24 ... 43), the models and their priors,
LPD = -ln n + logsumexp(sum of log-likelihoods) and the independence reference LPD_ind.
Prior draws: numpyro Predictive (no posterior samples), PRNGKey(seed of the model in the script: 0 HT, 23 HTS).
The log-likelihood is evaluated in chunks of draws (memory); LPD is also reported for the first n/4 and n/2 draws to
show the Monte Carlo convergence of the prior average.

--priors script : the priors of the script (default).
--priors generic: non-informative priors, flat or log-uniform over wide ranges (not from earlier fits), same kernels:
    c2 ~ Uniform(0.01, 200) km,  c1_half ~ Uniform(0, 1) (c1 in (0, 2)),
    c4 ~ LogUniform(0.01, 10) s, c5 ~ LogUniform(0.01, 10) (Say units),  c3_raw ~ Uniform(0, 1) (c3 = 0.5 + 0.499 c3_raw, as the script)
    (portfolio: periods 0.47-0.75 s, Say 0.15-0.50, so the length scales span ~2 decades below and above the data)

Usage: <phd python> prior_lpd.py [--draws 20000] [--chunk 250] [--priors script|generic]
Output: lpd_prior[_generic]_<model>.csv, summary printed and in prior_lpd[_generic]_log.txt
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIT_EQIDS = [137, 278]   # fitting events (Chi-Chi, Chuetsu); the eps scale is the std of their residuals
spec = importlib.util.spec_from_file_location("bm", HERE.parent / "bayesian_edp" / "bayesian_model_edp.py")
bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)

# settings of the script (bayesian_model-exp_n3.py, CONFIG)
BASE_SEED, EQIDS_TEST, LPPD_N_ALLOC, LPPD_N_BST = 24, [127], 20, 3
MODELS, SEEDS = [bm.modelHT, bm.modelHTS], [0, 23]


def modelHT_generic(X, eqids, z):
    import jax.numpy as jnp, numpyro, numpyro.distributions as dist
    c2 = numpyro.sample("c2", dist.Uniform(0.01, 200.0))
    c1_half = numpyro.sample("c1_half", dist.Uniform(0.0, 1.0))
    c1 = numpyro.deterministic("c1", 2.0 * c1_half)
    c2_pow_c1 = numpyro.deterministic("c2_pow_c1", jnp.power(c2, c1))
    c4 = numpyro.sample("c4", dist.LogUniform(0.01, 10.0))
    for i, eqid in enumerate(eqids):
        numpyro.sample(f"z_{eqid}", dist.MultivariateNormal(0, bm.rhoHT(X[i], c2_pow_c1, c1, c4)), obs=z[i])


def modelHTS_generic(X, eqids, z):
    import jax.numpy as jnp, numpyro, numpyro.distributions as dist
    c2 = numpyro.sample("c2", dist.Uniform(0.01, 200.0))
    c1_half = numpyro.sample("c1_half", dist.Uniform(0.0, 1.0))
    c3_raw = numpyro.sample("c3_raw", dist.Uniform(0.0, 1.0))
    c1 = numpyro.deterministic("c1", 2.0 * c1_half)
    c2_pow_c1 = numpyro.deterministic("c2_pow_c1", jnp.power(c2, c1))
    c3 = numpyro.deterministic("c3", 0.500 + (0.999 - 0.500) * c3_raw)
    c4 = numpyro.sample("c4", dist.LogUniform(0.01, 10.0))
    c5 = numpyro.sample("c5", dist.LogUniform(0.01, 10.0))
    for i, eqid in enumerate(eqids):
        numpyro.sample(f"z_{eqid}", dist.MultivariateNormal(0, bm.rhoHTS(X[i], c2_pow_c1, c1, c4, c5, c3)), obs=z[i])


def load_df():
    """Same data preparation as the script's main."""
    stations = pd.read_csv(DATA / "pooled" / "stations.csv")
    eps = pd.read_csv(DATA / "pooled" / "eps_PSD.csv")
    buildings = pd.read_csv(DATA / "portfolio" / "buildings.txt", names=["building"])
    build_modal = json.loads((DATA / "portfolio" / "modal_results.json").read_text())
    build_bilinear = json.loads((DATA / "portfolio" / "bilinear_results.json").read_text())
    buildings["period"] = [build_modal[b]["MX"]["period"] for b in buildings["building"]]
    buildings["Say"] = [build_bilinear[b]["X"]["force"][1] for b in buildings["building"]]
    eps = eps.merge(stations[["RSN", "EQID"]], on="RSN", how="inner")
    eps_long = eps.melt(id_vars=["EQID", "RSN", "latitude", "longitude"], var_name="building", value_name="eps")
    eps_long["building"] = eps_long["building"].str.replace("^eps_", "", regex=True)
    df = eps_long.merge(buildings, on="building", how="left")
    df = df[["EQID", "RSN", "building", "eps", "latitude", "longitude", "period", "Say"]].rename(columns={"EQID": "eqid"})
    df = df.dropna()
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    # Scale = std of the residuals used for the calibration (fitting events FIT_EQIDS, as `eqids` of
    # bayesian_edp/bayesian_model_edp.py), applied to the test event as well.
    eps_scale = df.loc[df.eqid.isin(FIT_EQIDS), "eps"].std()
    df["eps"] = df["eps"] / eps_scale
    return df.reset_index(drop=True), eps_scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--priors", choices=("script", "generic"), default="script")
    a = ap.parse_args()
    models = MODELS if a.priors == "script" else [modelHT_generic, modelHTS_generic]
    tag = "" if a.priors == "script" else "_generic"
    import jax
    import numpyro
    from numpyro.infer import Predictive, log_likelihood
    from scipy.special import logsumexp
    numpyro.set_platform("cpu")

    df_all, eps_scale = load_df()
    df_test_all = df_all[df_all.eqid.isin(EQIDS_TEST)]
    test_sets = []
    for k in range(LPPD_N_ALLOC):
        df_test = bm.sample_buildings(df_test_all, LPPD_N_BST, random_state=BASE_SEED + k)
        X, z = [], []
        for eqid in EQIDS_TEST:
            g = df_test[df_test.eqid == eqid]
            X.append(g[["latitude", "longitude", "period", "Say"]].values); z.append(g["eps"].values)
        test_sets.append((BASE_SEED + k, dict(X=X, eqids=EQIDS_TEST, z=z)))
    log = [f"eps scale {eps_scale:.4f}; Northridge stations {df_test_all.RSN.nunique()}, "
           f"{len(test_sets[0][1]['z'][0])} residuals per test allocation; {a.draws} prior draws per model; priors: {a.priors}"]
    print(log[0], flush=True)

    for m, seed in zip(models, SEEDS):
        prior = Predictive(m, num_samples=a.draws)(jax.random.PRNGKey(seed), **test_sets[0][1])
        prior = {k: v for k, v in prior.items() if not k.startswith("z_")}
        rows = []
        for k, (seed_k, data) in enumerate(test_sets):
            ll = []
            for i in range(0, a.draws, a.chunk):
                part = {kk: v[i:i + a.chunk] for kk, v in prior.items()}
                # numpyro log_likelihood with parallel=False (the default) returns wrong per-draw values for these MVN models; parallel=True, in chunks
                LL = log_likelihood(m, part, parallel=True, **data)
                ll.append(np.sum([np.asarray(v) for v in LL.values()], axis=0))
            ll = np.concatenate(ll)
            ll = ll[np.isfinite(ll)]
            n = len(ll)
            lpd = -np.log(n) + logsumexp(ll)
            z = np.hstack(data["z"])
            lpd_ind = -0.5 * (len(z) * np.log(2 * np.pi) + np.sum(z ** 2))
            part_lpd = {f"LPD_model_{f}": -np.log(n // f) + logsumexp(ll[: n // f]) for f in (4, 2)}
            rows.append(dict(alloc=k, seed=seed_k, n_draws=n, LPD_model=lpd, LPD_ind=lpd_ind, diff_nats=lpd - lpd_ind,
                             **part_lpd, max_weight=float(np.exp(ll.max() - logsumexp(ll)))))
            print(f"  {m.__name__} alloc {k:2d}: LPD {lpd:10.3f}  ind {lpd_ind:10.3f}  diff {lpd - lpd_ind:8.3f}  "
                  f"(n/4 {part_lpd['LPD_model_4'] - lpd_ind:8.3f}, n/2 {part_lpd['LPD_model_2'] - lpd_ind:8.3f}; "
                  f"largest draw weight {rows[-1]['max_weight']:.2f})", flush=True)
        t = pd.DataFrame(rows); t.to_csv(HERE / f"lpd_prior{tag}_{m.__name__.replace('_generic', '')}.csv", index=False)
        sd = t.diff_nats.std(ddof=1)
        line = (f"{m.__name__} (prior): mean LPD {t.LPD_model.mean():.3f}, LPD_ind {t.LPD_ind.mean():.3f}, "
                f"diff {t.diff_nats.mean():.3f} nats (sd {sd:.3f}, se {sd / np.sqrt(len(t)):.3f}); "
                f"with n/4 draws {(t.LPD_model_4 - t.LPD_ind).mean():.3f}, n/2 {(t.LPD_model_2 - t.LPD_ind).mean():.3f}")
        print(line, flush=True); log.append(line)
    (HERE / f"prior_lpd{tag}_log.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
