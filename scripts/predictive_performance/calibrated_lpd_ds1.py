"""
Posterior (calibrated) LPD of the DS1 Albert-Chib models on held-out test events, exactly as the LPPD block of
Bayesian_Damage_Model{HT,HTS}-AlbertChib.py: 500 posterior draws evenly spaced over all draws of the .npz
(np.linspace(0, n - 1, 500)), GHK log P(lo < u < hi) with 500 samples per draw (ghk_loglik of the script), keys
split(fold_in(PRNGKey(24), k), 500) for allocation k, LPD = -ln 500 + logsumexp; LPD_ind = independent_loglik.
Allocations: sample_buildings of the script (per-station-v2), 3 buildings per station, seeds 24 + k, k = 0..19.
Data: DS1 = geomean EDP >= 0.5, p from fragility_0.5.json["AvgSa2"]; test event 127 (Northridge) from geomean_pooled.csv.
Posteriors (fitting set Chi-Chi 137 + Chuetsu 278 only, see the server run logs):
    HTS: results_damage_albertchib-HTS_0.5_20261007_211920.npz   HT: results_damage_albertchib-HT_0.5_20261007_230944.npz
Usage: <phd python> calibrated_lpd_ds1.py [--models HTS] [--events 127] [--allocs 20]
Output: calibrated_<model>_ds1_ev<event>_local.csv (alloc, seed, LPD_model, LPD_ind, diff_nats)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
import prior_lpd_ds as Q          # loads both scripts (HT, HTS) unchanged
import jax
from scipy.special import logsumexp

NPZ = {"HTS": "ds1_HTS.npz", "HT": "ds1_HT.npz"}
EVENT_FILES = {127: "geomean_pooled.csv"}   # Northridge (testing event, Part II Table 7)
BASE_SEED, N_BST, N_MC, MAX_DRAWS = 24, 3, 500, 500


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="HTS"); ap.add_argument("--events", default="127")
    ap.add_argument("--allocs", type=int, default=20)
    a = ap.parse_args()
    frag = json.loads((DATA / "models" / "downstream" / "fragility_0.5.json").read_text())["AvgSa2"]
    for model in a.models.split(","):
        mod = Q.HTS if model == "HTS" else Q.HT
        post = {k.replace("model_0_", ""): v for k, v in np.load(DATA / "posteriors" / NPZ[model]).items()}
        n_post = len(post["c4"])
        draw_idx = np.unique(np.linspace(0, n_post - 1, min(MAX_DRAWS, n_post)).astype(int))
        for ev in (int(e) for e in a.events.split(",")):
            df = mod.damage_table(pd.read_csv(DATA / "pooled" / EVENT_FILES[ev]), 0.5, "AvgSa2", frag)
            t = df[df.eqid == ev]
            rows = []
            for k in range(a.allocs):
                t0 = time.time()
                X, lo, hi = mod.event_inputs(mod.sample_buildings(t, N_BST, random_state=BASE_SEED + k), [ev])
                keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(BASE_SEED), k), len(draw_idx)).reshape(len(draw_idx), 1, -1)
                if model == "HTS":
                    ll = [float(mod.ghk_loglik(X[0], lo[0], hi[0], post["c2"][s], post["c1"][s], post["c4"][s], post["c5"][s],
                                               post["c3"][s], keys[j, 0], n_mc=N_MC)) for j, s in enumerate(draw_idx)]
                else:
                    ll = [float(mod.ghk_loglik(X[0], lo[0], hi[0], post["c2"][s], post["c1"][s], post["c4"][s],
                                               keys[j, 0], n_mc=N_MC)) for j, s in enumerate(draw_idx)]
                lpd = -np.log(len(draw_idx)) + logsumexp(ll)
                ind = float(mod.independent_loglik(lo[0], hi[0]))
                rows.append(dict(alloc=k, seed=BASE_SEED + k, LPD_model=lpd, LPD_ind=ind, diff_nats=lpd - ind))
                print(f"{model} event {ev} alloc {k:2d}: LPD {lpd:8.3f} ind {ind:8.3f} diff {lpd - ind:7.3f} ({time.time() - t0:.0f} s)", flush=True)
            r = pd.DataFrame(rows); r.to_csv(HERE / f"calibrated_{model}_ds1_ev{ev}_local.csv", index=False)
            print(f"{model} event {ev}: mean diff {r.diff_nats.mean():.3f} (sd {r.diff_nats.std(ddof=1):.3f}, "
                  f"se {r.diff_nats.std(ddof=1) / np.sqrt(len(r)):.3f})", flush=True)


if __name__ == "__main__":
    main()
