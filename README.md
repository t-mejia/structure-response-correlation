# Structure-to-structure response correlation in regional seismic risk assessment

Data and code of a two-part study on the correlation between the seismic responses of different buildings. Part I presents damage-state- and demand-based procedures to include response correlation in scenario-based regional risk assessment and applies them to the 1994 Northridge earthquake. Part II calibrates the correlation models with Bayesian inference, using the responses of 80 reinforced concrete frame buildings (class `tr_0018_dch`, generated with SimDesign) to the recorded ground motions of Chi-Chi 1999 and Chuetsu 2007, and tests them on Northridge 1994.

## Reference

- Mejia, T. and O'Reilly, G.J. Structure-to-structure response correlation in regional seismic risk assessment. Part I: methodology for damage-state- and demand-based assessment. (submitted)
- Mejia, T. and O'Reilly, G.J. Structure-to-structure response correlation in regional seismic risk assessment. Part II: calibration of correlation models via Bayesian inference. (submitted)

Centre for Training and Research on Reduction of Seismic Risk (ROSE Centre), Scuola Universitaria Superiore IUSS Pavia.

## Repository structure

```
data/
  portfolio/        SDOF properties of the 80 buildings: bilinear capacity curves, modal properties, storey heights
  cloud_analysis/   500 records (djura), intensity measures and peak storey drifts of the cloud analysis (scale factors 1, 2, 6)
  models/           building-class and building-specific demand-intensity models and fragility functions (DS1-DS4)
  events/           NLTH results, stations and EDP residuals for Northridge (EQID 127), Chi-Chi (137) and Chuetsu (278)
  records/          record names and time steps of each event (records not included)
  pooled/           pooled tables of the three events used for the Bayesian inference
  posteriors/       posterior samples of the correlation models (Part II Tables 5 and 6)
scripts/
  sdof_cloud_analysis/     NLTH of the SDOF oscillators for the cloud analysis; IM computation
  demand_and_fragility/    demand-intensity models and fragility functions (Part II Tables 2-3; Part I Fig. 4)
  event_nlth/              NLTH of every SDOF with the records of each event, both horizontal components
  residuals/               EDP residuals and building-station tables per event
  ts_regression/           deterministic fit of Model TS (Part II Fig. 4)
  bayesian_edp/            Bayesian inference for the EDP residuals, Models HT and HTS (NumPyro/NUTS)
  bayesian_damage/         Bayesian inference for the damage outcomes, Models HT and HTS (Albert-Chib Gibbs sampler, JAX)
  predictive_performance/  log predictive density on the testing event (Part II Table 7)
  scenario_assessment/     DS- and EDP-based scenario assessment of Northridge (Part I Fig. 5 and Table 5)
```

The scripts are listed in the order in which they were run; each one reads its inputs from `data/` and writes its outputs to its own folder.

## How to use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/bayesian_edp/bayesian_model_edp.py
```

Every script can be run from any folder. `sdof_cloud_analysis` and `event_nlth` need the ground-motion records; every other step runs from the processed data provided.

- **Posterior samples.** `edp_HT_HTS.npz` holds the EDP-residual models (`model_0_*` = HT, `model_1_*` = HTS); `ds<k>_HT.npz` and `ds<k>_HTS.npz` the damage-outcome models of DS1-DS3. Parameters are named as the coefficients of the papers: `c1`, `c2` [km], `c3`, `c4` [s], `c5` [g]. Auxiliary samples: `c1_half` = c1 / 2 (sampled with a Beta(2, 2) prior), `c2_pow_c1` = c2^c1 and `c3_raw` (Beta(2, 2) draw rescaled to c3).
- **Damage states.** The two scripts in `bayesian_damage` are set to DS1; the values for the other damage states are given next to `DAMAGE_THRESHOLD` and `AC_SAMPLES`.
- **Residuals.** The residuals in `data/pooled/eps_PSD.csv` are raw; `bayesian_edp` and `predictive_performance` divide them by the standard deviation of the residuals of the fitting events (Chi-Chi and Chuetsu).
- **Reproducibility.** The MCMC runs are stochastic and long (the damage-outcome fits were run on an HPC cluster, about a day per model and DS). Re-running them gives statistically equivalent but not bit-identical posteriors; the samples used in the papers are those in `data/posteriors/`. All fits use the same building-to-station sampling (distinct buildings at each station, drawn independently across stations). Building allocations use `base_seed = 24` (+1000 per repetition); the scenario assessment uses `BASE_SEED = 1000` (+1 per allocation).
- **Ground-motion records.** The acceleration histories are not redistributed. All records belong to the PEER NGA-West2 database (https://ngawest2.berkeley.edu) and can be downloaded with the RSNs listed in `data/cloud_analysis/` and `data/records/`. Convert them to plain-text files with the names listed and place them in `scripts/sdof_cloud_analysis/records-tomas/` (two columns, time [s] and acceleration [g], already multiplied by the scale factor in `djura-unconditional-records.csv`) and `scripts/event_nlth/rec_<Event>/` (e.g. `rec_Northridge/`; one column, acceleration [g]). The name lists are read from `data/`.
- **Building models.** The three-dimensional OpenSees models are not distributed; they can be generated with SimDesign (Ozsarac et al., 2025) for building class `tr_0018_dch` (Hasanoğlu et al., 2025). Only the equivalent SDOF properties used in the analyses are provided.

## Licence

This repository is licensed under the GNU General Public License v3.0 (see `LICENSE`).

## Citation

If you use this material, please cite the two papers above (see `CITATION.cff`).

## Contact

Tomas Mejia, IUSS Pavia — <tomas.mejiasaldarriaga@iusspavia.it>
