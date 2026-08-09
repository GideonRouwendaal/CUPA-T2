# CUPA-T2* — covariance-aware uncertainty propagation into T2\* regression

This folder holds the **downstream** half of the pipeline: it turns Monte-Carlo samples
from the uncertainty-aware reconstruction into voxel-wise T2\* estimates with predicted
uncertainty, and evaluates them.

It uses reconstruction checkpoints produced by
`projects/recon_t2star/` (see that folder's README for data preparation and the changes
made to PHIMO), and reads/writes data under `<repo>/../data/`. The flow is:

```
recon_t2star/            this folder
  trained recon net  ->  MC samples  ->  Cholesky / low-rank factors
                                              |
                              hyperparameter tuning (tissue-specific
                              -> tissue-combined -> RMS weight)
                                              |
                                     regression fitting
                                              |
                          evaluation + uncertainty analysis + ablations
```

Four models are compared throughout: **PUQ** (homoscedastic baseline), **Het**
(heteroscedastic), **Cholesky_concat** (CUPA — the
structured-covariance arms). All four take the same input: the MC **mean magnitude
concatenated with the MC standard deviation**, normalised by the first echo.

---

## Requirements

Activate the environment and make the repo importable. Every command below assumes you
are in the **`iml-dl` repo root**:

```bash
conda activate cupa_T2_star
cd /path/to/CUPA_T2/iml-dl
export PYTHONPATH="$PWD"
```

**The acceleration rate `R` is configurable at every stage** (R = 2, 3, 4 were used).
Substitute `R` in the commands below.

Color map in a folder in this directory called color_map
https://cig-utrecht.org/blog/2023/11/27/colormap-relaxometry.html

---

## 1. Generate the Monte-Carlo samples

Runs the trained reconstruction network with MC dropout and stores **100 samples per
slice**, plus ground truth, brain/tissue masks and zero-filled references.

```bash
python ./core/Main.py \
  --config_path ./projects/recon_t2star_regress_CUPA_T2_STAR/configs/generate_t2_star/config_Generate_t2star_Acc2_train.yaml
```

Repeat for each `{Acc2,Acc3,Acc4} x {train,val,test}` config in that folder.

`R` is set inside the config, in two places that must agree:

| field | meaning |
|---|---|
| `downstream_tasks.T2StarReconstruction-With_Unc_Acc_<R>` | task name; `<R>` selects the output folder `acc_rate_<R>/` |
| `experiment.weights` | the reconstruction checkpoint, `results/recon_t2star/Final/AccRate<R>/First2000/best_model.pt` |

Output lands in `../data/T2_param_data/With_Unc/acc_rate_<R>/<split>/`, with
`raw_mc_predictions/` holding the samples consumed by the next stage.

## 2. Build the Cholesky / low-rank factors

Factorises the per-voxel MC covariance across the 12 echoes into a Cholesky factor and
rank-`k` low-rank factors. Needs `gt/`, `gt_s0/`, `raw_mc_predictions/` and
`brain_masks/` from stage 1.

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/generate_sample_matrices.py \
  --acc_rate R --split train --ranks 6 12
```

Run for `train`, `val` and `test`. 
`--skip_lowrank` builds Cholesky only. Writes `cholesky_means/`, `cholesky_matrices/`,
`lowrank_means/rank_<k>/` and `lowrank_U_matrices/rank_<k>/` alongside the inputs.

## 3. Hyperparameter tuning

Three stages, each narrowing the previous. The selection metric throughout is the
**mean of WM and GM NRMSE** on the validation split (CSF is reported but not ranked).

**(a) Tissue-specific (isolation)** — sweeps one tissue at a time:

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/scan_tissue_results.py \
  --log_dir <isolation_run_logs> --top_k 20 --out_csv <out>/top_runs.csv
```

**(b) Tissue-combined** — sweeps the per-tissue schedules jointly:

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/scan_combined_results.py \
  --log_dir ./projects/recon_t2star_regress_CUPA_T2_STAR/tissue_combined_results/Cholesky/AccR \
  --top_k 20 \
  --out_csv ./projects/recon_t2star_regress_CUPA_T2_STAR/tissue_combined_results/Cholesky/AccR/top_runs.csv

python ./projects/recon_t2star_regress_CUPA_T2_STAR/generate_final_configs.py \
  --csv <…>/top_runs.csv --top_k 20 --acc_rates R
```

**(c) RMS-correlation weight** — takes the top tissue-combined picks and sweeps the
weight on the RMS-correlation loss:

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/generate_final_configs_RMS.py \
  --csv <…>/top_runs.csv --top_k 3 --acc_rates R
```

Both generators emit ready-to-submit config batches and `.sbatch` scripts. The winning
values are recorded in **[`HYPERPARAMETERS.md`](HYPERPARAMETERS.md)**, together with two
settings the trainer overrides at runtime that the YAML does not show.

## 4. Fit the regression models

One config:

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/run_regression.py \
  --mode train \
  --config ./projects/recon_t2star_regress_CUPA_T2_STAR/configs/final/AccR/<config>.yaml
```

Every config in a directory (this is how the sweeps and ablations were run):

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/run_group.py \
  --config_path ./projects/recon_t2star_regress_CUPA_T2_STAR/configs/final/AccR
```

`configs/final/` holds the exact winning configs per arm, recovered by matching the run
hashes recorded in the stored evaluations. Checkpoints are written under
`results/regression/acc_rate_<R>/…/<run_tag>_<hash>/`, where `<hash>` is a stable
digest of the defining hyperparameters.

## 5. Evaluate

**Select and evaluate the top runs** (parses training logs, picks the best baseline and
sweep runs, evaluates each on the test split):

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/evaluate_final_RMS.py \
  --log_dir ./projects/recon_t2star_regress_CUPA_T2_STAR/RMS_results/Cholesky/AccR \
  --evaluator_py ./projects/recon_t2star_regress_CUPA_T2_STAR/regression_evaluator.py \
  --out_root ./projects/recon_t2star_regress_CUPA_T2_STAR/evaluations/Final/AccR \
  --acc_rate R --device cuda:0
```

Add `--dry_run` to print the selected runs without evaluating. Each arm gets
`eval_summary.json`, `per_slice_tissue_metrics.csv`, `test_metrics.json` and an
`uncertainty_analysis/` folder (calibration curves, ENCE, AURC / risk-coverage,
proper scoring, uncertainty decomposition).

To evaluate explicitly named checkpoints instead of log-selected ones, use
`regression_evaluator_exp_paths.py` with the same `--evaluator_py` / `--out_root` flags.

**Evaluate a single config** (same evaluator, one arm, no log parsing):

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/run_regression.py \
  --mode test \
  --config ./projects/recon_t2star_regress_CUPA_T2_STAR/configs/final/AccR/<config>.yaml
```

The checkpoint is found from the hyperparameters in the config. When several runs share
one hyperparameter directory — retraining the same config under a different `run_tag`
does this — the choice is ambiguous and the evaluator stops and lists the candidates;
set `model_path_override` (and optionally `output_dir_override`) in `regression_params`
to pick one.

**Aggregate tables across acceleration rates** (`--eval_root` is the `--out_root` used
above, one level up from the per-rate folders):

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/summarize_eval_jsons.py \
  --eval_root ./projects/recon_t2star_regress_CUPA_T2_STAR/evaluations/Final \
  --acc_rates 2,3,4 --mean_only
```

**Statistical tests** (Wilcoxon signed-rank, CUPA vs each baseline):

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/regression_evaluator_statistical_tests_final.py \
  --log_dir ./projects/recon_t2star_regress_CUPA_T2_STAR/RMS_results/Cholesky/AccR \
  --puq_eval_dir    <evaluations>/AccR/BASELINE_PUQ \
  --hetero_eval_dir <evaluations>/AccR/BASELINE_Hetero \
  --sweep_eval_dir  <evaluations>/AccR/SWEEP_TOP1 \
  --out_csv <evaluations>/AccR/wilcoxon_accR.csv \
  --metric nrmse --metric ssim --agg mean \
  --tissue wm --tissue gm --tissue csf --tissue overall \
  --unit subject --min_subjects 1
```

**Rebuild the uncertainty analysis without a checkpoint.** Each evaluation folder stores
`uncertainty_analysis/pixel_level_cache.npz` with all per-voxel predictions, errors,
aleatoric/epistemic/total uncertainty and tissue labels, and every analysis routine is a
pure function of it:

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/regenerate_uncertainty_analysis.py \
  --eval_dir <evaluations>/AccR/SWEEP_TOP1
```

**Qualitative figure** comparing the four arms on one slice:

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/generate_qResults_plot.py \
  --config <test_config>.yaml \
  --puq_ckpt <…> --hetero_ckpt <…> --cupa_chol_ckpt <…> \
  --subject sub-22 --slice_num 16 --out comparison.png
```

## 6. Ablations

Three arms at R=4, all `cholesky_concat`, differing only in how the uncertainty enters:
`normal` (full), `no_covar` (variance only, no covariance structure) and `no_sampling`.

```bash
python ./projects/recon_t2star_regress_CUPA_T2_STAR/run_group.py \
  --config_path ./projects/recon_t2star_regress_CUPA_T2_STAR/configs/Ablations/configs
```

---

## Layout

```
configs/
  generate_t2_star/   stage-1 configs, one per (R, split)
  final/Acc{2,3,4}/         the exact winning config per arm
  Ablations/                ablation configs + sbatch scripts
  regress/                  example train/test configs
used_sbatches/              the SLURM scripts these results were produced with
color_map/                  T2* colormaps (navia, lipari)
evaluations/, RMS_results/, tissue_combined_results/
                            stored outputs of the runs described above
```