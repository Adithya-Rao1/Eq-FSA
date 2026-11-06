# Baselines hand-off (native-trainer pipeline, 2026-11-05)

Nothing in this file has been run on Burgundy, and no image has been built. Everything below was written and unit-tested on a Mac (CPU only).

## What is being compared

EqFSA against **MACE, NequIP, Allegro, SchNet, PaiNN, EquiformerV2** (each trained by its library's own trainer) and **GAFF2 / OpenFF**
(through OpenMM), on **QM9** (energy only), **rMD17**, **MD22**, **ANI-1x** (fixed seeded subset), **SPICE**. The classical force fields run on
rMD17, MD22 and SPICE only (they need a bond graph). Dropped: Equiformer v1, ANI-2x, AIMNet2, all tier-3 foundation models, OMol25,
solvation free energy, MD-stability work.

Pipeline: `export_splits` writes extended XYZ (eV, Angstrom) + `split.json` once per dataset instance -> a wrapper trains with the native
trainer -> `evaluation/run_eval.py` scores val and test with one shared ASE evaluator -> `metrics.json` per run.

## Honest status

| Piece | State |
|---|---|
| Export, shared evaluator, 4 wrappers (MACE, NequIP/Allegro, SchNetPack, fairchem), classical FF, campaign/launch/templates | unit-tested locally; the wrappers also ran **tiny real CPU trainings** and calculator checks (3 conda/venv environments, not the images) |
| Images (`containers/*.def`) | **never built**. The fairchem PyG-extension source build (`mlip-transformer`) and the conda solve (`mlip-classical`) are the riskiest steps |
| Hyperparameters | **placeholders** (EquiformerV2 12 layers / lmax 4, NequIP / Allegro / SchNetPack sizes and learning rates). `--epochs` defaults to 1 on purpose |
| ANI-1x subset sizes | **placeholder** 100,000 / 10,000 / 10,000 (`campaign.ANI1X_CAPS`), seeded (`SUBSET_SEED = 0`), same for every model. ANI-1x filename `ani1x-release.h5` is an assumption (`--ani1x-file`) |
| NequIP energy-only (QM9) scaling | `global_rescale_scale: dataset_total_energy_std`; effect on accuracy untested |
| rMD17 | keys verified 2026-11-06 on the real malonaldehyde file (`coords, nuclear_charges, energies, forces`); loader fixed, and molecule `salicylic_acid` renamed `salicylic` to match the file. Other 9 molecules not yet inspected |
| Validation sets | rMD17 and SPICE ship no validation set but every trainer needs one, so the campaign uses split names `rmd17_split_0_val50` (950 train / 50 val carved with a seeded sample from the fixed 1000 train frames; test = the fixed 1000) and `molecule_95_5_val5` (5% of the training molecules held out; test molecules identical to the plain 95/5 split). The published splits (`rmd17_split_0`, `molecule_95_5`) still load with an empty val |
| Classical FF on large MD22 molecules | graph is inferred from geometry (RDKit); expect the 370-atom nanotube to be **skipped** (limit 200 atoms), and AM1-BCC on stachyose / AT-AT-CG-CG / buckyball-catcher is untested. Skips are counted in `metrics.json` (`n_molecules_skipped`, `skipped_reasons`) |
| CPU partition | `stingy` is an **assumption** for the CPU-only export / classical jobs (`--cpu-partition`); its default walltime here is 5 h and may be too short for the SPICE export and the SPICE classical run (`--time`) |
| Retired, unscheduled code | `adapters/tier0_legacy.py`, `tier1_mpnn.py`, `tier2_transformer.py` (stub), `tier3_foundation.py`, `md/`, `containers/mlip-foundation.def`: still in the tree with their tests, **not used by the campaign**. Safe to delete if you agree |

## Single machine without Slurm (what is actually in use, 2026-11-06)

The server in use is one Ubuntu box with an A10 and Apptainer, no Slurm. `slurm/run_local.py` runs the same jobs the campaign renders (the
commands come from `campaign.job_commands`), sequentially, through `apptainer exec` (`--nv` for smoke / train jobs). It skips jobs whose
`split.json` / `metrics.json` exists, so stopping and re-running is safe; it keeps going after a failure and prints a summary
(`--fail-fast` to stop). Logs, generated scripts and `results.jsonl` (status, seconds per job) go to `$HOME/scratch/campaign/local/`.
Run it with the host `python3` (it needs no torch), not inside an image. Add `--dry-run` to only write the scripts.

```
cd ~/scratch/Eq-FSA/src
python3 -m baselines.slurm.run_local --phase phase0_smoke --scratch-root $HOME/scratch
python3 -m baselines.slurm.run_local --phase phase1_export --datasets rmd17 --scratch-root $HOME/scratch
python3 -m baselines.slurm.run_local --phase phase1_export phase2_train_calibration --models schnet --datasets rmd17 --only rmd17:aspirin --pilot --epochs 1 --scratch-root $HOME/scratch
```

## Run order (all paths assume `$HOME/scratch`; `<EID>` is your CityU id)

### 1. Build the images (Linux machine with sudo + Singularity 3.7; not this Mac)
```
git clone <repo-url> Eq-FSA && cd Eq-FSA && git checkout new/baselines
bash src/baselines/containers/build_and_verify.sh            # all four, or append one name, e.g. mlip-classical
```
Builds `mlip-legacy`, `mlip-mpnn`, `mlip-transformer`, `mlip-classical`, writes `<name>.sif` + `<name>.sif.sha256`, and import-checks each
(the classical check also runs an AM1-BCC charge calculation). **Time: unmeasured, budget >= 1 h per image.** Compare
`conda list` in `mlip-classical` with the versions listed at the top of its `.def` (verified together only on osx-arm64).

### 2. Upload
```
sftp <EID>@burgundy.hpc.cityu.edu.hk
sftp> cd scratch
sftp> put src/baselines/containers/mlip-legacy.sif          # and .sif.sha256; repeat for mlip-mpnn, mlip-transformer, mlip-classical
```

### 3. Stage the repo and data
```
cd ~/scratch && git clone <repo-url> Eq-FSA && cd Eq-FSA && git checkout new/baselines
mkdir -p ~/scratch/data/{qm9,rmd17/npz_data,rmd17/splits,md22,spice,ani1x}
# qm9/ dsgdb9nsd_*.xyz   rmd17/npz_data/ rmd17_<mol>.npz + rmd17/splits/ index_*.csv   md22/ md22_<mol>.npz
# spice/SPICE-2.0.1.hdf5   ani1x/ani1x-release.h5   (rename with --spice-file / --ani1x-file if different)
```
The repo must be at `~/scratch/Eq-FSA` (or pass `--repo-dir`); jobs bind it read-only at `/repo` and the whole scratch tree at the same path.

### 4. Tests on the server (inside the images; a few tens of seconds each, CPU)
```
cd ~/scratch/Eq-FSA/src && export SINGULARITYENV_PYTHONPATH=$HOME/scratch/Eq-FSA/src
B="--bind $HOME/scratch:$HOME/scratch"
singularity exec $B ~/scratch/mlip-mpnn.sif       python -m pytest baselines/tests/test_export.py baselines/tests/test_evaluate.py baselines/tests/test_run_eval.py baselines/tests/test_mace_wrapper.py baselines/tests/test_nequip_wrapper.py baselines/tests/test_campaign.py baselines/tests/test_slurm_launch.py baselines/tests/test_units.py -q -p no:cacheprovider
singularity exec $B ~/scratch/mlip-legacy.sif     python -m pytest baselines/tests/test_schnetpack_wrapper.py -q -p no:cacheprovider
singularity exec $B ~/scratch/mlip-transformer.sif python -m pytest baselines/tests/test_fairchem_wrapper.py -q -p no:cacheprovider
singularity exec $B ~/scratch/mlip-classical.sif  python -m pytest baselines/tests/test_classical.py -q -p no:cacheprovider
```
Measured on a Mac: mpnn set ~2 min, schnetpack ~30 s, fairchem ~30 s, classical ~15 s. Do **not** run the whole `baselines/tests`
directory in one go: locally, `test_seeding.py::test_seed_worker_with_real_multiprocess_dataloader_changes_with_loader_seed` (a 2-worker DataLoader test of the retired loops) never returns on macOS; every other file finishes in seconds. Cause unconfirmed (maybe macOS spawn vs Linux fork), so tell me if it also hangs on the server.
`test_dataset_loaders.py` should now pass fully (the rMD17 key issue is fixed).

### 5. Smoke test (15 min walltime each, `gpu_a40-test`)
```
cd ~/scratch/Eq-FSA/src
python3 -m baselines.slurm.campaign --phase phase0_smoke --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
bash $HOME/scratch/campaign/submit_phase0_smoke.sh      # 4 jobs, one per image
```
Imports the wrappers and checks CUDA. It does not train anything.

### 6. Export every dataset instance (CPU, once; shared by all models)
```
python3 -m baselines.slurm.campaign --phase phase1_export --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
bash $HOME/scratch/campaign/submit_phase1_export.sh     # 20 jobs
```
Wait for these to finish: a training job stops with `no export at ...` if its export is missing. The rMD17 keys are resolved (CLAUDE.md issue 3).
Exports written by an older revision lack the `smiles` tag that the classical SPICE run needs; re-export if in doubt (delete the export dir).
Time: unmeasured.

### 7. Timing pilot (one seed, one dataset instance per model and family, `gpu_a40-test`)
```
python3 -m baselines.slurm.campaign --phase phase2_train_calibration --pilot --epochs 1 --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
bash $HOME/scratch/campaign/submit_phase2_train_calibration.sh      # 18 jobs
sacct -j <jobid> --format=JobID,JobName%40,Elapsed,MaxRSS,State
```
Look at `runs/phase2_train_calibration/*/metrics.json` and the `.out` files. This is the first time anything trains on a GPU.

### 8. Full runs (after choosing epochs / hyperparameters from the pilot)
```
python3 -m baselines.slurm.campaign --phase phase2_train_calibration --epochs <N> --chain 3 --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
python3 -m baselines.slurm.campaign --phase phase2_train_spice      --epochs <N> --chain 3 --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
python3 -m baselines.slurm.campaign --phase phase2_train_ani1x      --epochs <N> --chain 3 --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
python3 -m baselines.slurm.campaign --phase phase3_classical --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
bash $HOME/scratch/campaign/submit_<phase>.sh
```
Job counts: calibration **324** (6 models x 18 dataset instances x 3 seeds), SPICE 18, ANI-1x 18, classical 36. Trim with `--models`,
`--datasets`, `--seeds`; `campaign.TRAIN_MATRIX` is the single place to drop model/dataset pairs. `--chain 3` queues dependent copies for the
5-day cap: a finished run is a no-op and an interrupted one resumes from the trainer's own checkpoint, evaluation runs once.
**GPU-hours cannot be estimated** until the pilot gives per-epoch timings.

## Not done / decisions for you

- Hyperparameters, epochs, ANI-1x subset sizes (placeholders above).
- Whether to delete the retired adapters, `md/`, the `mlip-foundation` def, and the unused torchani / aimnetcentral pins in `mlip-legacy.def`.
- Reporting: nothing aggregates the `metrics.json` files into tables yet.
