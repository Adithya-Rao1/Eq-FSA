# Baselines hand-off (end of Wave 3, 2026-11-04)

Nothing in this file has been run on Burgundy. Everything below the "ready" line is what you can run now; everything above it
is why no baseline *numbers* can come out yet.

## Honest status

| Piece | State |
|---|---|
| Provenance (seeding, env manifest), config (units, schema) | unit-tested locally (400 of 402 tests pass; 1 expected failure, 1 CUDA skip) |
| Dataset loaders (QM9, rMD17, MD22, ANI-1x, SPICE split) | unit-tested on synthetic data; **rMD17 npz key names unverified** (test fails on purpose, see CLAUDE.md issue 3) |
| Checkpoint / resume (RNG, epoch, loader order, final marker) | unit-tested; verified locally with tiny models |
| MD analysis (RDF/MSD/VACF/drift, stability report) | unit-tested on synthetic data; LAMMPS/OpenMM runners never executed |
| Slurm campaign generator + templates | renders and `bash -n` clean; submit scripts never run |
| Containers (.def) | **never built**; this Mac cannot build Singularity images |
| **Tier 0/1 adapters** | build models and loop over epochs, but **read the code and found no neighbor-list/graph construction, no per-library batch schema and no collate_fn**; datasets yield `{x, atomic_numbers, energy, forces}`, which SchNetPack/NequIP/Allegro/MACE do not accept. Expected to fail on the first batch. Never executed (libraries not installed locally). |
| **No evaluation anywhere** | no val/test loop, no MAE/RMSE, `metrics.jsonl` is never written |
| Tier 2/3 adapters | stubs: build a model and return; no training, zero-shot or fine-tune logic |
| OMol25 | loader + OOD split builder exist; not wired into `adapters/dataset_loading.py` |
| Solvation free energy (weeks 6-7) | no code exists |
| MD vs classical FFs (week 8) | runners exist, but nothing exports a deployed/TorchScript potential for them to use |

## What is ready to run now (in order)

All paths assume `$HOME/scratch` on Burgundy. `<EID>` is your CityU id.

### 1. Build the images (Linux machine with sudo + Singularity 3.7; not this Mac)
```
git clone <repo-url> Eq-FSA && cd Eq-FSA && git checkout new/baselines && git submodule update --init --recursive
bash src/baselines/containers/build_and_verify.sh            # all four; or append an image name, e.g. mlip-mpnn
```
Writes `src/baselines/containers/<name>.sif` and `<name>.sif.sha256`, and import-checks each image (now including h5py, rdkit, pytest).
Not measured; budget an hour or more per image (multi-GB NGC base plus GPU libraries).

### 2. Upload
```
sftp <EID>@burgundy.hpc.cityu.edu.hk
sftp> cd scratch
sftp> put src/baselines/containers/mlip-legacy.sif
sftp> put src/baselines/containers/mlip-legacy.sif.sha256        # repeat for mlip-mpnn, mlip-transformer, mlip-foundation
```

### 3. Stage the repo and data on Burgundy
```
cd ~/scratch && git clone <repo-url> Eq-FSA && cd Eq-FSA && git checkout new/baselines && git submodule update --init --recursive
mkdir -p ~/scratch/data/{qm9,rmd17/npz_data,rmd17/splits,md22,spice}
# qm9/            *.xyz files from the GDB-9 release (dsgdb9nsd_*.xyz)
# rmd17/npz_data/ rmd17_<molecule>.npz        rmd17/splits/ index_train_01.csv .. index_test_05.csv
# md22/           md22_<molecule>.npz         spice/        SPICE-2.0.1.hdf5   (rename with --spice-file if different)
```
The repo must be at `~/scratch/Eq-FSA` (or pass `--repo-dir`); jobs bind-mount it read-only at `/repo`.

### 4. Run the test suite on the server (inside an image, so torch and the real libraries are present)
```
cd ~/scratch/Eq-FSA/src
singularity exec --nv ~/scratch/mlip-mpnn.sif python -m pytest baselines/tests -q --no-header -p no:cacheprovider
```
Locally it takes about 1 minute on CPU (402 tests). Expect exactly one failure, `test_load_rmd17_reads_the_published_rmd17_npz_key_names`
(issue 3). Run it in each image; some tests need libraries only one image carries.

### 5. Generate and submit the smoke tests (15 min walltime each, `gpu_a40-test`)
```
cd ~/scratch/Eq-FSA/src
python3 -m baselines.slurm.campaign --phase phase0_smoke --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
bash $HOME/scratch/campaign/submit_phase0_smoke.sh
squeue -u $USER                                   # then read the *.out files next to the submit script
```
These only import each adapter and check `torch.cuda.is_available()`; they do not test a forward/backward pass.

### 6. Timing pilot (only worth running after the adapters can produce a batch; see "Not ready")
```
python3 -m baselines.slurm.campaign --phase phase1_calibration --pilot --epochs 1 --scratch-root $HOME/scratch --out-dir $HOME/scratch/campaign
bash $HOME/scratch/campaign/submit_phase1_calibration.sh
sacct -j <jobid> --format=JobID,JobName%40,Elapsed,MaxRSS,State
```
Pilot = seed 0, one dataset instance per model, 1 epoch, `gpu_a40-test` (2 h default; pass `--time` up to 12:00:00 for SPICE).

## Time estimates (what can and cannot be said)

| Step | Estimate | Basis |
|---|---|---|
| Local pytest | ~1 min | measured: 56 s, CPU, 402 tests |
| Image build | >= 1 h each | unmeasured guess |
| Smoke jobs | <= 15 min each plus queue | template walltime |
| Pilot, 11 jobs | <= 2 h each | partition default; real value comes from `sacct` |
| Full calibration (249 jobs) and SPICE (15 jobs, 6 more blocked) | **cannot be estimated** | no per-epoch timing exists |

After the pilot: GPU-hours for a job family ~ (pilot epoch seconds x planned epochs) / 3600 per run, times seeds. The campaign tool
prints job counts only; multiplying 249 jobs by the 5-day cap gives 29,880 GPU-hours, which is a ceiling no real run will approach.
Epoch counts and learning rates are not chosen anywhere (`--epochs` defaults to 1 on purpose).

## Not ready: what blocks real results (needs your decision)

1. **Per-library batch pipelines.** Each of SchNetPack, NequIP/Allegro, MACE needs its own graph batching (neighbor lists, library
   key names, collate). Without it the first batch fails.
2. **An evaluation step.** Val/test energy and force MAE/RMSE in meV, written to `metrics.jsonl`; nothing produces a baseline number
   until this exists.
3. **Tier 2/3 training, zero-shot and fine-tune logic**, plus OMol25 wiring.
4. **Solvation free-energy code** and a **potential export** for the MD runners.

Also from the artifact's week-1 checklist, still on you: apply for OMol25/UMA gated Hugging Face access, ask CSC about fakeroot or
Sylabs remote build, and request a larger scratch quota before any OMol25 download.
