# Runbook: first run on a Slurm cluster (two GPU nodes, shared `$HOME/scratch`)

Assumptions: only this repo exists on the cluster; `$HOME/scratch` is visible from every node; Slurm; `apptainer` or `singularity` on the compute nodes.
Partitions on Burgundy (from the collaborator's `sinfo`, 2026-11-10): use `GPU_PART=gpu_v100s` (5-day limit, one node idle) and `CPU_PART=tiny` (5-day limit, two idle nodes;
`batch` was fully allocated). `gpu_a100` is the fallback. Pass `--time` with `tiny` (the renderer only knows default walltimes for the partitions in `campaign.py`).
`--fakeroot` image builds are allowed there. Never put data, images or outputs outside `$HOME/scratch`.

What has and has not been verified (2026-11-11):
- Verified on a single Ubuntu GPU box (A10 22 GiB, apptainer, no Slurm), via `slurm/run_local.py`: image builds, unit tests, exports for rMD17, QM9, MD22, ANI-1x and a capped SPICE
  pilot, and one-epoch pilot trainings of all six models on one instance per dataset family. 25 of the 30 pilot jobs had finished at the time of writing (see the OOM note below).
  GAFF2 / OpenFF classical runs on rMD17, MD22 (buckyball-catcher and the 370-atom nanotube are recorded skips) and a 100-frame SPICE sample (boron molecules are skipped).
- Never run on a real Slurm cluster: the sbatch templates, the submit scripts and `--fakeroot` image builds (allowed on Burgundy, but not yet tried). The built-in
  partition defaults (`gpu_a40-test`, `stingy`) do not exist or are reserved there, so always pass `--gpu-partition` and `--cpu-partition`. Need the `.err` files (see "If something fails").
- Known limit: MACE and EquiformerV2 ran out of GPU memory on the 22 GiB A10 for MD22 Ac-Ala3-NHMe and SPICE at the default batch sizes (MACE 32, EquiformerV2 8).
  Lower with `--batch-size` through `--extra-adapter-args` if the same happens on your GPUs. The rMD17 aspirin test in step 5 is not affected.
- Hyperparameters, epochs and per-library loss weights are placeholders (MACE's energy did not learn beyond atom counts in one ANI-1x epoch), so pilot accuracies are not benchmark results.

## 1. Stage the repo
```bash
mkdir -p $HOME/scratch && cd $HOME/scratch
git clone <repo-url> Eq-FSA && cd Eq-FSA && git checkout new/baselines
```
The repo must live at `$HOME/scratch/Eq-FSA`. Jobs bind it read-only at `/repo`.

## 2. Build the four images (about an hour each or more; unmeasured on this cluster)
Run it on a compute node, not the login node, e.g. `srun -p CPU_PART -c 8 --mem 32G -t 06:00:00 --pty bash`, then:
```bash
cd $HOME/scratch/Eq-FSA
bash src/baselines/containers/build_and_verify.sh 2>&1 | tee $HOME/scratch/build.log
```
Images, digests (`*.sif.sha256`) and the build cache go to `$HOME/scratch`. The script uses `sudo` if it is passwordless and otherwise `--fakeroot`. If
neither works on your cluster, build on a machine that has root and copy the four `.sif` and four `.sif.sha256` files into `$HOME/scratch`.
Expect `OK` for each of the four images. `cuda_available= False` is normal on a node without a GPU; the smoke jobs in step 5 test CUDA.

## 3. Get the rMD17 data
```bash
bash $HOME/scratch/Eq-FSA/src/baselines/datasets/fetch_rmd17.sh
```
It tries the two known hosts, checks size and sha256, and extracts to `$HOME/scratch/data/rmd17/{npz_data,splits}`. If both hosts refuse automated downloads from the first server (HTTP 403 / 404), download `rmd17.tar.bz2` in a browser from
<https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038>, place it at `$HOME/scratch/data/rmd17.tar.bz2`, and run the script again.

Do not upload it through a web file manager as one 1 GB file (this truncated it before). Instead, split it into 50 MB pieces and check each.

## 4. Unit tests inside the images (CPU, a few minutes in total)
```bash
cd $HOME/scratch/Eq-FSA/src
CT=$(command -v apptainer || command -v singularity)
B="--bind $HOME/scratch:$HOME/scratch --bind $HOME/scratch/Eq-FSA:/repo:ro"
for spec in "mlip-mpnn:test_export.py test_evaluate.py test_run_eval.py test_mace_wrapper.py test_nequip_wrapper.py test_campaign.py test_slurm_launch.py test_check_outputs.py test_run_local.py test_units.py" \
            "mlip-legacy:test_schnetpack_wrapper.py test_dataset_loaders.py test_split_gaps.py" "mlip-transformer:test_fairchem_wrapper.py" "mlip-classical:test_classical.py"; do
  img=${spec%%:*}; files=${spec#*:}; paths=""; for f in $files; do paths="$paths baselines/tests/$f"; done
  echo "== $img"; $CT exec $B --env PYTHONPATH=/repo/src $HOME/scratch/$img.sif python -m pytest $paths -q -p no:cacheprovider 2>&1 | tail -3
done
```
Every test should pass. Do not run the whole `baselines/tests` directory at once (one retired test hangs on some platforms). Any failure or skip is worth reporting.

## 5. Smoke, export, one-epoch pilot (the two-node test)
```bash
cd $HOME/scratch/Eq-FSA/src
S=$HOME/scratch; O=$S/campaign
python3 -m baselines.slurm.campaign --phase phase0_smoke --scratch-root $S --out-dir $O --gpu-partition GPU_PART
bash $O/submit_phase0_smoke.sh                 # 4 jobs; wait until `squeue -u $USER` is empty
python3 -m baselines.slurm.campaign --phase phase1_export --datasets rmd17 --scratch-root $S --out-dir $O --cpu-partition CPU_PART --time 01:00:00
bash $O/submit_phase1_export.sh                # 10 jobs; wait until squeue is empty (training jobs refuse to start without their export)
python3 -m baselines.slurm.campaign --phase phase2_train_calibration --datasets rmd17 --pilot --epochs 1 --scratch-root $S --out-dir $O --gpu-partition GPU_PART --time 02:00:00
bash $O/submit_phase2_train_calibration.sh     # 6 jobs (one per model, rMD17 aspirin); Slurm decides placement; with two GPU nodes free, two should run at once
```
Job scripts and the submit scripts are only written by `campaign`, and nothing is submitted until you run a `submit_*.sh`. Slurm `.out/.err` files go to`$O/logs/`. On the first server the six one-epoch jobs took 3 to 5 minutes each including start-up (EquiformerV2 the longest).

Before submitting, check the node sizes (the templates ask for 90G of memory and 8 CPUs per GPU job, and these figures were never checked against the V100S nodes):
```bash
scontrol show node gpu-v100s-06 | grep -E "RealMemory|CPUTot|Gres"
```
If `RealMemory` is below about 92000 (MB), edit `GPU_RESOURCES` in `slurm/campaign.py` before rendering. Report the three values either way.

## 6. Check the results
```bash
cd $HOME/scratch/Eq-FSA/src
python3 -m baselines.slurm.check_outputs --phase phase1_export phase2_train_calibration --datasets rmd17 --only rmd17:aspirin --pilot --scratch-root $HOME/scratch
```
It lists every expected export / metrics file, flags missing or non-finite ones, and scans `$O/logs` for `Traceback`, `CUDA out of memory`, `slurmstepd: error`.
Exit code 0 means all complete and no error text. Library `FutureWarning`s about `torch.load` are expected and do not count as errors.
To confirm nothing was written outside scratch: `du -sh $HOME/scratch/*` and `ls -la $HOME $HOME/.cache` (compare with the listing from before step 2).

## If something fails
Save: the failing job's `$O/logs/*.err` and `.out` (last 60 lines), `sinfo -s`, `$(command -v apptainer || command -v singularity) --version`, and the output of `check_outputs`. 
Likely first problems: partition names or missing `--time` (the renderer refuses unknown partitions without one), `--fakeroot`
builds not permitted, GPU not visible inside the container (`--nv`), memory limits (the templates ask for 90G on training jobs; edit `GPU_RESOURCES` in `slurm/campaign.py` if the nodes are smaller).

## Notes on the models
EquiformerV2 is trained with its published direct-force head, as in the paper. Its defaults here (energy weight 20, EMA 0.99) were chosen from rMD17 aspirin runs. The OC20 values (energy weight 2, EMA 0.999) left the energy at the mean baseline and the forces at 546 meV/Å after 5 epochs.
All other hyperparameters and epoch counts are not tested. Next, `--epochs` defaults to 1 for this test run. Per-model loss weights differ between libraries (SchNetPack 0.01 / 0.99, NequIP 1 / 1, MACE 1 / 100, EquiformerV2 20 / 100 for energy / forces) and have not been empirically fit.
