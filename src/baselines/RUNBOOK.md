# Runbook: first run on a Slurm cluster (two GPU nodes, shared `$HOME/scratch`)

Check before running: 
- Only this repo exists on the cluster
- `$HOME/scratch` is visible from every node
- Slurm
- `apptainer` or `singularity` on the compute nodes.
- For the partitions, use `gpu_v100s` for GPU jobs (5-day limit, one node idle) and `tiny` for CPU jobs (5-day limit, two idle nodes, and `batch` fully allocated). `gpu_a100` is the fallback. The commands below set these as the shell variables `GPU_PART` and `CPU_PART`, so change them there if you use other partitions.
- Pass `--time` with `tiny` (the renderer uses default walltimes for the partitions in `campaign.py`).
- All data, images or outputs stored inside `$HOME/scratch`.

Verified and pending (2026-11-11):
- Verified on a single Ubuntu GPU box (A10 22 GiB, apptainer, no Slurm), via `slurm/run_local.py`: image builds, unit tests, exports for rMD17, QM9, MD22, ANI-1x and a capped SPICE pilot run. We also ran one-epoch pilot trainings of all six models on one instance per dataset family. 25 of the 30 pilot jobs had finished at the time of writing (due to OOM).
- GAFF2 / OpenFF classical runs on rMD17, MD22 and a 100-frame SPICE sample. Skipped on purpose:
  - MD22 buckyball-catcher: antechamber hangs during AM1-BCC charge assignment.
  - MD22 double-walled nanotube: its 370 atoms exceed the 200-atom limit.
  - SPICE boron molecules: boron is outside the GAFF2 organic element set.
- Need to run on Slurm cluster: the sbatch templates, the submit scripts and `--fakeroot` image builds. The built-in
  partition defaults (`gpu_a40-test`, `stingy`) do not exist or are reserved there, so always pass `--gpu-partition` and `--cpu-partition`. Need the `.err` files (see "Troubleshooting").
- Failed on the first server: MACE and EquiformerV2 ran out of GPU memory on the 22 GiB A10 (24 GB card) for MD22 Ac-Ala3-NHMe and SPICE. The batch sizes that ran out of memory were the defaults, MACE 32 (train and validation) and EquiformerV2 8. Lower them with `--batch-size` through `--extra-adapter-args` if the same happens on cluster GPUs. The rMD17 aspirin test in step 5 was not affected.
- Hyperparameters, epochs and per-library loss weights are placeholders. The MACE model's energy did not learn beyond atom counts in one ANI-1x epoch, so we still need to tune these parameters prior to full benchmarks. 

## 1. Stage the repo
```bash
mkdir -p $HOME/scratch && cd $HOME/scratch
git clone <repo-url> Eq-FSA && cd Eq-FSA && git checkout new/baselines
```
The repo must live at `$HOME/scratch/Eq-FSA`. Jobs bind it read-only at `/repo`.

## 2. Build the four images (about an hour each or more; unmeasured on this cluster)
Run the following on a compute node, not the login node (e.g. `srun -p tiny -c 8 --mem 32G -t 06:00:00 --pty bash`):
```bash
cd $HOME/scratch/Eq-FSA
bash src/baselines/containers/build_and_verify.sh 2>&1 | tee $HOME/scratch/build.log
```
Images, digests (`*.sif.sha256`) and the build cache go to `$HOME/scratch`. The script uses `sudo` if it is passwordless and otherwise `--fakeroot`. If
neither works on the cluster, build on a machine that has root and copy the four `.sif` and four `.sif.sha256` files into `$HOME/scratch`.
Expect `OK` for each of the four images.

## 3. Get the rMD17 data
```bash
bash $HOME/scratch/Eq-FSA/src/baselines/datasets/fetch_rmd17.sh
```
It tries the two known hosts, checks size and sha256, and extracts to `$HOME/scratch/data/rmd17/{npz_data,splits}`. If both hosts refuse automated downloads from the first server (HTTP 403 / 404), download `rmd17.tar.bz2` in a browser from
<https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038>, place it at `$HOME/scratch/data/rmd17.tar.bz2`, and run the script again.

NOTE: If uploading through a web file manager fails for the full 1 GB archive (it truncated on the A10 server), the following method should fix it: 
1. Run `split -b 50M rmd17.tar.bz2 rmd17.part_` and `sha256sum rmd17.part_*`
2. Upload each piece
3. Check each with sha256sum on the cluster
4. Rebuild it with `cat rmd17.part_* > $HOME/scratch/data/rmd17.tar.bz2` and run the script again.

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
Every test should pass. Do not run the whole `baselines/tests` directory at once as one failed test hung the rest on some runs. Report any failures or skips.

## 5. Smoke, export, one-epoch pilot (the two-node test)
```bash
cd $HOME/scratch/Eq-FSA/src
S=$HOME/scratch; O=$S/campaign; GPU_PART=gpu_v100s; CPU_PART=tiny
python3 -m baselines.slurm.campaign --phase phase0_smoke --scratch-root $S --out-dir $O --gpu-partition $GPU_PART
bash $O/submit_phase0_smoke.sh                 # 4 jobs. Wait until `squeue -u $USER` is empty
python3 -m baselines.slurm.campaign --phase phase1_export --datasets rmd17 --scratch-root $S --out-dir $O --cpu-partition $CPU_PART --time 01:00:00
bash $O/submit_phase1_export.sh                # 10 jobs. Wait until squeue is empty (training jobs need their export to start)
python3 -m baselines.slurm.campaign --phase phase2_train_calibration --datasets rmd17 --pilot --epochs 1 --scratch-root $S --out-dir $O --gpu-partition $GPU_PART --time 02:00:00
bash $O/submit_phase2_train_calibration.sh     # 6 jobs (one per model, rMD17 aspirin). Slurm decides placement and with two GPU nodes free, two should run at once.
```
Job scripts and the submit scripts are only written by `campaign`, and nothing is submitted until you run a `submit_*.sh`. Slurm `.out/.err` files go to`$O/logs/`. On the first server, the six one-epoch jobs took 3 to 5 minutes each including start-up (EquiformerV2 model took the longest).

Before submitting, check the node sizes (90G of memory and 8 CPUs per GPU job according to templates):
```bash
scontrol show node gpu-v100s-06 | grep -E "RealMemory|CPUTot|Gres"
```
If `RealMemory` (reported in MB) is below ~92000, edit `GPU_RESOURCES` in `slurm/campaign.py` before rendering. Report the three values.

The CPU jobs on `tiny` (exports and classical) ask for 64G and 8 CPUs (`CPU_RESOURCES`), and the smoke jobs ask for 40G and 4 CPUs (`SMOKE_RESOURCES`). Check the other nodes too:
```bash
sinfo -p tiny,gpu_v100s -o "%P %n %m %c %G"
```
If a `tiny` node shows `%m` less than ~64000, lower `mem` in `CPU_RESOURCES`. If a `gpu_v100s` node shows less than ~40000, lower it in `SMOKE_RESOURCES`. Report the output.

## 6. Check the results
```bash
cd $HOME/scratch/Eq-FSA/src
python3 -m baselines.slurm.check_outputs --phase phase1_export phase2_train_calibration --datasets rmd17 --only rmd17:aspirin --pilot --scratch-root $HOME/scratch
```
It lists every expected export / metrics file, flags missing or non-finite ones, and scans `$O/logs` for `Traceback`, `CUDA out of memory`, `slurmstepd: error`.
Exit code 0 means all complete and no error text. Library `FutureWarning`s about `torch.load` are expected and do not count as errors.
To confirm nothing was written outside scratch, run `du -sh $HOME/scratch/*` and `ls -la $HOME $HOME/.cache` and compare with the listing from before step 2.

## 7. The other datasets and the classical force fields (after steps 1-6 pass)
Steps 3 and 5 cover rMD17 only. Everything below was run end to end on the single GPU box with `run_local`, but none of it has run through sbatch. Data layout the loaders expect:
`$HOME/scratch/data/{qm9/*.xyz, md22/md22_<molecule>.npz, spice/<file>.hdf5, ani1x/ani1x-release.h5, rmd17/}`.
```bash
S=$HOME/scratch; O=$S/campaign; GPU_PART=gpu_v100s; CPU_PART=tiny
mkdir -p $S/data/md22 && cd $S/data/md22
for m in Ac-Ala3-NHMe DHA stachyose AT-AT AT-AT-CG-CG buckyball-catcher double-walled_nanotube; do curl -L -C - -O http://www.quantum-machine.org/gdml/repo/datasets/md22_$m.npz; done
ls -la $S/data/md22
```
QM9 (133,885 molecules) and ANI-1x (5,590,846,027 bytes). On the first server `wget`/`curl` got HTTP 403 from figshare but Python `urllib` worked, so try that first:
```bash
mkdir -p $S/data/qm9 $S/data/ani1x
python3 -c "import urllib.request as u; u.urlretrieve('https://ndownloader.figshare.com/files/3195389', '$S/data/dsgdb9nsd.xyz.tar.bz2')"
tar -xjf $S/data/dsgdb9nsd.xyz.tar.bz2 -C $S/data/qm9 && ls $S/data/qm9/*.xyz | wc -l
python3 -c "import urllib.request as u; u.urlretrieve('https://ndownloader.figshare.com/files/18112775', '$S/data/ani1x/ani1x-release.h5')"
ls -l $S/data/ani1x/ani1x-release.h5
```
The QM9 count should be 133885. If it prints 0, the archive extracted into a subdirectory. To fix this, move its `.xyz` files up into `$S/data/qm9`.

### SPICE Dataset
SPICE was pre-downloaded on the remote server, so no script downloads it here. Download the SPICE 2.0.1 HDF5 file (about 37 GB, CC0) from the official release at [Zenodo record 10975225](https://zenodo.org/records/10975225) (DOI 10.5281/zenodo.10975225), `SPICE-2.0.1.hdf5` (37,479,271,148 bytes, md5 `bfba2224b6540e1390a579569b475510`. Optionally, check with `md5sum`). Save it as `$S/data/spice/SPICE.hdf5` and pass `--spice-file SPICE.hdf5` to every `campaign` / `run_local` command for SPICE (the default name is `SPICE-2.0.1.hdf5`). 

The full SPICE export would be about 5 GB of extxyz for training, so build a smaller run by hand (names the directory the training phase looks for):
```bash
CT=$(command -v apptainer || command -v singularity)
$CT exec --bind $S:$S --bind $S/Eq-FSA:/repo:ro --env PYTHONPATH=/repo/src $S/mlip-mpnn.sif python -m baselines.export.export_splits --dataset spice --split molecule_95_5_val5 --data-root $S/data/spice/SPICE.hdf5 --out-dir $S/exports/spice_molecule_95_5_val5 --subset-seed 0 --max-train 20000 --max-val 2000 --max-test 2000
```
Below are the commands for the exports and one-epoch training (seed 0, one instance per family) for all ML models, then the classical force-field evaluations. All submit scripts are written by `campaign`, same as step 5 (the partition variables are set in the first block of this step):
```bash
cd $S/Eq-FSA/src
python3 -m baselines.slurm.campaign --phase phase1_export --datasets qm9 md22 ani1x --scratch-root $S --out-dir $O --cpu-partition $CPU_PART --time 04:00:00
python3 -m baselines.slurm.campaign --phase phase2_train_calibration --datasets qm9 md22 --pilot --epochs 1 --scratch-root $S --out-dir $O --gpu-partition $GPU_PART --time 24:00:00
python3 -m baselines.slurm.campaign --phase phase2_train_ani1x --pilot --epochs 1 --scratch-root $S --out-dir $O --gpu-partition $GPU_PART --time 24:00:00
python3 -m baselines.slurm.campaign --phase phase2_train_spice --pilot --epochs 1 --spice-file SPICE.hdf5 --scratch-root $S --out-dir $O --gpu-partition $GPU_PART --time 24:00:00
python3 -m baselines.slurm.campaign --phase phase3_classical --datasets rmd17 md22 --scratch-root $S --out-dir $O --cpu-partition $CPU_PART --time 12:00:00
```
Then run the matching `submit_<phase>.sh` for each after its export has finished. Here are the wall-clock time measurements on the remote A10 for one epoch, for reference: ANI-1x (100k frames) SchNet 17 min, PaiNN 21 min, MACE 104 min.

Notes for these pilot runs: 
- QM9 does not yet exclude the 3,054 uncharacterized molecules that the standard split does.
- The MD22 buckyball-catcher hangs in antechamber (AM1-BCC charge assignment never ends), and the 370-atom nanotube is skipped because it exceeds the 200-atom limit. 
- `campaign` has no `--only`, so after rendering `phase3_classical` delete both the `gaff2` and `openff` `buckyball-catcher` job lines from `$O/submit_phase3_classical.sh` (or `scancel` both jobs). The nanotube jobs end in seconds with a recorded skip, so leave them.
- Do not add `spice` to the classical datasets for a pilot as every distinct molecule needs its own AM1-BCC (about a minute each, so a capped 2,000-frame set would take many hours) and boron molecules are skipped.

## Troubleshooting
If something fails, save: 
- the failing job's `$O/logs/*.err` and `.out` (last 60 lines)
- `sinfo -s`
- `$(command -v apptainer || command -v singularity) --version`
- output of `check_outputs`. 

Most likely first problems: partition names or missing `--time` (the renderer refuses unknown partitions without one), `--fakeroot` builds not permitted, GPU not visible inside the container (`--nv`), memory limits (the templates ask for 90G on training jobs, so edit `GPU_RESOURCES` in `slurm/campaign.py` if the nodes are smaller).

## Model notes for pilot runs
- EquiformerV2 defaults here (energy weight 20, EMA 0.99) were chosen from rMD17 aspirin runs. The OC20 values (energy weight 2, EMA 0.999) left the energy at the mean baseline and the forces at 546 meV/Å after 5 epochs.
- All other hyperparameters and epoch counts are not tested. 
- `--epochs` defaults to 1 for this test run. 
- Per-model loss weights differ between libraries (SchNetPack 0.01 / 0.99, NequIP 1 / 1, MACE 1 / 100, EquiformerV2 20 / 100 for energy / forces) and have not been tuned yet.
