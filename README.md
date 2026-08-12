# EqFSA: SE(3)-Equivariant Fourier-Space Attention

We present an SE(3)-equivariant Fourier-space attention deep-learning model mimicing Ewald decomposition for the prediction of binding-free solvation nergies of molecular systems. EqFSA combines a short-range, depth-wise tensor-product attention mechanism between neighboring atoms with a long-range, Fourier-space attention mechanism routed through a set of virtual master nodes, so that both local chemistry
and long-range electrostatic-style effects while completely mitigating the oversquashing problem of traditional MPNNs and SE(3)-equivariant transformers.

## Motivation

Traditional message-passing neural networks (MPNNs) grow their receptive field linearly with the
number of layers, which forces deep stacks to reach long-range effects and
causes oversquashing of information at each node. We fix these problems by introducing 2 novel methodologies:

- Each atom's update at layer `m` only depends on the initial (layer-0) features of its neighbors, so the local receptive field stays fixed at the cutoff radius regardless of depth. To sequentially improve feature representations, rather than expanding the receptive field, we propagate higher order tensor features up to the current layer index to hierarchically each atom's hidden embeddings conditioned on purely local interactions.
- We incorporate long-range effects, such as the Coulomb and Van der Waals potentials,through a small set of virtual master nodes centered at the molecular center of mass. Since SE(3)-invariant (type-0) features have been shown to be sufficient for encoding long-range effects, master-node updates are computed in Fourier space using plane-wave projections with wave vectors capped by an energy cutoff rather than by further local message passing. Since the frequencies rapidly decay with increasing wavelengths, these serve as a natural choice to encode long range effects whilst simultaneously being able to handle the periodic boundary conditions of MD simulations.
- After passing through the total L laers, a multli-head cross-attention layer lets each atom's local embedding attend over the global master-node embeddings via a depth-wise tensor-product with MLP-based attention weights along with nonlinear dynamic attention resembling GATv2 to improve expressivity.

Training follows a two-stage strategy bottom up and top-down strategy for learning on ab-initio and experimental data:

1. **Stage 1** Pre-trains the machine-learned potential backbone on the SPICE dataset with an energy+force matching objective.
2. **Stage 2** Fine-tunes on solvent-solute trajectory data (Solv@TUM + FreeSolv) to predict solvation free energies directly.
3. **Stage 3** Uses differentiable simulations to learn the autocorrelation function of the time derivative of the dipole moment to predict IR spectra conditioned on solute and solvent conformations/descriptors

## Main equations

**Local (short-range) update.** For atom `i` at layer `m`, the type-`l` feature update is

```
h_i^{l,m+1} = W^{lm} h_i^{m,m} + sigma( sum_{j in N_i} sum_{k=0}^{m} alpha_ij * W_V^{lk}(x_j - x_i) h_j^{k,0} )
```

where `sigma` is an activation, `⊗` (in the tensor-product sense) contracts spherical-harmonic
edge features with neighbor features, and

```
alpha_ij = Phi( f(h_i, h_j, e_ij) )
f(h_i, h_j, e_ij) = Psi(e_ij) * sigma( sum_{k=0}^{m} W^{0k} (h_i^{k,m} || h_j^{k,0}) )
W_V^{lk} = sum_{J=|k-l|}^{k+l} rho(||x_j - x_i||) * unvec( Q^{lk,T} Y_J(x_j - x_i) )
```

`Phi` is softmax, `Psi` is an MLP, `rho` is a scalar radial network, `Y_J` are spherical harmonics,
and `Q^{lk}` are the associated Clebsch-Gordan coefficients. This is implemented by `SRAttention`
and `EnergyLayer` in `src/stage_1/attention.py` / `src/stage_1/model.py`.

**Long-range (master-node) update.** For a virtual global node `a` centered at the graph's
center of mass, the type-0 feature update is

```
g_a^{m+1} = W g_a^m + sum_{i,j in G} sum_{k in Lambda'} Psi_g(h~(G), e_ij) * exp(i k . (x_j - x_i)) * alpha_ij * W_V g_a^m
```

with wave vectors `k` set by an energy cutoff, and

```
alpha_ij = Phi( Psi(h_i^{0,m}, h_j^{0,m}, g_a^m) )
W_V = Psi(g_a^m) * sigma( sum_{k=0}^{m} W^{0k} (h_i^{k,m} || h_j^{k,m}) )
```

**Stage-1 training objective (energy + force matching).**

```
L_1 = (1 / N_b) * sum_{l=1}^{N_b} [ lambda_1 * (U_l - U_hat_l)^2
                                    + lambda_2 * sum_{i=1}^{N_l,a} sum_{n=1}^{3} (F_{iln} - F_hat_{iln})^2 ]
```

where `N_b` is the batch size, `N_{l,a}` the number of atoms in configuration `l`, `U` the
potential energy, and `F` the per-atom, per-direction force; `lambda_1`/`lambda_2` weight the
energy and force terms respectively (`--lambda1`/`--lambda2` in `train_spice.py`, defaults `1.0`
and `0.5`).

**Unseen-solvent generalization distance.** To hold out chemically diverse solvents, solvents are
embedded via Morgan fingerprints and physicochemical descriptors, and a combined similarity
distance is computed as

```
D_ij = 0.5 * (1 - T_sim(m_i, m_j)) + 0.5 * d(x_i, x_j)
```

where `T_sim` is Tanimoto similarity over fingerprints and `d` is Euclidean distance over
standardized descriptors; farthest-point sampling on `D` then selects a diverse held-out solvent
set.

## Installation

```bash
git clone https://github.com/Adithya-Rao1/Eq-FSA.git
cd Eq-FSA
pip install -e .
```

Note: some modules import a few additional internal packages (`embeddings.bases`, `data_processing`,
`diffusion`, `log_setup`) that are not part of this repository and must be available separately on
the `PYTHONPATH` for training/inference to run end-to-end.

## Usage

```bash
# Download the SPICE dataset
python -m stage_1.download_spice

# Stage-1 pre-training on SPICE (energy + force matching)
python -m stage_1.train_spice --hdf5-paths <path/to/spice.hdf5> --subsets pubchem

# Baseline experiment comparisons (GAFF / COSMO-RS / SCCS / MACE / FieldSchNet)
python -m stage_1.experiments.run_experiments
```