# CATCHY — Catching Enzymes in Action
**Coarse-grained MD simulations of enzyme-polymer systems**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![OpenMM 8](https://img.shields.io/badge/OpenMM-8.x-green.svg)](https://openmm.org/)

---

## Scientific context

It directly extends **Sorichetti, Hugouvieux & Kob, *Macromolecules* 2021** (Paper 1), which
studied passive nanoparticle diffusion in a permanently cross-linked polymer network.

**Key extension:** NPs become **enzymes** that *cleave* network bonds upon contact.  
All other physics (Kremer-Grest network, potentials, observables, analysis) is kept identical
to Paper 1, so enzymatic enhancement is cleanly isolated by comparing active vs passive runs.

### Central scientific question
> How do enzyme size (confinement C = σ_E/λ), polymer affinity (attractive vs repulsive),
> and enzymatic activity (k_cat) collectively control enzyme progression through the matrix?


---

## Repository structure

```
CATCHY/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml              # N_m=8000, ρ=0.290, σ_E=[1–5], k_cat=0.005
│   ├── weak_confinement.yaml     # C ≲ 1
│   ├── strong_confinement.yaml   # 1 ≲ C ≲ 3
│   └── extreme_confinement.yaml  # C ≳ 3
├── src/
│   ├── __init__.py
│   ├── potentials.py             # WCA, FENE, expanded-LJ (Paper 1 eqs 2–4)
│   ├── network_builder.py        # Flory–Stockmayer polydisperse network
│   ├── enzyme_system.py          # Full OpenMM system assembly
│   ├── cleavage.py               # Stochastic enzymatic bond cleavage
│   └── utils.py                  # HDF5 writer, checkpointing, Paper 1 ref values
├── scripts/
│   ├── 01_build_network.py       # Build + equilibrate network, compute λ
│   ├── 02_embed_enzymes.py       # Insert enzymes, push-off, equilibrate
│   ├── 03_production.py          # Production run with cleavage loop
│   └── run_all.sh                # Full pipeline launcher
└── tests/
    └── test_potentials.py
```

---

## Quick start

```bash
# 1. Create conda environment
conda create -n catchy python=3.10 -y
conda activate catchy
conda install -c conda-forge openmm -y
pip install -r requirements.txt

# 2. Run full pipeline (GPU recommended — 5-7 days on default config)
cd scripts
bash run_all.sh --config ../configs/default.yaml --platform CUDA

# 3. CPU-only fallback
bash run_all.sh --config ../configs/default.yaml --platform CPU
```

### Running specific confinement regimes

Each config targets a different range of C = σ_E / λ via `sigma_list`:

```bash
# Weak confinement (C ≲ 1) — fastest, enzymes diffuse freely
bash run_all.sh --config ../configs/weak_confinement.yaml

# Strong confinement (1 ≲ C ≲ 3) — hopping regime
bash run_all.sh --config ../configs/strong_confinement.yaml

# Extreme confinement (C ≳ 3) — enzymes nearly trapped, cleavage essential
bash run_all.sh --config ../configs/extreme_confinement.yaml
```

Or step by step:

```bash
python 01_build_network.py --config ../configs/default.yaml
python 02_embed_enzymes.py --config ../configs/default.yaml
python 03_production.py    --config ../configs/default.yaml
```

---

## Physics model

### Polymer network — identical to Paper 1 (Sorichetti et al. 2021)

Kremer–Grest bead-spring model in LJ reduced units (σ=1, ε=1, m=1, τ*=1).

**WCA repulsion** between monomer pairs (eq. 2):

```
U_WCA(r) = 4ε[(σ/r)^12 − (σ/r)^6] + ε,   r < 2^(1/6)σ
           = 0,                              r ≥ 2^(1/6)σ
```

**FENE bonds** on backbone and cross-links (eq. 3):

```
U_FENE(r) = −(k R₀²/2) ln[1 − (r/R₀)²],   k = 30 ε/σ², R₀ = 1.5 σ
```

Equilibrium bond length after WCA+FENE balance: r_eq ≈ 0.96 σ.

**Network topology — cubic lattice + Flory–Stockmayer:**

Network construction differs from Paper 1 (which uses patchy-particle self-assembly)
but yields the same Kremer-Grest physics after equilibration:

1. Place N_m beads on a cubic lattice with **fixed spacing a = 0.8 R₀ = 1.2 σ**.  
   The box size follows: L = n_side × a, so ρ_actual is determined by a, not by ρ_target.  
   This guarantees all nearest-neighbour bonds satisfy r < R₀ — no FENE singularity possible.
2. Randomly select N_cl = c·N_m cross-link beads (valence 3); remaining beads are bivalent.
3. Build chains by walking lattice neighbours; strand lengths drawn from Flory–Stockmayer:  
   p(n) = (1/⟨n⟩)(1−1/⟨n⟩)^(n−1),  ⟨n⟩ = 6
4. Prune dangling ends iteratively; reindex isolated beads.  
   ρ_actual is updated after pruning — it will differ from ρ_target.
5. NVT equilibration with staged dt ramp restores equilibrium bond lengths.

Parameters: c = 0.1, γ_m = 0.1 τ*⁻¹, T = 1.0 ε/k_B, dt = 0.006 τ*.

### Enzyme–monomer interaction — Paper 1 eq. 4 (expanded LJ)

```
U(r) = 4ε [ (σ_m/(r−Δ))^12 − (σ_m/(r−Δ))^6 ] − E_c,   r < r_c + Δ
      = 0,                                                  r ≥ r_c + Δ
```

where σ_m = 1 is the monomer diameter (not the mixing rule σ_ij), and:

```
Δ_Nm = (σ_E − σ_m)/2      (enzyme–monomer shift)
Δ_NN = σ_E − σ_m           (enzyme–enzyme shift)
E_c  = 4ε[(σ_m/r_c)^12 − (σ_m/r_c)^6]   (continuity shift)
```

| Pair             | ε   | r_c                 |
|------------------|-----|---------------------|
| Enzyme–monomer repulsive  | 1 | 2^(1/6) σ_m |
| Enzyme–monomer attractive | 2 | 2.5 σ_m     |
| Enzyme–enzyme    | 1   | 2^(1/6) σ_m         |

Enzyme mass and friction scale with volume: m_E = m(σ_E/σ_m)³,  γ_E = γ_m(σ_m/σ_E)².

### Enzymatic cleavage — new in CATCHY

Every `check_interval` MD steps, `CleavageManager` checks each intact cross-link bond:

1. Compute bond midpoint (PBC minimum image).
2. For each enzyme within `r_cleave = 1.5 σ_E` of the midpoint:
3. Cleave with **Poisson probability**:

```
p = 1 − exp(−k_cat · dt · check_interval)
```

where `k_cat` is the cleavage rate in units of τ*⁻¹ (not probability per step).

Cleavage is **irreversible**: `k_bond → 0` via `fene.setBondParameters` +
`updateParametersInContext`. Only cross-link bonds are cleaved; backbone bonds are permanent.
Events are logged to `cleavage_{label}.log` with timestamp, bond indices, and S(t).

### Confinement parameter

```
C = σ_E / λ
```

λ = dynamic localization length measured from the long-time MSD plateau of all network
monomers after equilibration (Paper 1 definition, averaged over all beads).

Reference values from Paper 1 Table 1:

| ρ₀    | λ [σ] |
|-------|-------|
| 0.190 | 3.12  |
| 0.290 | 2.01  |
| 0.375 | 1.51  |

Note: due to pruning of isolated beads, effective density ρ_eff < ρ₀.
λ is therefore measured in situ and may differ from Paper 1 Table 1.

---



### Reading MSD data

```python
import numpy as np
d = np.load("output/msd_sigma1.0_attractive_active.npz")
# keys: time, msd, lag_steps, D, sigma_E
```


## Output files

For each label `sigma{X}_{attractive|repulsive}_{active|passive}`:

```
output/
├── network_data.npz            # positions, bonds, λ, cross-link MSD
├── enzyme_system_{label}.xml   # equilibrated OpenMM state
├── traj_{label}.h5             # trajectory: positions + active_bonds array
├── msd_{label}.npz             # enzyme MSD vs time
├── survival_{label}.npz        # bond survival S(t)
```

