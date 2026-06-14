# CATCHY — Catching Enzymes in Action
**Coarse-grained MD simulations of enzyme-polymer systems**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![OpenMM 8](https://img.shields.io/badge/OpenMM-8.x-green.svg)](https://openmm.org/)
[![ANR funded](https://img.shields.io/badge/funded-ANR-red.svg)](https://anr.fr/)

---

## Scientific context

This project is the numerical core of the **CATCHY** ANR project (IATE lab, Montpellier).  
It directly extends **Sorichetti, Hugouvieux & Kob, *Macromolecules* 2021** (Paper 1), which
studied passive nanoparticle (NP) diffusion in a permanently cross-linked polymer network.

**Key extension:** NPs become **enzymes** that *cleave* network bonds upon contact.  
All other physics (Kremer-Grest network, potentials, observables, analysis) is kept identical
to Paper 1, so enzymatic enhancement is cleanly isolated by comparing active vs passive runs.

### Central scientific question
> How do enzyme size (confinement C = σ_E/λ), polymer affinity (attractive vs repulsive),
> and enzymatic activity (k_cat) collectively control enzyme progression through the matrix?

### Predicted phenomenology
| Regime       | Passive (k_cat = 0)                    | Active (k_cat > 0)                         |
|--------------|----------------------------------------|--------------------------------------------|
| C ≲ 1 (free) | D follows Cai–Rubinstein               | Cleavage negligible — D_active ≈ D_passive |
| 1 ≲ C ≲ 3   | Activated hopping, large α₂            | Cleavage accelerates hops                  |
| C ≳ 3        | Near-zero D, extremely heterogeneous   | Cleavage opens new paths — D_active >> D_passive |

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
├── analysis/
│   ├── msd.py                    # MSD, D(C), β(t), theory fits
│   ├── van_hove.py               # G_s, G_d, hopping detection
│   ├── non_gaussian.py           # α₂(t)
│   ├── pore_size.py              # Pore-size distribution P(r), ξ(t)
│   ├── degradation.py            # Bond survival S(t), D_active/D_passive
│   └── plot_all.py               # All figures A–I
├── notebooks/
│   └── results_overview.ipynb
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
python 01_build_network.py --config ../configs/strong_confinement.yaml
python 02_embed_enzymes.py --config ../configs/strong_confinement.yaml
python 03_production.py    --config ../configs/strong_confinement.yaml
```

---

## Physics model

### Polymer network — identical to Paper 1 (Sorichetti et al. 2021)

- Kremer–Grest bead-spring model, N_m = 8000 monomers
- **WCA repulsion** between all beads: U_WCA = 4ε[(σ/r)^12 − (σ/r)^6] + ε, r < 2^(1/6)σ
- **FENE bonds** on backbone and cross-links: U_FENE = −(k R₀²/2) ln[1−(r/R₀)²], k=30, R₀=1.5
- Polydisperse strand lengths from **Flory–Stockmayer** distribution:
  p(n) = (1/⟨n⟩)(1−1/⟨n⟩)^(n−1),  ⟨n⟩ = 6
- Cross-link fraction c = 0.1, Langevin thermostat γ_m = 0.1, T = 1.0, dt = 0.006

### Enzyme–monomer interaction — Paper 1 eq. 4 (expanded LJ)

```
U_ij(r) = 4ε_ij [ (σ_ij/(r−Δ))^12 − (σ_ij/(r−Δ))^6 ] + ε_ij ,  r < r_c + Δ
         = 0                                                        ,  r ≥ r_c + Δ

σ_ij = (σ_E + σ_m)/2      (Lorentz–Berthelot mixing)
Δ    = (σ_E − σ_m)/2      (shift: contact distance = σ_ij)
```

| Type       | ε_ij | r_c                  |
|------------|------|----------------------|
| Repulsive  | 1    | 2^(1/6) σ_ij         |
| Attractive | 2    | 2.5 σ_ij             |

### Enzymatic cleavage — new in CATCHY

At each timestep, `CleavageManager` iterates over all intact FENE bonds:
1. Find the midpoint of the bond.
2. For each enzyme within `r_cleave = 1.5 σ_E` of the midpoint:
3. Draw uniform random u ∈ [0,1]. If u < k_cat × dt → **cleave** (k → 0 via `updateParametersInContext`).

Bond cleavage is **irreversible**. Cleaved bond count is recorded at each save frame.

### Confinement parameter

C = σ_E / λ

λ = dynamic localization length of cross-links (long-time MSD plateau of cross-link beads).

| ρ₀    | λ    |
|-------|------|
| 0.190 | 3.12 |
| 0.290 | 2.01 |
| 0.375 | 1.51 |

---

## Observables

### Reproduced from Paper 1
| Observable | Script | Description |
|------------|--------|-------------|
| MSD(t)     | analysis/msd.py | Enzyme mean-square displacement |
| D_E(C)     | analysis/msd.py | Long-time diffusion coefficient vs C |
| β(t)       | analysis/msd.py | Local subdiffusion exponent |
| G_s(r,t)   | analysis/van_hove.py | Self van Hove function |
| α₂(t)      | analysis/non_gaussian.py | Non-Gaussian parameter |

### New CATCHY observables
| Observable         | Script | Description |
|--------------------|--------|-------------|
| S(t)               | analysis/degradation.py | Bond survival fraction |
| D_active/D_passive | analysis/degradation.py | Enzymatic mobility enhancement vs C |
| ξ(t)               | analysis/pore_size.py | Mean mesh size during degradation |

---

## Output files

For each label `sigma{X}_{attractive|repulsive}_{active|passive}`:

```
output/
├── network_data.npz            # positions, bonds, λ, cross-link MSD
├── enzyme_system_{label}.xml   # equilibrated OpenMM state
├── traj_{label}.h5             # trajectory: positions + active_bonds array
├── msd_{label}.npz             # enzyme MSD vs time
├── survival_{label}.npz        # bond survival S(t)
└── figs/
    ├── fig_A_msd.pdf
    ├── fig_B_diffusion.pdf
    ├── fig_C_beta.pdf
    ├── fig_D_vanhove.pdf
    ├── fig_E_nongaussian.pdf
    ├── fig_F_survival.pdf
    └── fig_G_enhancement.pdf
```

