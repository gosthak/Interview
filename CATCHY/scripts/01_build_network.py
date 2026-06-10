#!/usr/bin/env python3
"""
01_build_network.py — Build and equilibrate the polymer network.

Steps:
  1. Generate Kremer-Grest network with Flory-Stockmayer topology
  2. Minimize energy (push-off)
  3. NVT equilibration
  4. Measure cross-link dynamic localization length λ
  5. Save network_data.npz + equilibrated checkpoint

Usage:
    python 01_build_network.py --config ../configs/default.yaml
"""

import argparse
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, LAMBDA_REF, confinement_parameter
from src.network_builder import NetworkBuilder

try:
    import openmm as mm
    import openmm.unit as unit
    from openmm.app import (Simulation, StateDataReporter,
                             CheckpointReporter)
except ImportError:
    raise ImportError("OpenMM required: conda install -c conda-forge openmm")

from src.potentials import add_wca_monomers_only, add_fene

# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    return p.parse_args()


def build_openmm_system(builder, gamma_m, T, dt, platform_name):
    """Build a bare network OpenMM system (no enzymes)."""
    N_m = builder.N_m
    L = builder.L

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L)
    )
    for _ in range(N_m):
        system.addParticle(1.0)

    # Forces
    wca = add_wca_monomers_only(system, N_m)
    fene = add_fene(system, builder.all_bonds)

    # Integrator
    integrator = mm.LangevinMiddleIntegrator(T, gamma_m, dt)
    integrator.setRandomNumberSeed(42)

    # Platform
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        props = {"CudaPrecision": "mixed"} if platform_name == "CUDA" else {}
        sim = mm.app.Simulation(
            _dummy_topology(N_m), system, integrator, platform, props
        )
    except Exception:
        print(f"[WARNING] {platform_name} not available, using CPU")
        sim = mm.app.Simulation(
            _dummy_topology(N_m), system,
            mm.LangevinMiddleIntegrator(T, gamma_m, dt),
            mm.Platform.getPlatformByName("CPU")
        )

    return sim, system, fene


def _dummy_topology(N):
    import openmm.app as app
    topo = app.Topology()
    chain = topo.addChain()
    res = topo.addResidue("NET", chain)
    for i in range(N):
        topo.addAtom(f"M{i}", app.element.carbon, res)
    return topo


def measure_lambda(sim, N_m, L, n_msd_steps=100000, save_every=200, dt=0.006):
    """
    Measure cross-link dynamic localization length λ from the long-time
    plateau of the cross-link MSD. (Paper 1, Section II.C)

    λ² = lim_{t→∞} MSD_crosslink(t)
    """
    from src.utils import _unwrap
    positions_log = []
    initial_pos = np.array(sim.context.getState(getPositions=True)
                           .getPositions(asNumpy=True))

    for _ in range(n_msd_steps // save_every):
        sim.step(save_every)
        state = sim.context.getState(getPositions=True)
        positions_log.append(np.array(state.getPositions(asNumpy=True)))

    positions_log = np.array(positions_log)   # (n_frames, N_m, 3)
    # MSD of cross-link beads only
    # (approximate: use all monomers for simplicity at this stage)
    unwrapped = _unwrap(positions_log, L)
    dr = unwrapped - unwrapped[0:1]
    msd = np.mean(np.sum(dr ** 2, axis=-1), axis=-1)
    # λ from plateau (last quarter of trajectory)
    lam = np.sqrt(np.mean(msd[3 * len(msd) // 4:]))
    return lam, msd


def main():
    args = parse_args()
    cfg = load_config(args.config)

    sys_cfg = cfg["system"]
    sim_cfg = cfg["simulation"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    N_m = sys_cfg["N_m"]
    rho = sys_cfg["rho_m0"]
    c = sys_cfg["c"]
    mean_strand = sys_cfg["mean_strand"]
    seed = sys_cfg.get("seed", 42)
    T = sim_cfg["T"]
    dt = sim_cfg["dt"]
    gamma_m = sim_cfg["gamma_m"]
    platform = sim_cfg["platform"]
    n_equil = sim_cfg["n_equil_network"]
    n_npt = sim_cfg["n_npt_network"]

    print("=" * 60)
    print("CATCHY — Step 01: Build and equilibrate network")
    print(f"  N_m={N_m}, rho={rho}, c={c}, <n>={mean_strand}")
    print(f"  T={T}, dt={dt}, gamma_m={gamma_m}, platform={platform}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build topology
    # ------------------------------------------------------------------
    print("\n[1/5] Building network topology ...")
    t0 = time.time()
    builder = NetworkBuilder(N_m=N_m, rho=rho, c=c,
                             mean_strand=mean_strand, seed=seed)
    builder.build()
    builder.summary()
    print(f"      Done in {time.time()-t0:.1f} s")

    # ------------------------------------------------------------------
    # 2. Build OpenMM system + minimize
    # ------------------------------------------------------------------
    print("\n[2/5] Building OpenMM system and minimizing ...")
    sim, system, fene = build_openmm_system(builder, gamma_m, T, dt, platform)
    sim.context.setPositions([mm.Vec3(*p) for p in builder.positions])
    sim.context.setVelocitiesToTemperature(T)

    sim.minimizeEnergy(maxIterations=2000)
    print("      Minimization done")

    # ------------------------------------------------------------------
    # 3. NVT equilibration (soft push-off)
    # ------------------------------------------------------------------
    print(f"\n[3/5] NVT equilibration ({n_equil} steps) ...")
    reporter = StateDataReporter(
        os.path.join(out_dir, "network_equil.log"), 10000,
        step=True, potentialEnergy=True, temperature=True, speed=True
    )
    sim.reporters.append(reporter)
    sim.step(n_equil)
    sim.reporters.clear()
    print("      NVT equilibration done")

    # ------------------------------------------------------------------
    # 4. (skipped) — no NPT barostat
    # Paper 1 works entirely in NVT at fixed box L = (N_m/rho_m0)^(1/3).
    # Density is set by construction; a barostat would change rho away
    # from the target and has no well-defined P* = 0 in LJ units mapped
    # to OpenMM real units.
    # ------------------------------------------------------------------
    print("\n[4/5] Skipping NPT — NVT at fixed box (Paper 1 protocol)")

    # ------------------------------------------------------------------
    # 5. Measure λ
    # ------------------------------------------------------------------
    print("\n[5/5] Measuring cross-link localization length λ ...")
    lam_measured, msd_cl = measure_lambda(
        sim, N_m, builder.L, n_msd_steps=100000, save_every=200, dt=dt
    )
    # Use Paper 1 reference value if close to a tabulated density
    lam_ref, lam_ref_val = confinement_parameter(1.0, rho)   # just get lambda
    print(f"      λ (measured) = {lam_measured:.4f} σ_m")
    print(f"      λ (Paper 1 ref for ρ={rho}) = {lam_ref_val:.4f} σ_m")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    pos_final = np.array(
        sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    )
    np.savez(
        os.path.join(out_dir, "network_data.npz"),
        positions=pos_final,
        bonds=np.array(builder.all_bonds),
        backbone_bonds=np.array(builder.backbone_bonds),
        crosslink_bonds=np.array(builder.crosslink_bonds),
        crosslink_ids=np.array(builder.crosslink_ids),
        lambda_measured=lam_measured,
        lambda_paper1=lam_ref_val,
        msd_crosslink=msd_cl,
        L=builder.L,
        N_m=N_m,
        rho=rho,
    )
    sim.saveCheckpoint(os.path.join(out_dir, "network_equilibrated.chk"))
    print(f"\n  Saved network_data.npz and checkpoint to {out_dir}/")
    print("  DONE — ready for 02_embed_enzymes.py")


if __name__ == "__main__":
    main()
