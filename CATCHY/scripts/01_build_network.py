#!/usr/bin/env python3
"""
01_build_network.py — Build and equilibrate the polymer network.

Steps:
  1. Generate network topology (NetworkBuilder)
  2. Build OpenMM system with FENE bonds
     - FENE uses PBC so wrapped lattice positions work fine
       close WITHOUT PBC (required by CustomBondForce)
  3. Energy minimization
  4. NVT equilibration (staged dt ramp)
  5. Measure cross-link localization length λ
  6. Save network_data.npz + checkpoint

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
from src.potentials import add_wca_monomers_only, add_fene

try:
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    raise ImportError("OpenMM required: conda install -c conda-forge openmm")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    return p.parse_args()


def dummy_topology(N):
    topo = app.Topology()
    chain = topo.addChain()
    res = topo.addResidue("NET", chain)
    for i in range(N):
        topo.addAtom(f"M{i}", app.element.carbon, res)
    return topo


def make_simulation(N, L, bonds, gamma_m, T, dt, platform_name):
    """Build OpenMM system + simulation from scratch."""
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(L, 0, 0), mm.Vec3(0, L, 0), mm.Vec3(0, 0, L)
    )
    for _ in range(N):
        system.addParticle(1.0)

    wca  = add_wca_monomers_only(system, N)
    fene = add_fene(system, bonds)

    integrator = mm.LangevinMiddleIntegrator(T, gamma_m, dt)
    integrator.setRandomNumberSeed(42)

    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        props = {"CudaPrecision": "mixed"} if platform_name == "CUDA" else {}
        sim = app.Simulation(dummy_topology(N), system, integrator, platform, props)
    except Exception:
        print(f"  [{platform_name} unavailable, using CPU]")
        sim = app.Simulation(
            dummy_topology(N), system,
            mm.LangevinMiddleIntegrator(T, gamma_m, dt),
            mm.Platform.getPlatformByName("CPU")
        )
    return sim, system, fene


def check_energy(sim, label="", builder=None):
    state = sim.context.getState(getEnergy=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    import math
    ok = not (math.isnan(e) or math.isinf(e))
    print(f"  E {label}: {e:.2f} kJ/mol  {'OK' if ok else '<<< NaN/Inf !!!'}")
    if not ok and builder is not None:
        # Diagnose bond lengths via PBC (matches FENE with PBC)
        pos = sim.context.getState(getPositions=True)\
              .getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        bv  = sim.context.getState(getPositions=True)\
              .getPeriodicBoxVectors(asNumpy=True)
        L   = float(bv[0][0].value_in_unit(unit.nanometer))
        R0  = 1.5
        bad = []
        for idx, (i, j) in enumerate(builder.all_bonds):
            dr = pos[i] - pos[j]
            dr -= L * np.round(dr / L)
            r = float(np.linalg.norm(dr))
            if r >= R0:
                bad.append((idx, i, j, r))
        print(f"  Bonds >= R0 (PBC): {len(bad)}/{len(builder.all_bonds)}")
        for idx, i, j, r in bad[:5]:
            print(f"    bond {idx}: ({i},{j}) r_pbc={r:.3f}")
    return ok, e


def measure_lambda(sim, N, L, n_steps=80000, save_every=200, dt=0.006):
    """Measure cross-link localization length from MSD plateau."""
    from src.utils import _unwrap
    positions_log = []
    for _ in range(n_steps // save_every):
        sim.step(save_every)
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        positions_log.append(pos)
    positions_log = np.array(positions_log)
    unwrapped = _unwrap(positions_log, L)
    dr = unwrapped - unwrapped[0:1]
    msd = np.mean(np.sum(dr**2, axis=-1), axis=-1)
    lam = float(np.sqrt(np.mean(msd[3*len(msd)//4:])))
    return lam, msd





def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    sys_cfg = cfg["system"]
    sim_cfg = cfg["simulation"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    N_m      = sys_cfg["N_m"]
    rho      = sys_cfg["rho_m0"]
    c        = sys_cfg["c"]
    mean_strand = sys_cfg["mean_strand"]
    seed     = sys_cfg.get("seed", 42)
    T        = sim_cfg["T"]
    dt       = sim_cfg["dt"]
    gamma_m  = sim_cfg["gamma_m"]
    platform = sim_cfg["platform"]
    n_equil  = sim_cfg["n_equil_network"]

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
    N   = builder.N_m   # actual N after pruning
    L   = builder.L
    rho_eff = N / L**3
    print(f"  rho_eff (after pruning) = {rho_eff:.4f}  "
          f"(target: {rho}, N_pruned: {N_m - N} beads removed)")
    print(f"      Done in {time.time()-t0:.1f} s")

    N   = builder.N_m   # may be less than N_m after pruning
    L   = builder.L
    pos = builder.positions.copy()   # wrapped lattice positions

    # Verify no bonds exceed R0 without PBC
    bl = builder._bond_lengths()
    n_bad = int((bl >= 1.5).sum())
    print(f"\n  Bond check: min={bl.min():.4f}  max={bl.max():.4f}  "
          f"bonds>=R0: {n_bad}  ← should be 0")
    if n_bad > 0:
        print("  WARNING: some bonds still >= R0. Attempting to fix...")
        # Last resort: nudge atoms toward each other
        for idx, (i, j) in enumerate(builder.all_bonds):
            dr = pos[i] - pos[j]
            r = np.linalg.norm(dr)
            if r >= 1.5:
                pos[i] = pos[i] - 0.4 * dr / r
                pos[j] = pos[j] + 0.4 * dr / r

    # ------------------------------------------------------------------
    # 2. Build OpenMM system
    # ------------------------------------------------------------------
    print("\n[2/5] Building OpenMM system ...")
    # FENE uses PBC (setUsesPeriodicBoundaryConditions=True)
    # so minimum image is applied automatically.
    sim, system, fene = make_simulation(
        N, L, builder.all_bonds, gamma_m, T, dt, platform
    )
    sim.context.setPositions([mm.Vec3(*p) for p in pos])

    # ------------------------------------------------------------------
    # 3. Minimization
    # ------------------------------------------------------------------
    print("\n[3/5] Energy minimization ...")
    sim.context.setPositions([mm.Vec3(*p) for p in pos])
    ok, _ = check_energy(sim, "before minimization", builder)

    sim.minimizeEnergy(maxIterations=10000)
    ok, _ = check_energy(sim, "after minimization", builder)
    if not ok:
        raise RuntimeError("Energy is NaN after minimization. Check topology.")

    sim.context.setVelocitiesToTemperature(T)

    # ------------------------------------------------------------------
    # 4. NVT equilibration — staged dt ramp
    # ------------------------------------------------------------------
    print(f"\n[4/5] NVT equilibration ({n_equil} steps) ...")

    # Ramp: (dt, T_ramp, n_steps)
    ramp = [
        (0.001, 0.3,  20000),
        (0.002, 0.6,  20000),
        (0.004, 0.8,  30000),
        (dt,    T,    n_equil),
    ]
    state = sim.context.getState(getPositions=True, getVelocities=True)
    for dt_r, T_r, n_r in ramp:
        integ = mm.LangevinMiddleIntegrator(T_r, gamma_m, dt_r)
        integ.setRandomNumberSeed(seed)
        try:
            platform_obj = mm.Platform.getPlatformByName(platform)
            props = {"CudaPrecision": "mixed"} if platform == "CUDA" else {}
            sim_r = app.Simulation(dummy_topology(N), system, integ,
                                   platform_obj, props)
        except Exception:
            sim_r = app.Simulation(dummy_topology(N), system, integ,
                                   mm.Platform.getPlatformByName("CPU"))
        sim_r.context.setState(state)
        sim_r.context.setVelocitiesToTemperature(T_r)

        reporter = app.StateDataReporter(
            os.path.join(out_dir, f"network_equil_dt{dt_r:.3f}.log"),
            max(1000, n_r // 20),
            step=True, potentialEnergy=True, temperature=True, speed=True
        )
        sim_r.reporters.append(reporter)
        sim_r.step(n_r)
        ok, _ = check_energy(sim_r, f"dt={dt_r} T={T_r}")
        if not ok:
            raise RuntimeError(f"NaN at dt={dt_r}, T={T_r}")
        state = sim_r.context.getState(getPositions=True, getVelocities=True)

    print("      NVT equilibration done")

    # Use final sim_r for λ measurement
    # ------------------------------------------------------------------
    # 5. Measure λ
    # ------------------------------------------------------------------
    print("\n[5/5] Measuring λ ...")
    lam, msd_cl = measure_lambda(sim_r, N, L, dt=dt)
    lam_ref = LAMBDA_REF.get(rho)
    print(f"  λ measured = {lam:.4f}")
    print(f"  λ Paper 1  = {lam_ref}  (rho={rho})")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    final_pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    np.savez(
        os.path.join(out_dir, "network_data.npz"),
        positions        = final_pos,
        bonds            = np.array(builder.all_bonds),
        backbone_bonds   = np.array(builder.backbone_bonds),
        crosslink_bonds  = np.array(builder.crosslink_bonds),
        crosslink_ids    = np.array(builder.crosslink_ids),
        lambda_measured  = lam,
        lambda_paper1    = lam_ref or 0.0,
        msd_crosslink    = msd_cl,
        L                = L,
        N_m              = N,
        rho              = rho,
    )
    sim_r.saveCheckpoint(os.path.join(out_dir, "network_equilibrated.chk"))
    print(f"\n  Saved to {out_dir}/")
    print("  DONE — ready for 02_embed_enzymes.py")


if __name__ == "__main__":
    main()
