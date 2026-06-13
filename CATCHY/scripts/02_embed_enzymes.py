#!/usr/bin/env python3
"""
02_embed_enzymes.py — Insert enzymes into equilibrated network and equilibrate.

For each sigma_E in config:
  For each mode in {active, passive}:
    1. Load equilibrated network checkpoint
    2. Add enzyme particles at random non-overlapping positions
    3. Soft push-off (scaled epsilon) to relax overlaps
    4. NPT equilibration
    5. NVT equilibration
    6. Save enzyme_system_{label}.xml checkpoint

Usage:
    python 02_embed_enzymes.py --config ../configs/default.yaml
"""

import argparse
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, confinement_parameter
from src.network_builder import NetworkBuilder
from src.enzyme_system import EnzymeSystem

try:
    import openmm as mm
except ImportError:
    raise ImportError("OpenMM required.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    sys_cfg = cfg["system"]
    enz_cfg = cfg["enzyme"]
    sim_cfg = cfg["simulation"]
    out_dir = cfg["output"]["dir"]

    N_m = sys_cfg["N_m"]
    rho = sys_cfg["rho_m0"]
    c = sys_cfg["c"]
    mean_strand = sys_cfg["mean_strand"]
    seed = sys_cfg.get("seed", 42)
    T = sim_cfg["T"]
    dt = sim_cfg["dt"]
    gamma_m = sim_cfg["gamma_m"]
    platform = sim_cfg["platform"]
    n_equil_enz = sim_cfg["n_equil_enzyme"]

    sigma_list = enz_cfg["sigma_list"]
    phi_E = enz_cfg["phi_E"]
    interaction = enz_cfg["interaction"]
    k_cat = enz_cfg["k_cat"]

    # Load saved network data
    net_path = os.path.join(out_dir, "network_data.npz")
    if not os.path.exists(net_path):
        raise FileNotFoundError(
            f"Network data not found at {net_path}. "
            "Run 01_build_network.py first."
        )
    net_data = np.load(net_path, allow_pickle=True)
    lam = float(net_data["lambda_paper1"])
    L   = float(net_data["L"])
    N_m_actual = int(net_data["N_m"])   # actual N after pruning in step 1

    # Load topology directly from saved data — no need to rebuild NetworkBuilder.
    # Step 1 already saved the exact bonds and positions after pruning.
    from src.network_builder import NetworkBuilder
    builder = NetworkBuilder.__new__(NetworkBuilder)
    builder.N_m          = N_m_actual
    builder.L            = L
    builder.rho          = rho
    builder.c            = c
    builder.mean_strand  = mean_strand
    builder.positions    = net_data["positions"]          # equilibrated positions
    builder.backbone_bonds  = [tuple(b) for b in net_data["backbone_bonds"]]
    builder.crosslink_bonds = [tuple(b) for b in net_data["crosslink_bonds"]]
    builder.crosslink_ids   = list(net_data["crosslink_ids"])
    builder._cl_set      = set(builder.crosslink_ids)
    builder._degree      = None
    builder._update_degree = lambda: None   # not needed here

    print("=" * 60)
    print("CATCHY — Step 02: Embed enzymes")
    print(f"  sigma_E values: {sigma_list}")
    print(f"  interaction: {interaction}, phi_E: {phi_E}")
    print("=" * 60)

    for sigma_E in sigma_list:
        C, _ = confinement_parameter(sigma_E, rho)
        label = f"sigma{sigma_E:.1f}_{interaction}"
        print(f"\n--- sigma_E={sigma_E:.1f}, C={C:.2f} --- [{label}]")

        t0 = time.time()
        enz_sys = EnzymeSystem(
            network=builder,
            sigma_E=sigma_E,
            phi_E=phi_E,
            attractive=(interaction == "attractive"),
            T=T, dt=dt, gamma_m=gamma_m,
            platform_name=platform,
            seed=seed + int(sigma_E * 100),
        )
        enz_sys.build()

        # Set network positions
        all_pos = [mm.Vec3(*p) for p in builder.positions]
        # Placeholder enzyme positions (will be overwritten by embed_enzymes)
        for _ in range(enz_sys.N_E):
            all_pos.append(mm.Vec3(0, 0, 0))
        enz_sys.simulation.context.setPositions(all_pos)

        print(f"  Embedding {enz_sys.N_E} enzymes ...")
        enz_sys.embed_enzymes()

        # Load equilibrated network velocities
        enz_sys.simulation.context.setVelocitiesToTemperature(T)

        # ------------------------------------------------------------------
        # Soft push-off: energy minimization to remove enzyme overlaps
        # ------------------------------------------------------------------
        print("  Soft push-off minimization ...")
        enz_sys.minimize(max_iter=5000)
        enz_sys.simulation.context.setVelocitiesToTemperature(T)

        # ------------------------------------------------------------------
        # NPT equilibration — let the box relax back to P*=0 after enzyme
        # insertion. Enzymes add volume that pushes pressure up; NPT
        # restores the network to its target density rho_m0.
        # We use a short NPT (~50k steps) then switch to NVT for production.
        # ------------------------------------------------------------------
        n_npt_enz = sim_cfg.get("n_npt_enzyme", 50000)
        print(f"  NPT equilibration ({n_npt_enz} steps, P*=0) ...")

        # MonteCarloBarostat at P=0 bar (= P*=0 in our LJ mapping)
        barostat = mm.MonteCarloBarostat(0.0, T, 25)
        enz_sys.system.addForce(barostat)
        enz_sys.simulation.context.reinitialize(preserveState=True)

        reporter_npt = mm.app.StateDataReporter(
            os.path.join(out_dir, f"npt_{label}.log"), 5000,
            step=True, potentialEnergy=True, volume=True, density=True
        )
        enz_sys.simulation.reporters.append(reporter_npt)
        enz_sys.run(n_npt_enz)
        enz_sys.simulation.reporters.clear()

        # Remove barostat — switch to NVT for the rest
        enz_sys.system.removeForce(enz_sys.system.getNumForces() - 1)
        enz_sys.simulation.context.reinitialize(preserveState=True)

        # ------------------------------------------------------------------
        # NVT equilibration with enzymes (fixed box from here on)
        # ------------------------------------------------------------------
        print(f"  NVT equilibration ({n_equil_enz} steps) ...")
        reporter = mm.app.StateDataReporter(
            os.path.join(out_dir, f"equil_{label}.log"), 10000,
            step=True, potentialEnergy=True, temperature=True, speed=True
        )
        enz_sys.simulation.reporters.append(reporter)
        enz_sys.run(n_equil_enz)
        enz_sys.simulation.reporters.clear()

        # ------------------------------------------------------------------
        # Save checkpoint
        # ------------------------------------------------------------------
        chk_path = os.path.join(out_dir, f"enzyme_system_{label}.chk")
        enz_sys.save_checkpoint(chk_path)
        print(f"  Saved checkpoint: {chk_path}  ({time.time()-t0:.0f} s)")

    print("\n  DONE — ready for 03_production.py")


if __name__ == "__main__":
    main()
