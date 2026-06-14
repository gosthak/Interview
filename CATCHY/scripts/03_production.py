#!/usr/bin/env python3
"""
03_production.py — Production run with enzymatic bond cleavage.

For each sigma_E × {active, passive}:
  1. Load enzyme_system checkpoint
  2. Run production in blocks of save_interval steps
  3. At each block end:
       - Write trajectory frame (positions + bond_status) to HDF5
       - Attempt cleavage (active runs only)
       - Compute running MSD of enzymes
  4. Save msd_{label}.npz + survival_{label}.npz

Usage:
    python 03_production.py --config ../configs/default.yaml [--sigma_E 3.0] [--mode active]
"""

import argparse
import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config, confinement_parameter, HDF5Writer, _unwrap
from src.network_builder import NetworkBuilder
from src.enzyme_system import EnzymeSystem
from src.cleavage import CleavageManager

try:
    import openmm as mm
except ImportError:
    raise ImportError("OpenMM required.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../configs/default.yaml")
    p.add_argument("--sigma_E", type=float, default=None,
                   help="Run only this sigma_E value (default: all in config)")
    p.add_argument("--mode", choices=["active", "passive", "both"],
                   default="both")
    return p.parse_args()


def run_production(enz_sys, cleavage_mgr, n_production, save_interval,
                   out_dir, label, dt):
    """
    Core production loop.

    Parameters
    ----------
    enz_sys : EnzymeSystem
    cleavage_mgr : CleavageManager or None (passive mode)
    n_production : int
    save_interval : int
    label : str
    """
    N_total = enz_sys.N_m + enz_sys.N_E
    N_cl = len(cleavage_mgr.cl_indices) if cleavage_mgr else 0
    L = enz_sys.L

    # HDF5 writer
    traj_path = os.path.join(out_dir, f"traj_{label}.h5")
    k_cat_val = cleavage_mgr.k_cat if cleavage_mgr else 0.0
    writer = HDF5Writer(
        traj_path, N_total, enz_sys.N_m, enz_sys.N_E, N_cl,
        sigma_E=enz_sys.sigma_E,
        rho=enz_sys.network.rho,
        k_cat=k_cat_val,
        L=L
    )

    # DCD reporter for VMD visualization
    dcd_path = os.path.join(out_dir, f"traj_{label}.dcd")
    enz_sys.simulation.reporters.append(
        mm.app.DCDReporter(dcd_path, save_interval)
    )

    n_blocks = n_production // save_interval
    enzyme_positions_log = []   # for MSD computation
    survival_log = []
    step_log = []
    time_log = []

    print(f"  Running {n_blocks} blocks × {save_interval} steps ...")
    t_wall = time.time()

    for block in range(n_blocks):
        enz_sys.run(save_interval)

        current_step = (block + 1) * save_interval
        current_time = current_step * dt

        # Get state
        state = enz_sys.simulation.context.getState(getPositions=True)
        pos = np.array(state.getPositions(asNumpy=True))   # (N_total, 3)

        # Bond status
        if cleavage_mgr:
            bond_status = cleavage_mgr.get_bond_status()
        else:
            bond_status = np.ones(N_cl, dtype=np.int8)

        writer.write_frame(pos, bond_status, current_step, current_time)

        # Store enzyme positions for MSD
        enzyme_positions_log.append(pos[enz_sys.N_m:enz_sys.N_m + enz_sys.N_E].copy())

        # Attempt cleavage (active only)
        if cleavage_mgr:
            n_new = cleavage_mgr.attempt_cleavage(
                enz_sys.simulation.context, current_step
            )
            survival_log.append(cleavage_mgr.survival_fraction)
        else:
            survival_log.append(1.0)

        step_log.append(current_step)
        time_log.append(current_time)

        # Progress report every 10%
        if (block + 1) % max(1, n_blocks // 10) == 0:
            elapsed = time.time() - t_wall
            eta = elapsed / (block + 1) * (n_blocks - block - 1)
            S = survival_log[-1]
            print(f"    block {block+1:5d}/{n_blocks}  "
                  f"t={current_time:.0f}  S={S:.3f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    writer.close()

    # ------------------------------------------------------------------
    # Compute enzyme MSD
    # ------------------------------------------------------------------
    enzyme_positions_log = np.array(enzyme_positions_log)  # (n_frames, N_E, 3)
    n_frames = len(enzyme_positions_log)
    max_lag = n_frames // 2

    lag_steps, msd = _compute_msd_from_log(enzyme_positions_log, L, max_lag)
    time_axis = lag_steps * save_interval * dt

    # Diffusion coefficient from long-time slope (last quarter of MSD)
    D = _fit_diffusion(time_axis, msd)

    np.savez(
        os.path.join(out_dir, f"msd_{label}.npz"),
        time=time_axis,
        msd=msd,
        lag_steps=lag_steps,
        D=D,
        sigma_E=enz_sys.sigma_E,
    )
    np.savez(
        os.path.join(out_dir, f"survival_{label}.npz"),
        step=np.array(step_log),
        time=np.array(time_log),
        survival=np.array(survival_log),
        sigma_E=enz_sys.sigma_E,
    )

    print(f"  D = {D:.4e}   S_final = {survival_log[-1]:.3f}")
    return D, survival_log[-1]


def _compute_msd_from_log(positions, L, max_lag):
    """positions: (n_frames, N_E, 3)."""
    n_frames, N_E, _ = positions.shape

    # Unwrap
    unwrapped = positions.copy().astype(float)
    for t in range(1, n_frames):
        delta = unwrapped[t] - unwrapped[t - 1]
        delta -= L * np.round(delta / L)
        unwrapped[t] = unwrapped[t - 1] + delta

    msd = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        dr = unwrapped[lag:] - unwrapped[:-lag]
        msd[lag - 1] = np.mean(np.sum(dr ** 2, axis=-1))

    return np.arange(1, max_lag + 1), msd


def _fit_diffusion(time_axis, msd):
    """Fit MSD = 6 D t in the last quarter of the trajectory."""
    n = len(msd)
    idx = slice(3 * n // 4, n)
    t = time_axis[idx]
    m = msd[idx]
    if len(t) < 2:
        return np.nan
    coeffs = np.polyfit(t, m, 1)
    D = coeffs[0] / 6.0
    return D


# ---------------------------------------------------------------------------

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
    n_production = sim_cfg["n_production"]
    save_interval = sim_cfg["save_interval"]

    sigma_list = enz_cfg["sigma_list"]
    if args.sigma_E is not None:
        sigma_list = [args.sigma_E]

    phi_E = enz_cfg["phi_E"]
    interaction = enz_cfg["interaction"]
    k_cat = enz_cfg["k_cat"]
    r_cleave = enz_cfg["r_cleave"]

    modes = {"active": True, "passive": False}
    if args.mode == "active":
        modes = {"active": True}
    elif args.mode == "passive":
        modes = {"passive": False}

    # Load network topology from file — same as step 2
    net_data = np.load(os.path.join(out_dir, "network_data.npz"), allow_pickle=True)
    L        = float(net_data["L"])
    N_m_actual = int(net_data["N_m"])

    builder = NetworkBuilder.__new__(NetworkBuilder)
    builder.N_m           = N_m_actual
    builder.L             = L
    builder.rho           = rho
    builder.c             = c
    builder.mean_strand   = mean_strand
    builder.positions     = net_data["positions"]
    builder.backbone_bonds  = [tuple(b) for b in net_data["backbone_bonds"]]
    builder.crosslink_bonds = [tuple(b) for b in net_data["crosslink_bonds"]]
    builder.crosslink_ids   = list(net_data["crosslink_ids"])
    builder._cl_set       = set(builder.crosslink_ids)
    builder._degree       = None
    builder._update_degree = lambda: None

    summary_rows = []

    for sigma_E in sigma_list:
        C, lam = confinement_parameter(sigma_E, rho)
        base_label = f"sigma{sigma_E:.1f}_{interaction}"

        for mode_name, is_active in modes.items():
            label = f"{base_label}_{mode_name}"
            print("\n" + "=" * 60)
            print(f"CATCHY Production: {label}  (C={C:.2f}, λ={lam:.3f})")
            print("=" * 60)

            # Build system
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

            # Load equilibrated checkpoint
            chk_path = os.path.join(out_dir, f"enzyme_system_{base_label}.chk")
            if not os.path.exists(chk_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: {chk_path}. "
                    "Run 02_embed_enzymes.py first."
                )
            enz_sys.load_checkpoint(chk_path)

            # Save PDB topology for VMD (needed to load DCD)
            pdb_path = os.path.join(out_dir, f"topology_{base_label}.pdb")
            if not os.path.exists(pdb_path):
                state = enz_sys.simulation.context.getState(getPositions=True)
                import openmm.unit as unit
                pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                # Wrap all positions into box [0, L)
                L_nm = enz_sys.L
                pos_wrapped = pos_nm % L_nm
                # Convert to OpenMM Quantity in nm for PDBFile
                pos_quantity = pos_wrapped * unit.nanometer
                mm.app.PDBFile.writeFile(
                    enz_sys.simulation.topology,
                    pos_quantity,
                    open(pdb_path, "w")
                )
                # Prepend CRYST1 record (box dimensions in Å)
                L_ang = L_nm * 10.0
                cryst1 = (
                    f"CRYST1{L_ang:9.3f}{L_ang:9.3f}{L_ang:9.3f}"
                    f"  90.00  90.00  90.00 P 1           1\n"
                )
                pdb_content = open(pdb_path).read()
                with open(pdb_path, "w") as pdb_f:
                    pdb_f.write(cryst1 + pdb_content)
                print(f"  Topology saved → {pdb_path}")

            # Set up cleavage manager
            if is_active:
                cleave_mgr = CleavageManager(
                    fene_force=enz_sys.fene_force,
                    crosslink_bond_indices=enz_sys.crosslink_bond_indices,
                    N_m=N_m, L=L,
                    k_cat=k_cat, r_cleave=r_cleave,
                    sigma_E=sigma_E, dt=dt,
                )
            else:
                cleave_mgr = None

            D, S_final = run_production(
                enz_sys, cleave_mgr, n_production, save_interval,
                out_dir, label, dt
            )
            summary_rows.append({
                "sigma_E": sigma_E, "C": C, "mode": mode_name,
                "D": D, "S_final": S_final
            })

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"{'sigma_E':>8} {'C':>6} {'mode':>8} {'D':>12} {'S_final':>8}")
    print("-" * 60)
    for r in summary_rows:
        print(f"{r['sigma_E']:8.1f} {r['C']:6.2f} {r['mode']:>8} "
              f"{r['D']:12.4e} {r['S_final']:8.3f}")
    print("=" * 60)
    print("  DONE — run analysis/plot_all.py to generate figures")


if __name__ == "__main__":
    main()
