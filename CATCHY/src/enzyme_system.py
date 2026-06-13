"""
EnzymeSystem — assembles the full OpenMM system for CATCHY simulations.

Particle layout:
    0 .. N_m-1           : monomers (backbone + cross-link sites)
    N_m .. N_m+N_E-1     : enzymes

Mass/friction scaling (Paper 1, Section II.B):
    m_E   = rho_m * (pi/6) * sigma_E^3   (same mass density as monomers)
    gamma_E = gamma_m * (sigma_m/sigma_E)^2  (same surface friction)
"""

import math
import numpy as np
from numpy.random import default_rng

try:
    import openmm as mm
    import openmm.unit as unit
    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat
    from openmm.app import Simulation, StateDataReporter, DCDReporter
except ImportError:
    raise ImportError("OpenMM not found. Install with: conda install -c conda-forge openmm")

from .potentials import (
    add_wca_monomers_only,
    add_fene,
    add_expanded_lj,
    add_enzyme_enzyme_wca,
)

SIGMA_M = 1.0
EPSILON_M = 1.0
FENE_K = 30.0
FENE_R0 = 1.5


class EnzymeSystem:
    """
    Build and manage the OpenMM simulation system.

    Parameters
    ----------
    network : NetworkBuilder  (already built)
    sigma_E : float   enzyme diameter
    phi_E   : float   enzyme volume fraction
    attractive : bool  True → ANP-like, False → RNP-like
    T, dt, gamma_m : thermodynamic / integration parameters
    platform_name : 'CUDA', 'OpenCL', or 'CPU'
    seed : int
    """

    def __init__(self, network, sigma_E, phi_E=0.02,
                 attractive=True, T=1.0, dt=0.006, gamma_m=0.1,
                 platform_name="CUDA", seed=0):
        self.network = network
        self.sigma_E = sigma_E
        self.phi_E = phi_E
        self.attractive = attractive
        self.T = T
        self.dt = dt
        self.gamma_m = gamma_m
        self.platform_name = platform_name
        self.rng = default_rng(seed)

        self.N_m = network.N_m
        self.L = network.L
        self.N_E = self._compute_N_enzymes()

        self.system = None
        self.simulation = None
        self.fene_force = None        # reference kept for CleavageManager
        self.crosslink_bond_indices = None   # positions in fene_force bond list

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self):
        """Assemble OpenMM System and Simulation."""
        self.system = mm.System()
        self._add_box()
        self._add_particles()
        self._add_forces()
        self._build_simulation()
        return self

    def set_positions(self, positions):
        """Set positions (nm or openmm Quantity). Wraps into box."""
        self.simulation.context.setPositions(positions)

    def get_positions(self):
        state = self.simulation.context.getState(getPositions=True)
        return state.getPositions(asNumpy=True)

    def minimize(self, max_iter=1000):
        self.simulation.minimizeEnergy(maxIterations=max_iter)

    def run(self, n_steps):
        self.simulation.step(n_steps)

    def get_state(self):
        return self.simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )

    def save_checkpoint(self, path):
        self.simulation.saveCheckpoint(path)

    def load_checkpoint(self, path):
        self.simulation.loadCheckpoint(path)

    def save_xml(self, path):
        """Save full system state as XML."""
        import openmm.app as app
        state = self.simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        with open(path, "w") as f:
            f.write(mm.XmlSerializer.serialize(state))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_N_enzymes(self):
        V = self.L ** 3
        v_E = (math.pi / 6.0) * self.sigma_E ** 3
        N_E = max(1, int(self.phi_E * V / v_E))
        return N_E

    def _add_box(self):
        L_nm = self.L * 0.1   # LJ units → nm (1 σ_m ≈ 0.1 nm conceptually,
        # but OpenMM works in nm; we keep LJ units by setting σ_m = 1 nm)
        # Here we use 1 LJ unit = 1 nm for simplicity
        self.system.setDefaultPeriodicBoxVectors(
            mm.Vec3(self.L, 0, 0),
            mm.Vec3(0, self.L, 0),
            mm.Vec3(0, 0, self.L),
        )

    def _add_particles(self):
        """Add monomers then enzymes to system."""
        # Monomer mass = 1.0 (LJ units, Paper 1)
        for _ in range(self.N_m):
            self.system.addParticle(1.0)

        # Enzyme mass: same mass density as monomers
        rho_m_mass = 1.0 / ((math.pi / 6.0) * SIGMA_M ** 3)
        m_E = rho_m_mass * (math.pi / 6.0) * self.sigma_E ** 3
        for _ in range(self.N_E):
            self.system.addParticle(m_E)

    def _add_forces(self):
        N = self.N_m + self.N_E
        monomer_idx = list(range(self.N_m))
        enzyme_idx = list(range(self.N_m, self.N_m + self.N_E))

        # 1. WCA between monomers — must register N_total particles
        add_wca_monomers_only(self.system, self.N_m,
                              n_total=self.N_m + self.N_E)

        # 2. FENE bonds (backbone + cross-links)
        all_bonds = self.network.all_bonds
        self.fene_force = add_fene(self.system, all_bonds)
        # Record cross-link bond start index for CleavageManager
        self.crosslink_bond_indices = list(
            range(len(self.network.backbone_bonds), len(all_bonds))
        )

        # 3. Expanded LJ — enzyme-monomer interaction (Paper 1 eq. 4)
        self.elj_force = add_expanded_lj(
            self.system, enzyme_idx, monomer_idx,
            self.sigma_E, attractive=self.attractive
        )

        # 4. WCA — enzyme-enzyme
        if self.N_E > 1:
            add_enzyme_enzyme_wca(self.system, enzyme_idx, self.sigma_E)

    def _build_simulation(self):
        """Build Langevin integrator and Simulation."""
        # Friction for enzymes: gamma_E = gamma_m * (sigma_m/sigma_E)^2
        # OpenMM LangevinMiddleIntegrator uses one friction for all particles.
        # We use a CustomIntegrator to allow per-particle friction.
        integrator = self._build_per_particle_langevin()

        try:
            platform = mm.Platform.getPlatformByName(self.platform_name)
            properties = {}
            if self.platform_name == "CUDA":
                properties = {"CudaPrecision": "mixed"}
            self.simulation = mm.app.Simulation(
                _dummy_topology(self.N_m + self.N_E),
                self.system, integrator, platform, properties
            )
        except Exception:
            print(f"[WARNING] Platform {self.platform_name} not available, falling back to CPU")
            platform = mm.Platform.getPlatformByName("CPU")
            self.simulation = mm.app.Simulation(
                _dummy_topology(self.N_m + self.N_E),
                self.system, integrator, platform
            )

    def _build_per_particle_langevin(self):
        """
        CustomIntegrator implementing per-particle friction LJ-Langevin dynamics.
        gamma_m for monomers, gamma_E = gamma_m*(sigma_m/sigma_E)^2 for enzymes.
        Uses BAOAB (LangevinMiddle) splitting.
        """
        kT = self.T   # k_B = 1 in LJ units
        gamma_E = self.gamma_m * (SIGMA_M / self.sigma_E) ** 2

        integrator = mm.CustomIntegrator(self.dt)
        integrator.addGlobalVariable("kT", kT)
        integrator.addPerDofVariable("gamma", 0)
        integrator.addPerDofVariable("noise", 0)

        # Set per-dof friction
        integrator.addComputePerDof("gamma",
            f"select(step(particleIndex - {self.N_m - 0.5}), "
            f"{gamma_E:.8f}, {self.gamma_m:.8f})"
        )

        # BAOAB: B-A-O-A-B steps
        dt = self.dt
        # B: half-step velocity update from forces
        integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
        # A: half-step position update
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        # O: Ornstein-Uhlenbeck step
        integrator.addComputePerDof("noise", "gaussian")
        integrator.addComputePerDof(
            "v",
            "v*exp(-gamma*dt) + sqrt(kT/m*(1-exp(-2*gamma*dt)))*noise"
        )
        # A: second half-step position
        integrator.addComputePerDof("x", "x + 0.5*dt*v")
        integrator.addUpdateContextState()
        # B: second half-step velocity from new forces
        integrator.addComputePerDof("v", "v + 0.5*dt*f/m")

        return integrator

    def get_enzyme_positions(self):
        """Return enzyme positions as numpy array (N_E, 3)."""
        pos = self.get_positions()
        return np.array(pos[self.N_m:self.N_m + self.N_E])

    def get_monomer_positions(self):
        """Return monomer positions as numpy array (N_m, 3)."""
        pos = self.get_positions()
        return np.array(pos[:self.N_m])

    def embed_enzymes(self):
        """
        Place enzymes at random positions avoiding monomer overlaps.
        Uses a simple rejection sampler.
        """
        mono_pos = self.get_monomer_positions()
        enzyme_positions = []
        max_attempts = 100000
        min_dist = 0.5 * (self.sigma_E + SIGMA_M)

        for ie in range(self.N_E):
            for _ in range(max_attempts):
                trial = self.rng.uniform(0, self.L, 3)
                # Check against monomers
                dr = mono_pos - trial
                dr -= self.L * np.round(dr / self.L)
                dists = np.linalg.norm(dr, axis=1)
                if np.all(dists > min_dist):
                    # Check against already placed enzymes
                    ok = True
                    for ep in enzyme_positions:
                        d = np.linalg.norm(
                            (trial - ep) - self.L * np.round((trial - ep) / self.L)
                        )
                        if d < self.sigma_E:
                            ok = False
                            break
                    if ok:
                        enzyme_positions.append(trial)
                        break
            else:
                raise RuntimeError(
                    f"Could not place enzyme {ie} after {max_attempts} attempts. "
                    "Try reducing phi_E or sigma_E."
                )

        all_pos = list(self.get_positions())
        for ie, ep in enumerate(enzyme_positions):
            all_pos[self.N_m + ie] = mm.Vec3(*ep)
        self.simulation.context.setPositions(all_pos)


# ---------------------------------------------------------------------------
# Helper: minimal OpenMM Topology for a system with no chemistry
# ---------------------------------------------------------------------------

def _dummy_topology(N_total):
    """Create a minimal Topology with N_total particles (no bonds, one chain)."""
    import openmm.app as app
    topo = app.Topology()
    chain = topo.addChain()
    res = topo.addResidue("SYS", chain)
    for i in range(N_total):
        topo.addAtom(f"P{i}", app.element.carbon, res)
    return topo
