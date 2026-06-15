"""
Polymer network builder — Sorichetti et al. 2021 (Paper 1).

Topology rules:
  - Cross-link beads:  valence 3 (trivalent)
  - Backbone monomers: valence 2 (bivalent)
  - No dangling ends
  - ALL bond lengths < R0 = 1.5 σ_m  (FENE constraint)

Algorithm
─────────
1. Place N_m beads on a cubic lattice with spacing a < R0.
2. Select N_cl = c*N_m cross-links randomly.
3. Build chains by walking the lattice:
     - Start each chain from a cross-link.
     - Extend greedily to the nearest unvisited bead until
       another cross-link is reached or the chain ends.
     - Each cross-link fires 3 chains; each bivalent bead
       is visited exactly once (valence 2).
4. Prune dangling ends.
5. Verify: no bond >= R0, no degree-1 beads.
"""

import numpy as np
from numpy.random import default_rng
from collections import defaultdict

FENE_R0 = 1.5


class NetworkBuilder:
    def __init__(self, N_m=8000, rho=0.290, c=0.1, mean_strand=6, seed=42):
        self.N_m = N_m
        self.rho = rho
        self.c = c
        self.mean_strand = mean_strand
        self.rng = default_rng(seed)
        self.L = (N_m / rho) ** (1.0 / 3.0)

        self.positions = None
        self.backbone_bonds = []
        self.crosslink_bonds = []
        self.crosslink_ids = []
        self._degree = None
        self._cl_set = set()

    def build(self):
        self._place_beads()
        self._assign_crosslinks()
        self._build_topology()
        self._prune_dangling()
        self._verify()
        return self

    @property
    def all_bonds(self):
        return self.backbone_bonds + self.crosslink_bonds

    @property
    def bonds(self):
        return self.all_bonds

    def summary(self):
        deg = self._degree
        bl = self._bond_lengths()
        print("NetworkBuilder summary")
        print(f"  N_m              = {self.N_m}")
        print(f"  rho              = {self.rho:.3f}   L = {self.L:.3f}")
        print(f"  c                = {self.c:.3f}   N_cl = {len(self.crosslink_ids)}")
        print(f"  mean_strand <n>  = {self.mean_strand}")
        print(f"  backbone bonds   = {len(self.backbone_bonds)}")
        print(f"  cross-link bonds = {len(self.crosslink_bonds)}")
        print(f"  total bonds      = {len(self.all_bonds)}")
        if deg is not None:
            print(f"  degree-2 beads   = {int((deg==2).sum())}  (backbone monomers)")
            print(f"  degree-3 beads   = {int((deg==3).sum())}  (cross-links)")
            print(f"  degree ≤1 beads  = {int((deg<=1).sum())}  ← should be 0")
        if bl is not None and len(bl) > 0:
            print(f"  bond lengths     = min={bl.min():.3f}  "
                  f"mean={bl.mean():.3f}  max={bl.max():.3f}  (R0={FENE_R0})")
            print(f"  bonds >= R0      = {int((bl >= FENE_R0).sum())}  ← should be 0")
        # Strand length distribution
        sl = self._strand_lengths()
        if sl is not None and len(sl) > 0:
            import numpy as np
            print(f"  strand lengths   = mean={np.mean(sl):.2f}  "
                  f"std={np.std(sl):.2f}  (target <n>={self.mean_strand})")

    # ------------------------------------------------------------------ #
    #  Step 1: lattice placement                                           #
    # ------------------------------------------------------------------ #

    def _place_beads(self):
        """
        Standard cubic lattice with spacing a = L / n_side.

        No wrapping tricks — points are strictly in [0, L) with
        spacing a on all three axes. This guarantees:
          - no duplicate positions under PBC
          - nearest-neighbour distance = a (uniform, no surprises)

        If a > R0 (very low density), the topology builder will still
        find neighbours because we search up to r_bond_max > a.
        The compression step in the script fixes any bonds > R0
        before dynamics starts.
        """
        n_side = int(np.ceil(self.N_m ** (1.0 / 3.0)))
        a = self.L / n_side   # lattice spacing

        pts = []
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if len(pts) >= self.N_m:
                        break
                    pts.append([ix * a, iy * a, iz * a])

        pts = np.array(pts[:self.N_m], dtype=float)
        # Small jitter to break perfect symmetry — max 2% of spacing
        pts += self.rng.uniform(-0.02 * a, 0.02 * a, pts.shape)
        # Keep strictly inside box (jitter is tiny, this is a safeguard)
        self.positions = pts % self.L

    # ------------------------------------------------------------------ #
    #  Step 2: assign cross-links                                          #
    # ------------------------------------------------------------------ #

    def _assign_crosslinks(self):
        N_cl = max(4, int(round(self.c * self.N_m)))
        self.crosslink_ids = list(
            self.rng.choice(self.N_m, size=N_cl, replace=False)
        )
        self._cl_set = set(self.crosslink_ids)

    # ------------------------------------------------------------------ #
    #  Step 3: build topology                                              #
    # ------------------------------------------------------------------ #

    def _build_topology(self):
        """
        Build topology using wrapped lattice positions.
        FENE uses setUsesPeriodicBoundaryConditions(True) so minimum
        image is applied automatically — no unwrapping needed.
        """
        L   = self.L
        N   = self.N_m
        cl_set = self._cl_set
        r_max = FENE_R0 * 1.3

        nbrs = self._cell_list(r_max)

        valence_max = np.array([3 if i in cl_set else 2 for i in range(N)])
        valence_cur = np.zeros(N, dtype=int)
        bond_set = set()
        backbone_bonds = []
        crosslink_bonds = []

        def try_bond(i, j):
            if valence_cur[i] >= valence_max[i]: return False
            if valence_cur[j] >= valence_max[j]: return False
            if i in cl_set and j in cl_set:      return False
            key = (min(i,j), max(i,j))
            if key in bond_set:                  return False
            # PBC distance — FENE uses minimum image automatically
            dr = self.positions[i] - self.positions[j]
            dr -= L * np.round(dr / L)
            if np.linalg.norm(dr) >= FENE_R0:
                return False
            bond_set.add(key)
            valence_cur[i] += 1
            valence_cur[j] += 1
            if i in cl_set or j in cl_set:
                crosslink_bonds.append((i, j))
            else:
                backbone_bonds.append((i, j))
            return True

        def nearest_free_nbr(i, exclude=None):
            candidates = []
            for j in nbrs[i]:
                if j == i: continue
                if exclude is not None and j == exclude: continue
                if valence_cur[j] >= valence_max[j]: continue
                if i in cl_set and j in cl_set: continue
                key = (min(i,j), max(i,j))
                if key in bond_set: continue
                # Use original wrapped positions for neighbour search
                dr = self.positions[i] - self.positions[j]
                dr -= L * np.round(dr / L)
                candidates.append((np.linalg.norm(dr), j))
            if not candidates:
                return None
            return min(candidates)[1]

        cl_shuffled = list(self.crosslink_ids)
        self.rng.shuffle(cl_shuffled)

        for cl in cl_shuffled:
            while valence_cur[cl] < valence_max[cl]:
                n_target = max(1, int(self.rng.geometric(1.0 / self.mean_strand)))
                prev = cl
                cur = nearest_free_nbr(cl)
                if cur is None or not try_bond(cl, cur):
                    break
                for _ in range(n_target - 1):
                    if cur in cl_set:
                        break
                    nxt = nearest_free_nbr(cur, exclude=prev)
                    if nxt is None or not try_bond(cur, nxt):
                        break
                    prev, cur = cur, nxt
                if cur not in cl_set:
                    for j in nbrs[cur]:
                        if j in cl_set and valence_cur[j] < valence_max[j]:
                            try_bond(cur, j)
                            break

        for i in self.rng.permutation(N):
            while valence_cur[i] < valence_max[i]:
                j = nearest_free_nbr(i)
                if j is None or not try_bond(i, j):
                    break

        self.backbone_bonds = backbone_bonds
        self.crosslink_bonds = crosslink_bonds
        self._update_degree()

    def _cell_list(self, r_cut):
        """Return dict i -> [j, ...] of neighbours within r_cut (PBC)."""
        pos = self.positions
        L   = self.L
        N   = self.N_m

        n_cells = max(1, int(L / r_cut))
        cs = L / n_cells
        cell_idx = (pos / cs).astype(int) % n_cells
        cells = defaultdict(list)
        for i in range(N):
            cells[tuple(cell_idx[i])].append(i)

        r_cut2 = r_cut ** 2
        nbrs = defaultdict(list)
        for i in range(N):
            cx, cy, cz = cell_idx[i]
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for dz in (-1,0,1):
                        for j in cells[((cx+dx)%n_cells,
                                        (cy+dy)%n_cells,
                                        (cz+dz)%n_cells)]:
                            if j == i: continue
                            dr = pos[i] - pos[j]
                            dr -= L * np.round(dr / L)
                            if np.dot(dr, dr) < r_cut2:
                                nbrs[i].append(j)
        return nbrs

    def _unwrap_positions(self):
        """
        Unwrap positions so every bonded pair (i,j) satisfies
        |pos[i] - pos[j]| < R0 WITHOUT minimum image convention.

        Uses iterative BFS: after each pass check if any bonds
        still cross the boundary and repeat until all are fixed.
        """
        from collections import defaultdict, deque

        pos = self.positions.copy().astype(float)
        L   = self.L

        adj = defaultdict(list)
        for i, j in self.all_bonds:
            adj[i].append(j)
            adj[j].append(i)

        # Iterative BFS — repeat until no bond crosses boundary
        for iteration in range(20):
            visited = np.zeros(self.N_m, dtype=bool)
            changed = False

            for start in range(self.N_m):
                if visited[start]:
                    continue
                visited[start] = True
                queue = deque([start])
                while queue:
                    i = queue.popleft()
                    for j in adj[i]:
                        if visited[j]:
                            continue
                        visited[j] = True
                        dr = pos[j] - pos[i]
                        shift = np.round(dr / L)
                        if np.any(shift != 0):
                            pos[j] -= shift * L
                            changed = True
                        queue.append(j)

            if not changed:
                break

        self.positions = pos

        # Verify
        bad = [(i, j, float(np.linalg.norm(pos[i]-pos[j])))
               for i, j in self.all_bonds
               if np.linalg.norm(pos[i]-pos[j]) >= 1.5]
        if bad:
            import warnings
            warnings.warn(
                f"{len(bad)} bonds >= R0=1.5 after unwrap "
                f"(max={max(r for _,_,r in bad):.4f}).",
                stacklevel=2
            )

    

    def _prune_dangling(self):
        # Iteratively remove degree-1 (dangling) nodes
        changed = True
        while changed:
            changed = False
            self._update_degree()
            dangling = set(np.where(self._degree == 1)[0].tolist())
            if dangling:
                self.backbone_bonds = [
                    (u,v) for u,v in self.backbone_bonds
                    if u not in dangling and v not in dangling
                ]
                self.crosslink_bonds = [
                    (u,v) for u,v in self.crosslink_bonds
                    if u not in dangling and v not in dangling
                ]
                self.crosslink_ids = [
                    cl for cl in self.crosslink_ids if cl not in dangling
                ]
                self._cl_set -= dangling
                changed = True

        # Remove isolated (degree-0) beads: re-index everything
        self._update_degree()
        isolated = set(np.where(self._degree == 0)[0].tolist())
        if isolated:
            old2new = {}
            new_idx = 0
            for i in range(self.N_m):
                if i not in isolated:
                    old2new[i] = new_idx
                    new_idx += 1

            self.positions = self.positions[
                [i for i in range(self.N_m) if i not in isolated]
            ]
            self.backbone_bonds = [
                (old2new[u], old2new[v]) for u,v in self.backbone_bonds
            ]
            self.crosslink_bonds = [
                (old2new[u], old2new[v]) for u,v in self.crosslink_bonds
            ]
            self.crosslink_ids = [
                old2new[cl] for cl in self.crosslink_ids
                if cl not in isolated
            ]
            self._cl_set = set(self.crosslink_ids)
            self.N_m = new_idx

        self._update_degree()

    # ------------------------------------------------------------------ #
    #  Step 5: verify                                                      #
    # ------------------------------------------------------------------ #

    def _verify(self):
        self._update_degree()
        bl = self._bond_lengths()
        if bl is not None and len(bl) > 0:
            n_bad = int((bl >= FENE_R0).sum())
            if n_bad > 0:
                import warnings
                warnings.warn(
                    f"{n_bad} bonds >= R0={FENE_R0} "
                    f"(max={bl.max():.3f}). "
                    "Script will compress these before dynamics.",
                    stacklevel=2
                )
        deg = self._degree
        n_isolated = int((deg == 0).sum())
        if n_isolated > 0:
            pct = 100 * n_isolated / self.N_m
            import warnings
            warnings.warn(
                f"{n_isolated} isolated beads ({pct:.1f}%). "
                "Consider increasing N_m.",
                stacklevel=2
            )

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _update_degree(self):
        deg = np.zeros(self.N_m, dtype=int)
        for u, v in self.all_bonds:
            deg[u] += 1; deg[v] += 1
        self._degree = deg

    def _strand_lengths(self):
        """Compute lengths of backbone strands between crosslinks."""
        from collections import defaultdict
        import numpy as np
        if not self.backbone_bonds:
            return None
        adj = defaultdict(list)
        for u, v in self.backbone_bonds:
            adj[u].append(v)
            adj[v].append(u)
        cl_set = self._cl_set
        visited = set()
        strand_lengths = []
        for cl in self.crosslink_ids:
            for nbr in adj[cl]:
                if nbr in cl_set:
                    continue
                key = (min(cl, nbr), max(cl, nbr))
                if key in visited:
                    continue
                visited.add(key)
                length = 1
                prev, cur = cl, nbr
                while cur not in cl_set:
                    nexts = [n for n in adj[cur] if n != prev]
                    if not nexts:
                        break
                    prev, cur = cur, nexts[0]
                    length += 1
                strand_lengths.append(length)
        return np.array(strand_lengths) if strand_lengths else None

    def _bond_lengths(self):
        if not self.all_bonds or self.positions is None:
            return None
        pos = self.positions; L = self.L
        lengths = []
        for u, v in self.all_bonds:
            dr = pos[u] - pos[v]
            dr -= L * np.round(dr / L)
            lengths.append(np.linalg.norm(dr))
        return np.array(lengths)
