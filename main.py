import numpy as np
import matplotlib.pyplot as plt

# ---------------- Task 1 & 2: Random walkers (your code) ----------------
def _plot_paths_3d(paths, title, max_traces=15):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    k = min(paths.shape[0], max_traces)
    for i in range(k):
        x, y, z = paths[i, :, 0], paths[i, :, 1], paths[i, :, 2]
        ax.plot(x, y, z, linewidth=1.0)
        ax.scatter([x[0]], [y[0]], [z[0]], s=12)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=12)
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.grid(True); plt.tight_layout(); plt.show()

def _sample_starts(n, xlim=(0.0, 200.0), ylim=(0.0, 200.0), zlim=(0.0, 200.0), rng=None):
    rng = rng or np.random.default_rng()
    xs = rng.uniform(xlim[0], xlim[1], size=n)
    ys = rng.uniform(ylim[0], ylim[1], size=n)
    zs = rng.uniform(zlim[0], zlim[1], size=n)
    return np.column_stack([xs, ys, zs])

def _wrap_periodic(coords, xlim, ylim, zlim):
    mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=float)
    lens = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]], dtype=float)
    return (coords - mins) % lens + mins

def random_walk_np(n_walkers=5, n_steps=10_000, step_low=-1.0, step_high=1.0,
                   xlim=(0.0,200.0), ylim=(0.0,200.0), zlim=(0.0,200.0),
                   periodic=False, seed=42):
    rng = np.random.default_rng(seed)
    starts = _sample_starts(n_walkers, xlim, ylim, zlim, rng)
    paths = np.empty((n_walkers, n_steps + 1, 3), dtype=float); paths[:, 0, :] = starts
    for i in range(n_walkers):
        steps = rng.uniform(step_low, step_high, size=(n_steps, 3))
        traj = starts[i] + np.cumsum(steps, axis=0)
        traj = np.vstack([starts[i], traj])
        if periodic: traj = _wrap_periodic(traj, xlim, ylim, zlim)
        paths[i] = traj
    _plot_paths_3d(paths, "Task 1: random_walk_np (readable NumPy)")
    return starts, paths

def random_walk_np_fast(n_walkers=5, n_steps=10_000, step_low=-1.0, step_high=1.0,
                        xlim=(0.0,200.0), ylim=(0.0,200.0), zlim=(0.0,200.0),
                        periodic=False, seed=0):
    rng = np.random.default_rng(seed)
    starts = _sample_starts(n_walkers, xlim, ylim, zlim, rng)
    steps = rng.uniform(step_low, step_high, size=(n_walkers, n_steps, 3))
    disp = np.cumsum(steps, axis=1)
    paths = np.empty((n_walkers, n_steps + 1, 3), dtype=float)
    paths[:, 0, :] = starts
    paths[:, 1:, :] = starts[:, None, :] + disp
    if periodic: paths = _wrap_periodic(paths, xlim, ylim, zlim)
    _plot_paths_3d(paths, "Task 2: random_walk_np_fast (vectorized)")
    return starts, paths

# ---------------- Task 5: Monte Carlo accessible volume ----------------
class CellList:
    def __init__(self, xyz, r_infl, box_min, box_max, cell_size=None):
        self.xyz = np.asarray(xyz, float)
        self.r_infl = np.asarray(r_infl, float)
        self.box_min = np.asarray(box_min, float)
        self.box_max = np.asarray(box_max, float)
        self.box_len = self.box_max - self.box_min
        if cell_size is None: cell_size = max(0.5, float(np.median(self.r_infl))) if len(self.r_infl) else 1.0
        self.cell_size = float(cell_size)
        nxyz = np.ceil(self.box_len / self.cell_size).astype(int)
        self.nx, self.ny, self.nz = np.maximum(1, nxyz)
        self.cells = {}
        for i, p in enumerate(self.xyz):
            self.cells.setdefault(self._key(p), []).append(i)

    def _key(self, p):
        ijk = np.floor((p - self.box_min) / self.cell_size).astype(int)
        ijk = np.clip(ijk, 0, [self.nx-1, self.ny-1, self.nz-1])
        return tuple(ijk.tolist())

    def neighbors(self, q):
        i, j, k = self._key(q)
        for di in (-1,0,1):
            for dj in (-1,0,1):
                for dk in (-1,0,1):
                    ii, jj, kk = i+di, j+dj, k+dk
                    if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                        for idx in self.cells.get((ii,jj,kk), []):
                            yield idx

def mc_accessible_volume(atoms_xyz, atoms_vdw, probe_radius,
                         box_min, box_max, n_samples=200_000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = np.asarray(atoms_xyz, float)
    r_infl = np.asarray(atoms_vdw, float) + float(probe_radius)
    box_min = np.asarray(box_min, float); box_max = np.asarray(box_max, float)
    L = box_max - box_min; V_box = float(np.prod(L))

    if len(xyz) == 0:
        frac = 1.0
        return V_box * frac, frac

    cl = CellList(xyz, r_infl, box_min, box_max)
    pts = box_min + rng.random((n_samples, 3)) * L
    r2 = cl.r_infl * cl.r_infl
    accessible = 0
    for q in pts:
        hit = False
        for idx in cl.neighbors(q):
            if np.sum((q - cl.xyz[idx])**2) < r2[idx]:
                hit = True; break
        if not hit: accessible += 1
    frac = accessible / n_samples
    return V_box * frac, frac

# -------- Helpers for quick verification (Task 4 checks) --------
def analytic_accessible_single_sphere(box_min, box_max, r_vdw, probe):
    L = np.array(box_max) - np.array(box_min)
    V_box = float(np.prod(L))
    R = r_vdw + probe
    return V_box - (4.0/3.0)*np.pi*R**3

def make_mock_dna(n_bp=12, rise=3.4, radius=10.0, r_vdw=1.7):
    ang = np.linspace(0, 2*np.pi, n_bp, endpoint=False)
    z = np.arange(n_bp)*rise
    x1, y1 = radius*np.cos(ang), radius*np.sin(ang)
    x2, y2 = radius*np.cos(ang+np.pi), radius*np.sin(ang+np.pi)
    xyz = np.vstack([np.stack([x1,y1,z],1), np.stack([x2,y2,z],1)])
    radii = np.full(xyz.shape[0], r_vdw)
    return xyz, radii

# ---------------- Demo runner (satisfies “code and test”) ----------------
if __name__ == "__main__":
    # Optional: show walkers
    random_walk_np(n_walkers=5, n_steps=3000, periodic=True)
    random_walk_np_fast(n_walkers=5, n_steps=3000, periodic=True)

    # Box
    box_min = np.array([0.,0.,0.]); box_max = np.array([60.,60.,60.])
    V_box = np.prod(box_max - box_min)

    # Empty-box control
    V_mc, f = mc_accessible_volume([], [], 1.4, box_min, box_max, n_samples=50_000, seed=1)
    print(f"[Empty] V_box={V_box:.3f}  V_mc={V_mc:.3f}  frac={f:.5f}")

    # Single-sphere analytic check
    r_vdw = 2.0; probe = 1.4; center = np.array([[30.,30.,30.]])
    V_true = analytic_accessible_single_sphere(box_min, box_max, r_vdw, probe)
    V_mc, _ = mc_accessible_volume(center, [r_vdw], probe, box_min, box_max, n_samples=200_000, seed=2)
    err = abs(V_mc - V_true)/V_true
    print(f"[Sphere] V_true={V_true:.3f}  V_mc={V_mc:.3f}  rel.err={err:.3%}")

    # Monotonicity + convergence with a crude DNA-like scaffold
    dna_xyz, dna_r = make_mock_dna()
    pad = 15.0
    bb_min = np.min(dna_xyz, axis=0) - pad
    bb_max = np.max(dna_xyz, axis=0) + pad
    V_small,_ = mc_accessible_volume(dna_xyz, dna_r, 1.0, bb_min, bb_max, n_samples=80_000, seed=3)
    V_big,_   = mc_accessible_volume(dna_xyz, dna_r, 2.0, bb_min, bb_max, n_samples=80_000, seed=4)
    print(f"[Mono]   V(probe=1.0)={V_small:.3f}  V(probe=2.0)={V_big:.3f}  OK={V_big <= V_small}")
    for N in [20_000, 50_000, 100_000, 200_000]:
        V, frac = mc_accessible_volume(dna_xyz, dna_r, 1.4, bb_min, bb_max, n_samples=N, seed=5)
        print(f"[Conv]   N={N:6d}  V={V:.3f}  frac={frac:.6f}")
