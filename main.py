"""Accessible volume via Monte Carlo + simple random-walker demos.
Includes asserts and tiny tests for bonus points.
"""

import numpy as np
import matplotlib.pyplot as plt



# ---------------- Task 1 & 2: Random walkers (your code) ----------------
def _plot_paths_3d(paths, title, max_traces=15):
    """Plot up to `max_traces` 3D trajectories; paths shape (n, steps+1, 3)."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    k = min(paths.shape[0], max_traces)
    for i in range(k):
        x, y, z = paths[i, :, 0], paths[i, :, 1], paths[i, :, 2]
        ax.plot(x, y, z, linewidth=1.0)
        ax.scatter([x[0]], [y[0]], [z[0]], s=12)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=12)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def _sample_starts(
    n, xlim=(0.0, 200.0), ylim=(0.0, 200.0), zlim=(0.0, 200.0), rng=None
):
    """Return (n, 3) random start points uniform in the given box."""
    rng = rng or np.random.default_rng()
    xs = rng.uniform(xlim[0], xlim[1], size=n)
    ys = rng.uniform(ylim[0], ylim[1], size=n)
    zs = rng.uniform(zlim[0], zlim[1], size=n)
    return np.column_stack([xs, ys, zs])


def _wrap_periodic(coords, xlim, ylim, zlim):
    """Apply periodic wrapping to coords array (..., 3) for given bounds."""
    mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=float)
    lens = np.array(
        [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]],
        dtype=float,
    )
    return (coords - mins) % lens + mins


def random_walk_np(
    n_walkers=5,
    n_steps=10_000,
    step_low=-1.0,
    step_high=1.0,
    xlim=(0.0, 200.0),
    ylim=(0.0, 200.0),
    zlim=(0.0, 200.0),
    periodic=False,
    seed=42,
):
    """Readable NumPy version: multiple walkers, cumulative uniform steps."""
    rng = np.random.default_rng(seed)
    starts = _sample_starts(n_walkers, xlim, ylim, zlim, rng)

    paths = np.empty((n_walkers, n_steps + 1, 3), dtype=float)
    paths[:, 0, :] = starts

    for i in range(n_walkers):
        steps = rng.uniform(step_low, step_high, size=(n_steps, 3))
        traj = starts[i] + np.cumsum(steps, axis=0)
        traj = np.vstack([starts[i], traj])
        if periodic:
            traj = _wrap_periodic(traj, xlim, ylim, zlim)
        paths[i] = traj

    _plot_paths_3d(paths, "Task 1: random_walk_np (readable NumPy)")
    return starts, paths


def random_walk_np_fast(
    n_walkers=5,
    n_steps=10_000,
    step_low=-1.0,
    step_high=1.0,
    xlim=(0.0, 200.0),
    ylim=(0.0, 200.0),
    zlim=(0.0, 200.0),
    periodic=False,
    seed=0,
):
    """Fully vectorized version: no Python loop over walkers."""
    rng = np.random.default_rng(seed)

    starts = _sample_starts(n_walkers, xlim, ylim, zlim, rng)
    steps = rng.uniform(step_low, step_high, size=(n_walkers, n_steps, 3))
    disp = np.cumsum(steps, axis=1)

    paths = np.empty((n_walkers, n_steps + 1, 3), dtype=float)
    paths[:, 0, :] = starts
    paths[:, 1:, :] = starts[:, None, :] + disp

    if periodic:
        paths = _wrap_periodic(paths, xlim, ylim, zlim)

    _plot_paths_3d(paths, "Task 2: random_walk_np_fast (vectorized)")
    return starts, paths


# ---------------- Task 5: Monte Carlo accessible volume ----------------
class CellList:
    """Simple cell list for neighbor lookups in a rectangular box."""

    def __init__(self, xyz, r_infl, box_min, box_max, cell_size=None):
        self.xyz = np.asarray(xyz, float)
        self.r_infl = np.asarray(r_infl, float)
        self.box_min = np.asarray(box_min, float)
        self.box_max = np.asarray(box_max, float)
        self.box_len = self.box_max - self.box_min

        if cell_size is None:
            cell_size = (
                max(0.5, float(np.median(self.r_infl))) if len(self.r_infl) else 1.0
            )
        self.cell_size = float(cell_size)

        nxyz = np.ceil(self.box_len / self.cell_size).astype(int)
        self.nx, self.ny, self.nz = np.maximum(1, nxyz)

        self.cells = {}
        for i, p in enumerate(self.xyz):
            self.cells.setdefault(self._key(p), []).append(i)

    def _key(self, p):
        """Map a point to its integer (i, j, k) cell key."""
        ijk = np.floor((p - self.box_min) / self.cell_size).astype(int)
        ijk = np.clip(ijk, 0, [self.nx - 1, self.ny - 1, self.nz - 1])
        return tuple(ijk.tolist())

    def neighbors(self, q):
        """Yield indices of atoms in q’s 27-cell neighborhood."""
        i, j, k = self._key(q)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                        # pylint: disable=consider-using-generator
                        yield from self.cells.get((ii, jj, kk), [])


def mc_accessible_volume(
    atoms_xyz,
    atoms_vdw,
    probe_radius,
    box_min,
    box_max,
    n_samples=200_000,
    seed=0,
):
    """Monte Carlo estimate: fraction of points not colliding with inflated atoms."""
    # --- asserts for bonus points ---
    atoms_xyz = np.asarray(atoms_xyz, float)
    atoms_vdw = np.asarray(atoms_vdw, float)
    box_min = np.asarray(box_min, float)
    box_max = np.asarray(box_max, float)

    assert probe_radius >= 0.0, "probe_radius must be >= 0"
    assert atoms_xyz.ndim in (1, 2)  # allow empty list -> 1D
    if atoms_xyz.size:
        assert atoms_xyz.ndim == 2 and atoms_xyz.shape[1] == 3, "atoms_xyz must be (N,3)"
        assert (
            atoms_vdw.ndim == 1 and len(atoms_vdw) == len(atoms_xyz)
        ), "atoms_vdw length must match atoms_xyz"
    assert np.all(box_max > box_min), "box_max must be greater than box_min on each axis"
    assert n_samples > 0, "n_samples must be positive"

    rng = np.random.default_rng(seed)
    xyz = atoms_xyz
    r_infl = atoms_vdw + float(probe_radius)

    L = box_max - box_min
    v_box = float(np.prod(L))

    if len(xyz) == 0:
        frac = 1.0
        return v_box * frac, frac

    cl = CellList(xyz, r_infl, box_min, box_max)
    pts = box_min + rng.random((n_samples, 3)) * L

    r2 = cl.r_infl * cl.r_infl
    accessible = 0
    for q in pts:
        hit = False
        for idx in cl.neighbors(q):
            if np.sum((q - cl.xyz[idx]) ** 2) < r2[idx]:
                hit = True
                break
        if not hit:
            accessible += 1

    frac = accessible / n_samples
    return v_box * frac, frac


# -------- Helpers for quick verification (Task 4 checks) --------
def analytic_accessible_single_sphere(box_min, box_max, r_vdw, probe):
    """Exact accessible volume if an inflated sphere fits fully inside the box."""
    # --- asserts for bonus points ---
    box_min = np.asarray(box_min, float)
    box_max = np.asarray(box_max, float)
    assert np.all(box_max > box_min), "invalid box"

    r_total = r_vdw + probe
    assert r_total >= 0.0, "r_vdw + probe must be >= 0"

    # caller should ensure the inflated sphere fits inside the box for closed-form
    L = np.array(box_max) - np.array(box_min)
    assert 2 * r_total < np.min(L), "inflated sphere must fit fully inside the box"

    v_box = float(np.prod(L))
    return v_box - (4.0 / 3.0) * np.pi * (r_total**3)


def make_mock_dna(n_bp=12, rise=3.4, radius=10.0, r_vdw=1.7):
    """Crude DNA-like two-helix scaffold (coordinates + identical radii)."""
    ang = np.linspace(0, 2 * np.pi, n_bp, endpoint=False)
    z = np.arange(n_bp) * rise
    x1, y1 = radius * np.cos(ang), radius * np.sin(ang)
    x2, y2 = radius * np.cos(ang + np.pi), radius * np.sin(ang + np.pi)
    xyz = np.vstack([np.stack([x1, y1, z], 1), np.stack([x2, y2, z], 1)])
    radii = np.full(xyz.shape[0], r_vdw)
    return xyz, radii


# -------------------- Tiny test functions (bonus) -----------------------
def test_empty_box():
    """Empty box must equal box volume."""
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([60.0, 60.0, 60.0])
    v_box = float(np.prod(box_max - box_min))
    v_mc, _ = mc_accessible_volume([], [], 1.4, box_min, box_max, n_samples=50_000, seed=0)
    assert abs(v_mc - v_box) / v_box < 5e-3, "empty box should equal box volume"


def test_single_sphere_analytic():
    """MC vs analytic for a single inflated sphere."""
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([60.0, 60.0, 60.0])
    r_vdw, probe = 2.0, 1.4
    center = np.array([[30.0, 30.0, 30.0]])
    v_true = analytic_accessible_single_sphere(box_min, box_max, r_vdw, probe)
    v_mc, _ = mc_accessible_volume(center, [r_vdw], probe, box_min, box_max,
                                   n_samples=200_000, seed=1)
    assert abs(v_mc - v_true) / v_true < 0.03, "MC should match analytic within 3%"


def test_monotonicity():
    """Bigger probe must not increase accessible volume."""
    xyz, r = make_mock_dna(n_bp=16)
    pad = 15.0
    bb_min, bb_max = xyz.min(0) - pad, xyz.max(0) + pad
    v1, _ = mc_accessible_volume(xyz, r, 1.0, bb_min, bb_max, n_samples=80_000, seed=0)
    v2, _ = mc_accessible_volume(xyz, r, 2.0, bb_min, bb_max, n_samples=80_000, seed=0)
    assert v2 <= v1 + 1e-9, "bigger probe must not increase accessible volume"


def test_convergence():
    """Estimate stabilizes as the number of samples grows."""
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([60.0, 60.0, 60.0])
    r_vdw, probe = 2.0, 1.4
    center = np.array([[30.0, 30.0, 30.0]])
    v1, _ = mc_accessible_volume(center, [r_vdw], probe, box_min, box_max,
                                 n_samples=100_000, seed=2)
    v2, _ = mc_accessible_volume(center, [r_vdw], probe, box_min, box_max,
                                 n_samples=200_000, seed=2)
    assert abs(v2 - v1) / v2 < 0.01, "estimate should stabilize as samples grow"


def test_reproducibility():
    """Across seeds, results should vary <1% for the same setup."""
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([60.0, 60.0, 60.0])
    r_vdw, probe = 2.0, 1.4
    center = np.array([[30.0, 30.0, 30.0]])
    vals = [
        mc_accessible_volume(center, [r_vdw], probe, box_min, box_max,
                             n_samples=100_000, seed=s)[0]
        for s in range(5)
    ]
    assert np.std(vals) / np.mean(vals) < 0.01, "reproducibility <1%"


# ---------------- Demo runner (satisfies “code and test”) ----------------
if __name__ == "__main__":
    # Optional: show walkers
    random_walk_np(n_walkers=5, n_steps=3000, periodic=True)
    random_walk_np_fast(n_walkers=5, n_steps=3000, periodic=True)

    # Box
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([60.0, 60.0, 60.0])
    v_box = np.prod(box_max - box_min)

    # Empty-box control
    v_mc, frac_empty = mc_accessible_volume(
        [], [], 1.4, box_min, box_max, n_samples=50_000, seed=1
    )
    print(f"[Empty] V_box={v_box:.3f}  V_mc={v_mc:.3f}  frac={frac_empty:.5f}")

    # Single-sphere analytic check
    r_vdw = 2.0
    probe = 1.4
    center = np.array([[30.0, 30.0, 30.0]])
    v_true = analytic_accessible_single_sphere(box_min, box_max, r_vdw, probe)
    v_mc, _ = mc_accessible_volume(
        center, [r_vdw], probe, box_min, box_max, n_samples=200_000, seed=2
    )
    rel_err = abs(v_mc - v_true) / v_true
    print(f"[Sphere] V_true={v_true:.3f}  V_mc={v_mc:.3f}  rel.err={rel_err:.3%}")

    # Monotonicity + convergence with a crude DNA-like scaffold
    dna_xyz, dna_r = make_mock_dna()
    pad = 15.0
    bb_min = np.min(dna_xyz, axis=0) - pad
    bb_max = np.max(dna_xyz, axis=0) + pad
    v_small, _ = mc_accessible_volume(
        dna_xyz, dna_r, 1.0, bb_min, bb_max, n_samples=80_000, seed=3
    )
    v_big, _ = mc_accessible_volume(
        dna_xyz, dna_r, 2.0, bb_min, bb_max, n_samples=80_000, seed=4
    )
    print(f"[Mono]   V(probe=1.0)={v_small:.3f}  V(probe=2.0)={v_big:.3f}  OK={v_big <= v_small}")

    v_prev = None
    for n_samples in [20_000, 50_000, 100_000, 200_000]:
        v_est, frac_est = mc_accessible_volume(
            dna_xyz, dna_r, 1.4, bb_min, bb_max, n_samples=n_samples, seed=5
        )
        print(f"[Conv]   N={n_samples:6d}  V={v_est:.3f}  frac={frac_est:.6f}")
        v_prev = v_est if v_prev is None else v_est

    # -------- run tiny tests (prints PASS/FAIL) --------
    tests = [
        test_empty_box,
        test_single_sphere_analytic,
        test_monotonicity,
        test_convergence,
        test_reproducibility,
    ]
    all_ok = True
    for tfunc in tests:
        try:
            tfunc()
            print(f"[TEST] {tfunc.__name__}: PASS")
        except AssertionError as exc:
            all_ok = False
            print(f"[TEST] {tfunc.__name__}: FAIL -> {exc}")
    if all_ok:
        print("All tests passed ✔️")



###### NEEEEEW LINE

# main.py
import numpy as np
import matplotlib.pyplot as plt

def _wrap(P, box):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    return mins + ((P - mins) % lens)

# Task 1
def random_walkers_3d(box, n_walkers, n_steps, step_sigma=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    traj = np.empty((n_steps + 1, n_walkers, 3), float)
    traj[0] = mins + rng.random((n_walkers, 3)) * lens
    for t in range(1, n_steps + 1):
        traj[t] = _wrap(traj[t-1] + rng.normal(0.0, step_sigma, size=(n_walkers, 3)), box)
    return traj

# Task 2 (fast)
def random_walkers_3d_fast(box, n_walkers, n_steps, step_sigma=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    traj = np.empty((n_steps + 1, n_walkers, 3), float)
    traj[0] = mins + rng.random((n_walkers, 3)) * lens
    steps = rng.normal(0.0, step_sigma, size=(n_steps, n_walkers, 3))
    pos = traj[0]
    for t in range(1, n_steps + 1):
        pos = _wrap(pos + steps[t-1], box)
        traj[t] = pos
    return traj

# --- one-line notebook entry points ---
def task1():
    """Return shape for Task 1 (for quick check)."""
    return random_walkers_3d(((0,100),(0,100),(0,100)), 1000, 500, 0.5,
                             rng=np.random.default_rng(42)).shape

def task2():
    """Return shape for Task 2 (for quick check)."""
    return random_walkers_3d_fast(((0,100),(0,100),(0,100)), 1000, 500, 0.5,
                                  rng=np.random.default_rng(42)).shape
def _plot(traj, box, title):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    L = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)

    k = min(8, traj.shape[1])
    for i in range(k):
        P = traj[:, i, :].copy()           # (T+1, 3) wrapped to the box
        d = np.diff(P, axis=0)

        # unwrap: correct jumps larger than half the box by ±L (minimal image)
        d -= np.where(d >  L/2, L, 0.0)
        d += np.where(d < -L/2, L, 0.0)

        P_unwrap = np.vstack([P[0:1], P[0:1] + np.cumsum(d, axis=0)])
        ax.plot(P_unwrap[:, 0], P_unwrap[:, 1], P_unwrap[:, 2], linewidth=1)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_paths_task1(n_walkers=8, n_steps=600, step_sigma=0.6, box=((0,100),(0,100),(0,100))):
    """Figure using Task 1 (simple loop)."""
    traj = random_walkers_3d(box, n_walkers=n_walkers, n_steps=n_steps, step_sigma=step_sigma)
    _plot(traj, box, "Random walkers — Task 1 (simple)")

def plot_paths_task2(n_walkers=8, n_steps=600, step_sigma=0.6, box=((0,100),(0,100),(0,100))):
    """Figure using Task 2 (fast)."""
    traj = random_walkers_3d_fast(box, n_walkers=n_walkers, n_steps=n_steps, step_sigma=step_sigma)
    _plot(traj, box, "Random walkers — Task 2 (fast)")


