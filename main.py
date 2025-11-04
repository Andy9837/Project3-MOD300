import numpy as np
import matplotlib.pyplot as plt


# -------- plotting helper --------
def _plot_paths_3d(paths, title, max_traces=15):
    """
    Plot up to max_traces trajectories in 3D.
    paths shape: (n_walkers, n_steps+1, 3)
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    k = min(paths.shape[0], max_traces)
    for i in range(k):
        x, y, z = paths[i, :, 0], paths[i, :, 1], paths[i, :, 2]
        ax.plot(x, y, z, linewidth=1.0)
        ax.scatter([x[0]], [y[0]], [z[0]], s=12)     # start
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=12)  # end
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# -------- helpers for starts & wrapping --------
def _sample_starts(n, xlim=(0.0, 200.0), ylim=(0.0, 200.0), zlim=(0.0, 200.0), rng=None):
    """Return (n,3) random start points uniform in the given box."""
    rng = rng or np.random.default_rng()
    xs = rng.uniform(xlim[0], xlim[1], size=n)
    ys = rng.uniform(ylim[0], ylim[1], size=n)
    zs = rng.uniform(zlim[0], zlim[1], size=n)
    return np.column_stack([xs, ys, zs])


def _wrap_periodic(coords, xlim, ylim, zlim):
    """
    Periodic wrapping for coords array (..., 3) w.r.t. [min,max] per axis.
    Works for any non-zero box lengths.
    """
    mins = np.array([xlim[0], ylim[0], zlim[0]], dtype=float)
    lens = np.array([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]], dtype=float)
    return (coords - mins) % lens + mins


# -------- Task 1: readable NumPy version --------
def random_walk_np(n_walkers=5,
                   n_steps=10_000,
                   step_low=-1.0,
                   step_high=1.0,
                   xlim=(0.0, 200.0),
                   ylim=(0.0, 200.0),
                   zlim=(0.0, 200.0),
                   periodic=False,
                   seed=42):
    """
    3D random walkers from different random starts.
    NumPy for steps; simple loop over walkers.
    Returns: starts (n,3), paths (n, n_steps+1, 3)
    """
    rng = np.random.default_rng(seed)

    starts = _sample_starts(n_walkers, xlim, ylim, zlim, rng)
    paths = np.empty((n_walkers, n_steps + 1, 3), dtype=float)
    paths[:, 0, :] = starts

    for i in range(n_walkers):
        steps = rng.uniform(step_low, step_high, size=(n_steps, 3))
        traj = starts[i] + np.cumsum(steps, axis=0)   # (n_steps,3)
        traj = np.vstack([starts[i], traj])           # include start
        if periodic:
            traj = _wrap_periodic(traj, xlim, ylim, zlim)
        paths[i] = traj

    _plot_paths_3d(paths, "Task 1: random_walk_np (readable NumPy)")
    return starts, paths


# -------- Task 2: fully vectorized (fast) --------
def random_walk_np_fast(n_walkers=5,
                        n_steps=10_000,
                        step_low=-1.0,
                        step_high=1.0,
                        xlim=(0.0, 200.0),
                        ylim=(0.0, 200.0),
                        zlim=(0.0, 200.0),
                        periodic=False,
                        seed=0):
    """
    Vectorized 3D random walkers (no Python loops over walkers).
    Returns: starts (n,3), paths (n, n_steps+1, 3)
    """
    rng = np.random.default_rng(seed)

    starts = _sample_starts(n_walkers, xlim, ylim, zlim, rng)              # (n,3)
    steps = rng.uniform(step_low, step_high, size=(n_walkers, n_steps, 3)) # (n,steps,3)

    disp = np.cumsum(steps, axis=1)                                        # (n,steps,3)

    paths = np.empty((n_walkers, n_steps + 1, 3), dtype=float)
    paths[:, 0, :] = starts
    paths[:, 1:, :] = starts[:, None, :] + disp

    if periodic:
        paths = _wrap_periodic(paths, xlim, ylim, zlim)

    _plot_paths_3d(paths, "Task 2: random_walk_np_fast (vectorized)")
    return starts, paths
