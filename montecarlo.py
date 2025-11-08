import numpy as np
import matplotlib.pyplot as plt
### ----------------------------Topic 1 ---------------------------------
#Task 0
def make_box(xlen, ylen, zlen, origin=(0, 0, 0)):
    """Return a 3D simulation box as ((xmin, xmax), (ymin, ymax), (zmin, zmax))."""
    ox, oy, oz = origin
    return ((ox, ox + xlen), (oy, oy + ylen), (oz, oz + zlen))

#Task 1
def random_point_in_box(box, rng=None):
    """ Generate on random point inside 3D box.
    
    Parameters:
    box: tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    rng: np.random.Generator

    Returns: 
    np.ndarray: A 3D point uniformly distributed in the box.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    u = rng.random(3) 
    point = np.array([
        xmin + u[0] * (xmax - xmin),
        ymin + u[1] * (ymax - ymin),
        zmin + u[2] * (zmax - zmin)
    ])
    return point
    
# Task 2
def random_sphere_in_box(box, rmin, rmax, rng=None):
    """
    Generate a random sphere fully contained within a 3D box.

    Parameters:
    box : tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    rmin, rmax : float, Minimum and maximum sphere radius.
    rng : np.random.Generator

    Returns:
    tuple: (center, radius), where center is a np.ndarray of shape (3,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw random radius
    r = rmin + (rmax - rmin) * rng.random()

    # Calculate allowed region for the center (so the sphere fits)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    xmin_c, xmax_c = xmin + r, xmax - r
    ymin_c, ymax_c = ymin + r, ymax - r
    zmin_c, zmax_c = zmin + r, zmax - r

    # Random center within those limits
    center = np.array([
        rng.uniform(xmin_c, xmax_c),
        rng.uniform(ymin_c, ymax_c),
        rng.uniform(zmin_c, zmax_c)
    ])

    return center, r

import numpy as np

def inside_sphere(p, c, r):
    """
    Check if a point is inside (or on) a sphere.

    Parameters:
    p : array-like of shape (3,)
        The point coordinates (x, y, z).
    c : array-like of shape (3,)
        The sphere center (cx, cy, cz).
    r : float
        Sphere radius.

    Returns:
    bool
        True if the point lies inside or on the sphere (inclusive), False otherwise.
    """
    p = np.asarray(p, dtype=float)
    c = np.asarray(c, dtype=float)
    return np.dot(p - c, p - c) <= r * r


def points_in_sphere(points, c, r):
    """
    Vectorized check: which points are inside (or on) a sphere.

    Parameters:
    points : array-like of shape (N, 3)
        Points to test.
    c : array-like of shape (3,)
        The sphere center (cx, cy, cz).
    r : float
        Sphere radius.

    Returns:
    np.ndarray
        Boolean mask of shape (N,), True where point is inside or on the sphere.
    """
    P = np.asarray(points, dtype=float)
    c = np.asarray(c, dtype=float)
    d2 = np.sum((P - c) ** 2, axis=1)
    return d2 <= r * r

import numpy as np

def box_volume(box):
    """
    Compute the volume of a 3D axis-aligned box.

    Parameters:
    box : tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))

    Returns:
    float: box volume
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    return (xmax - xmin) * (ymax - ymin) * (zmax - zmin)


def fraction_inside_sphere(box, center, radius, n, rng=None):
    """
    Estimate the fraction of uniformly sampled points in 'box'
    that fall inside (or on) a given sphere.

    Parameters:
    box : tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    center : array-like of shape (3,)
        Sphere center (cx, cy, cz)
    radius : float
        Sphere radius
    n : int
        Number of Monte Carlo samples
    rng : np.random.Generator, optional
        Random number generator for reproducibility

    Returns:
    tuple: (p, se)
        p  = estimated fraction in [0,1]
        se = standard error of the estimate
    """
    if rng is None:
        rng = np.random.default_rng()

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], dtype=float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)

    # sample N points uniformly in the box (vectorized)
    P = mins + rng.random((n, 3)) * lens

    # inside test (vectorized)
    C = np.asarray(center, dtype=float)
    r2 = float(radius) ** 2
    d2 = np.sum((P - C) ** 2, axis=1)
    inside = d2 <= r2

    p = inside.mean()
    se = np.sqrt(p * (1.0 - p) / n)
    return p, se
# source: https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291



#task5 
def estimate_pi_2d(n, rng=None):
    """
    Estimate pi using Monte Carlo in 2D: area of a quarter unit circle in [0,1]^2.

    Parameters:
    n : int
        Number of random samples.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns:
    tuple: (pi_hat, se)
        pi_hat : float
            Monte Carlo estimate of pi.
        se : float
            Standard error of the estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample (x,y) ~ U([0,1]^2)
    xy = rng.random((n, 2))
    d2 = np.sum(xy**2, axis=1)          # x^2 + y^2
    inside = d2 <= 1.0                  # quarter circle of radius 1

    p_hat = inside.mean()               # area fraction of quarter circle
    pi_hat = 4.0 * p_hat                # quarter circle area = pi/4
    # SE for p, then scale by 4
    se_p = np.sqrt(p_hat * (1.0 - p_hat) / n)
    se = 4.0 * se_p
    return pi_hat, se
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291

#task 7
def fraction_inside_union(box, centers, radii, n, rng=None):
    """
    Estimate the fraction of uniformly sampled points in 'box'
    that fall inside the union of multiple spheres.

    Parameters:
    box : tuple ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    centers : array-like of shape (M, 3)
        Sphere centers.
    radii : array-like of shape (M,)
        Sphere radii.
    n : int
        Number of Monte Carlo samples.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns:
    tuple: (p, se)
        p  = estimated fraction in [0,1]
        se = standard error of the estimate
    """
    if rng is None:
        rng = np.random.default_rng()

    centers = np.asarray(centers, dtype=float)     # (M,3)
    radii = np.asarray(radii, dtype=float)         # (M,)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], dtype=float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], dtype=float)

    # Sample N points uniformly in the box
    P = mins + rng.random((n, 3)) * lens           # (N,3)

    # Compute squared distances from each point to each center (broadcasting)
    # d2 has shape (N, M)
    d2 = np.sum((P[:, None, :] - centers[None, :, :])**2, axis=2)

    # For each point, is it inside ANY sphere?
    inside_any = (d2 <= (radii[None, :]**2)).any(axis=1)  # (N,)

    p = inside_any.mean()
    se = np.sqrt(p * (1.0 - p) / n)
    return p, se
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291



#task 8
def read_dna_coordinates(filename):
    """
    Read atomic coordinates from a plain-text file.

    Parameters:
    filename : str
        Path to a file where each line starts with an element symbol (H, C, N, O, P)
        followed by three coordinates.

    Returns:
    tuple: (elements, coords)
        elements : list of str
        coords   : np.ndarray of shape (N,3)
    """
    elements, coords = [], []
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, np.array(coords, dtype=float)


def atomic_radii(elements):
    """
    Map element symbols to approximate van der Waals radii (Å).

    Parameters:
    elements : list of str

    Returns:
    np.ndarray of radii (Å)
    """
    table = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "P": 1.8}
    return np.array([table.get(e, 1.5) for e in elements], dtype=float)

#task 9
def atomic_masses(elements):
    """
    Return approximate atomic masses (in atomic mass units, u) for given elements.

    Parameters:
    elements : list of str

    Returns:
    np.ndarray of atomic masses (u)
    """
    table = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974}
    return np.array([table.get(e, 12.0) for e in elements], dtype=float)


def density_from_atoms(elements, volume_angstrom3):
    """
    Estimate density of a molecule given element list and volume.

    Parameters:
    elements : list of str
        Atomic symbols (H, C, N, O, P)
    volume_angstrom3 : float
        Volume in Å³ (from Monte Carlo)

    Returns:
    float: density in g/cm³
    """
    import numpy as np

    # Total mass in atomic mass units (u)
    masses = atomic_masses(elements)
    total_mass_u = masses.sum()

    # Convert u → grams (1 u = 1.66054 × 10⁻²⁴ g)
    total_mass_g = total_mass_u * 1.66054e-24

    # Convert Å³ → cm³ (1 Å³ = 10⁻²⁴ cm³)
    volume_cm3 = volume_angstrom3 * 1e-24

    density = total_mass_g / volume_cm3
    return density

###------------------Topic 2---------------------------

def _wrap(P, box):
    """
        Wrap 3D point(s) P into the periodic box.

    Parameters
    ----------
    P : array_like, shape (..., 3)
        Point or points to wrap.
    box : ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        Axis-aligned bounds.

    Returns
    -------
    ndarray, shape (..., 3)
        Wrapped point(s) inside the box.
        
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    return mins + ((P - mins) % lens)


# Task 1
def random_walkers_3d(box, n_walkers, n_steps, step_sigma=1.0, rng=None):
    """
    Generate 3D random-walk trajectories.

    Parameters
    ----------
    box : tuple
        Axis-aligned box as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    n_walkers : int
        Number of walkers.
    n_steps : int
        Number of steps per walker.
    step_sigma : float, optional
        Std. dev. of each step (per axis).
    rng : np.random.Generator, optional
        RNG to use; a new default_rng() is created if None.

    Returns
    -------
    traj : ndarray, shape (n_steps+1, n_walkers, 3)
        Positions at t = 0..n_steps. Starts uniform in the box.
    
    """
    rng = np.random.default_rng() if rng is None else rng
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)
    traj = np.empty((n_steps + 1, n_walkers, 3), float)
    traj[0] = mins + rng.random((n_walkers, 3)) * lens
    for t in range(1, n_steps + 1):
        traj[t] = _wrap(traj[t-1] + rng.normal(0.0, step_sigma, size=(n_walkers, 3)), box)
    return traj

# Task 2 
def random_walkers_3d_fast(box, n_walkers, n_steps, step_sigma=1.0, rng=None):
    """
       Generate 3D random-walk trajectories (vectorized/fast).

    Same API as `random_walkers_3d`, but precomputes all steps and
    updates positions in a vectorized loop for speed.

    Parameters
    ----------
    box : tuple
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    n_walkers : int
    n_steps : int
    step_sigma : float, optional
    rng : np.random.Generator, optional

    Returns
    -------
    traj : ndarray, shape (n_steps+1, n_walkers, 3)
        Positions at t = 0..n_steps.
    """
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

        # unwrap: correct jumps larger than half the box by ± L 
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



## Topic 2, Task 3-5

def bounding_box_from_atoms(coords, radii, pad=2.0):
    """
        Compute an axis-aligned bounding box that contains all spheres:
    centers = coords[i], radii = radii[i], each sphere inflated by `pad` (Å).

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Atomic coordinates (Å).
    radii : array_like, shape (N,)
        Atomic radii (Å).
    pad : float, optional
        Extra margin added to all sides (Å). Default 2.0.

    Returns
    -------
    box : tuple
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)), in Å.
    
    """
    coords = np.asarray(coords, float)
    radii = np.asarray(radii, float)
    mins = coords.min(axis=0) - radii.max() - pad
    maxs = coords.max(axis=0) + radii.max() + pad
    return ((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2]))

def inaccessible_fraction_union(box, centers, radii, n, rng=None):
    """
        Estimate the fraction of the box occupied by the union of spheres.

    Parameters
    ----------
    box : ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    centers : array_like, shape (M,3)
    radii : array_like, shape (M,)
        Use inflated radii (ri + rp).
    n : int
        Number of Monte Carlo samples.
    rng : np.random.Generator, optional
    batch : int or None
        If set, process samples in batches of this size to reduce memory.

    Returns
    -------
    p : float
        Estimated blocked fraction (inaccessible).
    se : float
        Standard error of p (binomial SE).
    """
    if rng is None:
        rng = np.random.default_rng()
    centers = np.asarray(centers, float)     # (M,3)
    radii = np.asarray(radii, float)         # (M,)

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = box
    mins = np.array([xmin, ymin, zmin], float)
    lens = np.array([xmax - xmin, ymax - ymin, zmax - zmin], float)

    P = mins + rng.random((n, 3)) * lens     # (N,3)
    d2 = np.sum((P[:, None, :] - centers[None, :, :])**2, axis=2)  # (N,M)
    inside_any = (d2 <= (radii[None, :]**2)).any(axis=1)

    p = inside_any.mean()
    se = np.sqrt(p * (1.0 - p) / n)
    return p, se
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291


def accessible_volume_mc(coords, radii, rp=1.4, n=1_000_000, pad=2.0, rng=None):
    """
    Monte Carlo estimate of solvent-accessible volume (Å^3) for a probe of radius rp.

    Parameters
    ----------
    coords : array_like, shape (N,3)
        Atomic coordinates (Å).
    radii : array_like, shape (N,)
        Van der Waals radii (Å).
    rp : float, optional
        Probe radius (Å), e.g. 1.4 for water.
    n : int, optional
        Number of random samples.
    pad : float, optional
        Extra box padding (Å) added around inflated atoms.
    rng : np.random.Generator, optional
        RNG to use; a new default_rng() is created if None.
    batch : int or None, optional
        If set, process samples in batches of this size.
    with_details : bool, optional
        If True, also return (rp, accessible_fraction).

    Returns
    -------
    V_acc : float
        Accessible volume (Å^3).
    SE_acc : float
        Standard error of V_acc (Å^3).
    box : tuple
        ((xmin,xmax),(ymin,ymax),(zmin,zmax)) used for sampling.
    [rp, frac_acc] : optional
        Only if with_details=True.
    """
    if rng is None:
        rng = np.random.default_rng()
    coords = np.asarray(coords, float)
    radii = np.asarray(radii, float)
    box = bounding_box_from_atoms(coords, radii + rp, pad=pad)
    V_box = box_volume(box)

    # Use inflated radii (ri + rp) for inaccessibility
    inflated = radii + rp
    p_in, se_p = inaccessible_fraction_union(box, coords, inflated, n, rng=rng)
    p_acc = 1.0 - p_in
    V_acc = V_box * p_acc
    SE_acc = V_box * se_p
    return V_acc, SE_acc, box
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291


def blocked_volume_mc(coords, radii, rp=1.4, n=1_000_000, pad=2.0, rng=None):
    """
        Estimate the absolute blocked (inaccessible) volume for a probe of radius `rp`
    using Monte Carlo sampling.

    The blocked region is the union of spheres with radii (r_i + rp) centered at
    `coords`. We sample uniformly in a padded bounding box that encloses all
    inflated spheres, compute the accessible volume V_acc via Monte Carlo, then
    return V_blocked = V_box - V_acc along with its standard error.

    Parameters
    ----------
    coords : array_like, shape (N, 3)
        Atomic coordinates in Å.
    radii : array_like, shape (N,)
        Atomic (van der Waals) radii in Å.
    rp : float, optional
        Probe radius in Å (e.g., 1.4 for water). Default is 1.4.
    n : int, optional
        Number of Monte Carlo samples. Default is 1_000_000.
    pad : float, optional
        Extra padding (Å) added to all sides of the bounding box. Default is 2.0.
    rng : np.random.Generator, optional
        Random number generator. If None, uses np.random.default_rng().

    Returns
    -------
    V_inacc : float
        Blocked (inaccessible) volume in Å³.
    SE_inacc : float
        Standard error of V_inacc in Å³ (same magnitude as the SE of V_acc).
    box : tuple
        Sampling box as ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in Å.
    """
    V_acc, SE_acc, box = accessible_volume_mc(coords, radii, rp=rp, n=n, pad=pad, rng=rng)
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = box
    V_box = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    V_inacc = V_box - V_acc
    SE_inacc = SE_acc  # same magnitude, since SE scales with V_box
    return V_inacc, SE_inacc, box
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291


def dna_accessible_volume(filename, rp=1.4, n=1_000_000, pad=2.0, rng=None):
    """
    Estimate DNA solvent-accessible volume (Å³) for a probe of radius `rp`.

    Reads a text file with lines: <element> <x> <y> <z> (Å), maps elements to vdW radii,
    inflates by `rp`, builds a padded box, and uses Monte Carlo with `n` samples.

    Params
    ------
    filename : str
    rp : float  (probe radius, Å)
    n : int     (samples)
    pad : float (box padding, Å)
    rng : np.random.Generator | None

    Returns
    -------
    V_acc : float      # accessible volume (Å³)
    SE_acc : float     # MC standard error (Å³)
    box : tuple        # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) in Å
    """
    elems, coords = read_dna_coordinates(filename)
    radii = atomic_radii(elems)
    return accessible_volume_mc(coords, radii, rp=rp, n=n, pad=pad, rng=rng)
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291



def _sphere_test(r=10.0, L=100.0, n=2_000_000, rp=0.0, seed=0):
    """
        Sanity check against the analytic sphere volume.

    Parameters
    ----------
    r : float        # sphere radius (Å)
    L : float        # cubic box side (Å), box = [0,L]^3
    n : int          # number of Monte Carlo samples
    rp : float       # probe radius (Å), uses inflated radius r+rp
    seed : int       # RNG seed

    Returns
    -------
    V_mc : float     # MC estimate of blocked volume (Å³)
    V_true : float   # analytic blocked volume = (4/3)π(r+rp)^3 (Å³)
    z : float        # z-score = (V_mc - V_true) / SE, should be |z| <= 2
    """
    rng = np.random.default_rng(seed)
    c = np.array([[L/2, L/2, L/2]])
    box = ((0, L), (0, L), (0, L))
    inflated = np.array([r + rp])
    p_in, se_p = inaccessible_fraction_union(box, c, inflated, n, rng=rng)
    V_box = box_volume(box)
    V_mc = V_box * p_in
    V_true = (4.0/3.0) * np.pi * (r + rp)**3
    z = (V_mc - V_true) / (V_box * se_p + 1e-12)  
    return V_mc, V_true, z
# Sources:https://en.wikipedia.org/w/index.php?title=Monte_Carlo_integration&oldid=1319230291


