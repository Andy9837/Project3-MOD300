import numpy as np

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

import numpy as np

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

import numpy as np

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
