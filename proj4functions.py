import mw_plot
from astropy.coordinates import SkyCoord
from astropy import units as u

# Topic 1 
# Task 1
def get_galactic_coords(name):
    """
    Look up the sky positions of an astronomical object by name and return its Galactic position
    
    Parameters:
    name -str ("M31", "Polaris", "M13")
    
    Return
    (l, b) giving the Galactic longitude and latitude in decimal degrees.
    """
    obj = SkyCoord.from_name(name)         
    gal = obj.galactic                     
    return gal.l.degree, gal.b.degree      

import math
import numpy as np
from scipy.optimize import curve_fit

# Task 3
def plt2rgbarr(fig):
    """
    A function to transform a matplotlib to a 3d rgb np.array 

    Input
    -----
    fig: matplotlib.figure.Figure
        The plot that we want to encode.        

    Output
    ------
    np.array(ndim, ndim, 3): A 3d map of each pixel in a rgb encoding (the three dimensions are x, y, and rgb)
    
    """
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h, w, 4)

    return rgba_arr[:, :, :3]  

#
#--------------------------------
# Task 0 - Import from Project 2


def euler_solve(f, t_span, y0, h):
    """
    Forward Euler for y' = f(y,t) with fixed step h.

    Returns
    -------
    T : (m,) ndarray
        Time grid from t_span[0] to t_span[1] with step h (last step shortened).
    Y : (m, d) ndarray
        States at each time in T; rows align with T.
    """
    t0, t1 = t_span
    y = np.asarray(y0, float)
    t = t0
    T, Y = [t], [y.copy()]
    # small epsilon so we don't overshoot due to float roundoff
    while t < t1 - 1e-12:
        hk = min(h, t1 - t)
        y = y + hk * f(y, t)
        t = t + hk
        T.append(t); Y.append(y.copy())
    return np.array(T), np.vstack(Y)


def make_ebola_rhs(beta0, lam, N, sigma, gamma):
    """
    Build f(y,t) for the SEIR-like Ebola model with decaying transmission:
        beta(t) = beta0 * exp(-lam * t)

    State y = [S, E, Z, R]
        dS/dt = -beta(t) * S * Z / N
        dE/dt =  beta(t) * S * Z / N - sigma * E
        dZ/dt =  sigma * E - gamma * Z
        dR/dt =  gamma * Z
    """
    def f(y, t):
        S, E, Z, R = y
        beta_t = beta0 * math.exp(-lam * t)
        inf = beta_t * S * Z / N
        dS = -inf
        dE = inf - sigma * E
        dZ = sigma * E - gamma * Z
        dR = gamma * Z
        return np.array([dS, dE, dZ, dR], float)
    return f


# data helper

def load_country(path):
    """
    Load country data file with columns:
      col 1 = day index, col 2 = new cases per day (confirmed+probable)
    Skips header row.
    """
    data = np.loadtxt(path, skiprows=1, usecols=(1, 2))
    days = data[:, 0].astype(int)
    new_cases = data[:, 1]
    return days, new_cases


def reindex_days(days):
    """Shift day numbers so the first day is 0."""
    return days - int(days.min())


# simulation

def simulate_cumulative(days0, beta0, lam, *,
                        N=10_000_000, sigma=1/9.7, gamma=1/7, h=1.0):
    """
    Integrate the model on the integer grid covering 'days0' and
    return (T, cum_model, new_model) where:
      - T is the integer grid [0, 1, ..., tmax]
      - new_model ≈ sigma * E
      - cum_model is the running sum of new_model
    """
    t0, t1 = int(days0.min()), int(days0.max())
    S0, E0, Z0, R0 = N - 1, 0.0, 1.0, 0.0
    f = make_ebola_rhs(beta0, lam, N, sigma, gamma)
    T, Y = euler_solve(f, (t0, t1), [S0, E0, Z0, R0], h)
    # Y rows align with T; extract trajectories
    S, E, Z, R = Y.T
    new_model = sigma * E
    cum_model = np.cumsum(new_model)
    return T, cum_model, new_model


def _curvefit_target(t, beta0, lam,
                     N=10_000_000, sigma=1/9.7, gamma=1/7, h=1.0):
    """
    curve_fit target: return cumulative model values at integer times 't'.
    't' is an array of integer days (reindexed so first day is 0).
    """
    T, cum_model, _ = simulate_cumulative(t, beta0, lam,
                                          N=N, sigma=sigma, gamma=gamma, h=h)
    # T is [0..tmax]; we assume t are integers within that range
    return cum_model[t.astype(int)]


def fit_country(days, new_cases, *,
                N=10_000_000, sigma=1/9.7, gamma=1/7,
                p0=(0.25, 0.02), bounds=((0.0, 0.0), (np.inf, np.inf)),
                maxfev=20000):
    """
    Fit (beta0, lam) to cumulative data for one country using scipy.curve_fit.

    Returns
    -------
    (beta0_hat, lam_hat), (T, cum_model, new_model), cum_data, days0
    """
    days0 = reindex_days(days)
    cum_data = np.cumsum(new_cases)

    # Partial that carries constants via lambda
    target = lambda t, b0, l: _curvefit_target(t, b0, l,
                                               N=N, sigma=sigma, gamma=gamma, h=1.0)

    (beta0_hat, lam_hat), pcov = curve_fit(
        target, days0, cum_data, p0=p0, bounds=bounds, maxfev=maxfev
    )

    T, cum_model, new_model = simulate_cumulative(
        days0, beta0_hat, lam_hat, N=N, sigma=sigma, gamma=gamma, h=1.0
    )
    return (beta0_hat, lam_hat), (T, cum_model, new_model), cum_data, days0

#-------------------------------

# Topic 2 task 5 Func from tutorial: create dataset with look-back window

def create_dataset(dataset, look_back=1):
    """
    Turn a 1D time series into (X, y) pairs.

    For each point, X is the previous `look_back` values
    and y is the next value.

    dataset: 2D array of shape (n_samples, 1)
    look_back: how many past steps to use

    Returns:
      X: (n_samples - look_back - 1, look_back)
      y: (n_samples - look_back - 1,)
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)