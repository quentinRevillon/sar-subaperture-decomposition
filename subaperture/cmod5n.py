"""CMOD5.N geophysical model function for C-band VV backscatter over the ocean.

Vectorised Python version by K. F. Dagestad (NERSC, 2011), taken from the
openwind project: https://github.com/nansencenter/openwind/blob/master/openwind/cmod5n.py
Only the layout and docstrings were changed; the coefficients and equations are unchanged.

Usage:
    import numpy as np
    from subaperture.cmod5n import cmod5n_forward, cmod5n_inverse

    shape = (2, 2)
    sigma0 = cmod5n_forward(np.full(shape, 10.0), np.full(shape, 45.0), np.full(shape, 35.0))
    wind = cmod5n_inverse(sigma0, np.full(shape, 45.0), np.full(shape, 35.0))

History of the model (from the original header):
    A. Stoffelen              May 1991   ECMWF  CMOD4
    A. Stoffelen, S. de Haan  Dec 2001   KNMI   CMOD5 prototype
    H. Hersbach               June 2002  ECMWF  complete revision
    J. de Kloe                July 2003  KNMI   rewritten in Fortran 90
    A. Verhoef                Jan 2008   KNMI   CMOD5 for neutral winds
    K. F. Dagestad            Oct 2011   NERSC  vectorised Python version
"""

import numpy as np

DTOR = 57.29577951
THETM = 40.0
THETHR = 25.0
ZPOW = 1.6

# A 0 is prepended so that indices match the 1-based coefficients of the reference implementation.
C = [0, -0.6878, -0.7957, 0.3380, -0.1728, 0.0000, 0.0040, 0.1103, 0.0159,
     6.7329, 2.7713, -2.2885, 0.4971, -0.7250, 0.0450,
     0.0066, 0.3222, 0.0120, 22.7000, 2.0813, 3.0000, 8.3659,
     -3.3428, 1.3236, 6.2437, 2.3893, 0.3249, 4.1590, 1.6930]


def cmod5n_forward(v, phi, theta):
    """Return the CMOD5.N normalised radar cross section (linear units).

    Args:
        v: 10 m neutral wind speed in m/s (>= 0), numpy array.
        phi: angle between radar azimuth and wind direction in degrees, same shape as v.
        theta: incidence angle in degrees, same shape as v.

    Returns:
        Array of sigma0 values, same shape as the inputs.
    """
    y0 = C[19]
    pn = C[20]
    a = C[19] - (C[19] - 1) / C[20]
    b = 1.0 / (C[20] * (C[19] - 1.0) ** (3 - 1))

    fi = phi / DTOR
    csfi = np.cos(fi)
    cs2fi = 2.00 * csfi * csfi - 1.00

    x = (theta - THETM) / THETHR
    xx = x * x

    # B0: function of wind speed and incidence angle
    a0 = C[1] + C[2] * x + C[3] * xx + C[4] * x * xx
    a1 = C[5] + C[6] * x
    a2 = C[7] + C[8] * x
    gam = C[9] + C[10] * x + C[11] * xx
    s0 = C[12] + C[13] * x

    s = a2 * v
    s_vec = s.copy()
    s_below = s_vec < s0
    s_vec[s_below] = s0[s_below]
    a3 = 1.0 / (1.0 + np.exp(-s_vec))
    s_below = s < s0
    a3[s_below] = a3[s_below] * (s[s_below] / s0[s_below]) ** (s0[s_below] * (1.0 - a3[s_below]))
    b0 = (a3 ** gam) * 10.0 ** (a0 + a1 * v)

    # B1: function of wind speed and incidence angle
    b1 = C[15] * v * (0.5 + x - np.tanh(4.0 * (x + C[16] + C[17] * v)))
    b1 = C[14] * (1.0 + x) - b1
    b1 = b1 / (np.exp(0.34 * (v - C[18])) + 1.0)

    # B2: function of wind speed and incidence angle
    v0 = C[21] + C[22] * x + C[23] * xx
    d1 = C[24] + C[25] * x + C[26] * xx
    d2 = C[27] + C[28] * x
    v2 = v / v0 + 1.0
    v2_below = v2 < y0
    v2[v2_below] = a + b * (v2[v2_below] - 1.0) ** pn
    b2 = (-d1 + d2 * v2) * np.exp(-v2)

    # Combine the three Fourier terms
    return b0 * (1.0 + b1 * csfi + b2 * cs2fi) ** ZPOW


def cmod5n_inverse(sigma0_obs, phi, incidence, iterations=10):
    """Return the 10 m neutral wind speed (m/s) matching an observed sigma0.

    Bisection on wind speed: cmod5n_forward is iterated until it agrees with sigma0_obs.

    Args:
        sigma0_obs: observed normalised radar cross section, linear units.
        phi: angle between radar azimuth and wind direction in degrees.
        incidence: incidence angle in degrees.
        iterations: number of bisection steps.

    Returns:
        Array of wind speeds, same shape as sigma0_obs.
    """
    v = np.full(sigma0_obs.shape, 10.0)
    step = 10.0
    for _ in range(1, iterations):
        too_high = cmod5n_forward(v, phi, incidence) - sigma0_obs > 0
        v = v + step
        v[too_high] = v[too_high] - 2 * step
        step = step / 2
    return v
