"""Radiometric calibration and CMOD5.N normalisation of a complex SLC image.

The paper divides sigma0 by a CMOD5.N reference computed for a constant 10 m/s wind at 45 degrees
from the antenna look direction. The resulting ratio is called sea surface roughness (SSR) here.
To keep the phase needed by the azimuth FFT, the normalisation is applied to the complex signal:
|normalize_slc(...)|**2 equals SSR.

Usage:
    from subaperture.calibration import sigma0, cmod5n_reference, normalize_slc, to_db

    reference = cmod5n_reference(incidence)            # linear sigma0 of the reference wind
    ssr = sigma0(slc, lut) / reference                  # real SSR image
    slc_norm = normalize_slc(slc, lut, reference)       # complex image with |slc_norm|**2 == ssr
    ssr_db = to_db(ssr)
"""

import numpy as np

from subaperture.cmod5n import cmod5n_forward


def sigma0(slc, lut):
    """Return the calibrated sigma0 (linear) of a complex SLC: |DN|^2 / A_sigma^2."""
    return np.abs(slc) ** 2 / lut ** 2


def cmod5n_reference(incidence, wind_speed=10.0, wind_direction=45.0):
    """Return the CMOD5.N sigma0 (linear) for a constant wind, on the incidence-angle grid.

    Args:
        incidence: incidence angle map in degrees (its dtype sets the output dtype).
        wind_speed: reference wind speed in m/s.
        wind_direction: angle between radar azimuth and wind in degrees.
    """
    speed = np.full_like(incidence, wind_speed)
    direction = np.full_like(incidence, wind_direction)
    return cmod5n_forward(speed, direction, incidence)


def normalize_slc(slc, lut, reference):
    """Return the complex SLC scaled so that its intensity equals sigma0 / reference."""
    return (slc / (lut * np.sqrt(reference))).astype(np.complex64)


def to_db(intensity, eps=1e-10):
    """Convert a linear intensity (power) image to decibels."""
    return 10 * np.log10(intensity + eps)
