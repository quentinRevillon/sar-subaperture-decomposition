"""Radiometric calibration and CMOD5.N normalisation of a complex SLC image.

The paper divides sigma0 by a reference computed for a constant 10 m/s wind at 45 degrees from the
antenna look direction, without naming the model function; this implementation uses CMOD5.N.
The resulting ratio is called sea surface roughness (SSR) here.
To keep the phase needed by the azimuth FFT, the normalisation is applied to the complex signal:
|normalize_slc(...)|**2 equals SSR.

Usage:
    from subaperture.calibration import load_normalized_slc, sigma0, cmod5n_reference, normalize_slc, to_db

    slc_norm = load_normalized_slc("data/S1C_WV_SLC.SAFE", "s1c-wv1-slc-vv-...-001")  # one call from a .SAFE

    reference = cmod5n_reference(incidence)            # linear sigma0 of the reference wind
    ssr = sigma0(slc, lut) / reference                  # real SSR image
    slc_norm = normalize_slc(slc, lut, reference)       # complex image with |slc_norm|**2 == ssr
    ssr_db = to_db(ssr)
"""

import numpy as np

from subaperture.cmod5n import cmod5n_forward
from subaperture.sentinel1 import product_paths, read_incidence_angle, read_sigma0_lut, read_slc


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


def load_normalized_slc(safe_dir, image_id, wind_speed=10.0, wind_direction=45.0):
    """Read one imagette of a .SAFE product and return its CMOD5.N-normalised complex SLC (|.|^2 = SSR).

    Args:
        safe_dir: path to the .SAFE directory.
        image_id: imagette name without extension.
        wind_speed: reference wind speed in m/s.
        wind_direction: angle between radar azimuth and wind in degrees.
    """
    paths = product_paths(safe_dir, image_id)
    slc = read_slc(paths["measurement"])
    lut = read_sigma0_lut(paths["calibration"], slc.shape)
    incidence = read_incidence_angle(paths["annotation"], slc.shape)
    return normalize_slc(slc, lut, cmod5n_reference(incidence, wind_speed, wind_direction))


def to_db(intensity, eps=1e-10):
    """Convert a linear intensity (power) image to decibels."""
    return 10 * np.log10(intensity + eps)
