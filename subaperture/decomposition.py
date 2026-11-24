"""Subaperture decomposition (SD) of a complex SAR image along azimuth (Ristea et al., 2022).

Steps: azimuth FFT -> Hamming window compensation (0.75) -> N shifted Hamming windows (0.75)
-> azimuth iFFT -> intensity -> 10 x 10 mean filter -> decimation by 10.
Axis 0 of every image is azimuth, axis 1 is range.

Usage:
    from subaperture.decomposition import decompose

    looks = decompose(slc_norm)                  # (4, ceil(H / 10), ceil(W / 10)) intensity images
    looks = decompose(slc_norm, n_looks=3, filter_size=5, factor=5)
"""

import numpy as np
from scipy import fft
from scipy.ndimage import uniform_filter


def hamming_window(n, alpha=0.75):
    """Return a centred generalised Hamming window of length n: alpha + (1 - alpha) cos(2 pi k / (n - 1))."""
    k = np.arange(n) - (n - 1) / 2
    return (alpha + (1 - alpha) * np.cos(2 * np.pi * k / (n - 1))).astype(np.float32)


def azimuth_spectrum(image):
    """Return the azimuth spectrum (FFT along axis 0, zero frequency centred, complex64 kept)."""
    return fft.fftshift(fft.fft(image, axis=0), axes=0)


def azimuth_frequencies(n, time_interval):
    """Return the centred azimuth frequency axis in Hz for n samples spaced by time_interval seconds."""
    return fft.fftshift(fft.fftfreq(n, d=time_interval))


def compensate_hamming(spectrum, alpha=0.75):
    """Divide the azimuth spectrum by the processing Hamming window to flatten it."""
    return spectrum / hamming_window(spectrum.shape[0], alpha)[:, None]


def subaperture_windows(n, n_looks=4, alpha=0.75):
    """Return n_looks Hamming windows of length n // n_looks, shifted across a band of n bins.

    Windows are evenly spaced, the first starts at bin 0 and the last ends at bin n.

    Returns:
        Array of shape (n_looks, n), zero outside each window.
    """
    size = n // n_looks
    starts = np.linspace(0, n - size, n_looks).astype(int)
    windows = np.zeros((n_looks, n), dtype=np.float32)
    for look, start in enumerate(starts):
        windows[look, start:start + size] = hamming_window(size, alpha)
    return windows


def subaperture_intensity(spectrum, window):
    """Return the intensity image of one subaperture: |iFFT(window * spectrum)|^2."""
    look = fft.ifft(fft.ifftshift(spectrum * window[:, None], axes=0), axis=0)
    return np.abs(look) ** 2


def mean_filter(image, size=10):
    """Low-pass filter an image with a size x size moving average (coefficients 1 / size^2)."""
    return uniform_filter(image, size=size, mode="reflect")


def decimate(image, factor=10):
    """Keep one pixel out of factor along both axes."""
    return image[::factor, ::factor]


def decompose(image, n_looks=4, alpha=0.75, filter_size=10, factor=10):
    """Return the decimated intensity images of the n_looks azimuth subapertures.

    Args:
        image: complex image (azimuth, range), e.g. the CMOD5.N-normalised SLC.
        n_looks: number of subapertures.
        alpha: Hamming coefficient used for compensation and for the subaperture windows.
        filter_size: side of the moving-average window, in pixels.
        factor: decimation factor.

    Returns:
        Array of shape (n_looks, ceil(H / factor), ceil(W / factor)).
    """
    spectrum = compensate_hamming(azimuth_spectrum(image), alpha)
    windows = subaperture_windows(image.shape[0], n_looks, alpha)
    return np.stack([decimate(mean_filter(subaperture_intensity(spectrum, w), filter_size), factor)
                     for w in windows])
