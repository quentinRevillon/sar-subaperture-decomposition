"""Readers for one Sentinel-1 SLC imagette (wave mode) inside a .SAFE product.

Usage:
    from subaperture.sentinel1 import product_paths, read_slc, read_sigma0_lut, read_incidence_angle

    paths = product_paths("data/S1C_WV_SLC.SAFE", "s1c-wv1-slc-vv-20250907t201714-20250907t201717-004023-007ffa-001")
    slc = read_slc(paths["measurement"])                          # complex64, (azimuth, range)
    lut = read_sigma0_lut(paths["calibration"], slc.shape)         # sigmaNought LUT, same shape
    incidence = read_incidence_angle(paths["annotation"], slc.shape)  # degrees, same shape
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


def product_paths(safe_dir, image_id):
    """Return the measurement, annotation and calibration file paths of one imagette.

    Args:
        safe_dir: path to the .SAFE directory.
        image_id: imagette name without extension, e.g. "s1c-wv1-slc-vv-...-001".

    Returns:
        Dict with keys "measurement", "annotation" and "calibration" (pathlib.Path).
    """
    safe_dir = Path(safe_dir)
    return {
        "measurement": safe_dir / "measurement" / f"{image_id}.tiff",
        "annotation": safe_dir / "annotation" / f"{image_id}.xml",
        "calibration": safe_dir / "annotation" / "calibration" / f"calibration-{image_id}.xml",
    }


def read_slc(path):
    """Return the complex SLC image of a measurement GeoTIFF, shape (azimuth, range)."""
    slc = tifffile.imread(path)
    assert np.iscomplexobj(slc), f"Expected a complex SLC, got {slc.dtype}"
    return slc


def interpolate_grid(lines, pixels, values, shape):
    """Bilinearly interpolate values given on a regular (lines x pixels) tie-point grid.

    Args:
        lines: 1D increasing array of azimuth indices of the grid rows.
        pixels: 1D increasing array of range indices of the grid columns.
        values: 2D array of shape (len(lines), len(pixels)).
        shape: (height, width) of the output image.

    Returns:
        2D float32 array of the given shape (edge values are repeated outside the grid).
    """
    along_range = np.stack([np.interp(np.arange(shape[1]), pixels, row) for row in values])
    rows = np.arange(shape[0])
    full = np.empty(shape, dtype=np.float32)
    for col in range(shape[1]):
        full[:, col] = np.interp(rows, lines, along_range[:, col])
    return full


def read_sigma0_lut(calibration_xml, shape):
    """Return the sigmaNought calibration LUT interpolated to the full image grid.

    Args:
        calibration_xml: path to the calibration-*.xml annotation file.
        shape: (height, width) of the SLC image.
    """
    vectors = ET.parse(calibration_xml).getroot().findall(".//calibrationVector")
    lines = np.array([int(v.findtext("line")) for v in vectors])
    pixels = np.array(vectors[0].findtext("pixel").split(), dtype=int)
    values = np.array([v.findtext("sigmaNought").split() for v in vectors], dtype=np.float32)
    return interpolate_grid(lines, pixels, values, shape)


def read_incidence_angle(annotation_xml, shape):
    """Return the incidence angle (degrees) interpolated from the geolocation grid.

    Args:
        annotation_xml: path to the product annotation *.xml file.
        shape: (height, width) of the SLC image.
    """
    points = ET.parse(annotation_xml).getroot().findall(".//geolocationGridPoint")
    lines = np.array([int(p.findtext("line")) for p in points])
    pixels = np.array([int(p.findtext("pixel")) for p in points])
    angles = np.array([float(p.findtext("incidenceAngle")) for p in points])
    grid_lines, grid_pixels = np.unique(lines), np.unique(pixels)
    assert grid_lines.size * grid_pixels.size == angles.size, "Geolocation grid is not regular"
    values = angles[np.lexsort((pixels, lines))].reshape(grid_lines.size, grid_pixels.size)
    return interpolate_grid(grid_lines, grid_pixels, values, shape)


def read_azimuth_time_interval(annotation_xml):
    """Return the azimuth sampling interval in seconds (inverse of the PRF of the image)."""
    return float(ET.parse(annotation_xml).getroot().findtext(".//imageInformation/azimuthTimeInterval"))
