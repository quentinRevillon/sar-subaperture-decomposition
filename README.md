# sar-subaperture-decomposition — azimuth subapertures from Sentinel-1 wave-mode SLC

**sar-subaperture-decomposition** re-implements the subaperture decomposition (SD) preprocessing of
Ristea et al. (2022), [*Guided deep learning by subaperture decomposition: ocean patterns from SAR imagery*](https://arxiv.org/abs/2204.04438).
It turns one complex Sentinel-1 wave-mode (WV) imagette into 4 intensity images, each built from a quarter of the
azimuth (Doppler) spectrum, i.e. a slightly different look angle, as multi-channel input for ocean pattern models.

<img src="figures/teaser.png" width="100%" />

*SSR image and its 4 subapertures (dB, 10 x 10 mean filter, decimation by 10) for imagette `-001` of
`S1C_WV_SLC__1SSV_20250907T201714_20250907T202352_004023_007FFA_3CF9`, produced by `decompose.py`.*

## Overview

- Pipeline from the paper: sigma0 normalisation by a reference for a constant 10 m/s wind at 45 degrees -> azimuth
  FFT -> Hamming window compensation (0.75) -> 4 shifted Hamming windows (0.75) -> azimuth iFFT -> intensity
  -> 10 x 10 mean filter -> decimation by 10. The paper does not name the model function for the reference;
  this implementation uses CMOD5.N.
- Scope: preprocessing only. The paper's classification (ResNet-18, InceptionV3 on TenGeoP-SARwv) and unsupervised
  segmentation experiments are not reproduced here, so this repository reports no accuracy figures.
- Written in September 2025 as a study of the paper; this is not the authors' code.

## Installation

```bash
conda create -n subaperture python=3.12 -y && conda activate subaperture
pip install -e .
```

## Quick start

Download a Sentinel-1 WV SLC product (see [Data](#data)), then run on one imagette:

```bash
python decompose.py \
    --safe data/S1C_WV_SLC__1SSV_20250907T201714_20250907T202352_004023_007FFA_3CF9.SAFE \
    --image-id s1c-wv1-slc-vv-20250907t201714-20250907t201717-004023-007ffa-001 \
    --output-dir outputs
```

| Output | Content |
|---|---|
| `outputs/<image-id>_subapertures.npy` | float32 array `(4, ceil(H/10), ceil(W/10))`, linear intensity: `(4, 492, 569)` for the 4918 x 5681 example imagette |
| `outputs/<image-id>_subapertures.png` | SSR image and the 4 subapertures in dB |

Options: `--looks` (number of subapertures, default 4), `--wind-speed` (m/s, default 10), `--wind-direction`
(degrees between radar azimuth and wind, default 45), `--filter-size` (pixels, default 10), `--factor` (decimation, default 10).

From Python:

```python
from subaperture.calibration import load_normalized_slc
from subaperture.decomposition import decompose

slc = load_normalized_slc("data/S1C_WV_SLC__1SSV_20250907T201714_20250907T202352_004023_007FFA_3CF9.SAFE",
                          "s1c-wv1-slc-vv-20250907t201714-20250907t201717-004023-007ffa-001",
                          wind_speed=10.0, wind_direction=45.0)
looks = decompose(slc, n_looks=4)
```

## Repository structure

- `decompose.py`: command-line entry point, one imagette in, subaperture stack and figure out.
- `subaperture/sentinel1.py`: reads the SLC GeoTIFF, the sigmaNought calibration LUT and the incidence angle grid of a .SAFE product.
- `subaperture/calibration.py`: sigma0, CMOD5.N reference, normalisation of the complex SLC, one-call loader from a .SAFE product.
- `subaperture/decomposition.py`: azimuth FFT, Hamming compensation, subaperture windows, iFFT, filtering, decimation.
- `subaperture/cmod5n.py`: CMOD5.N model function, taken from [openwind](https://github.com/nansencenter/openwind) (K. F. Dagestad, NERSC).
- `notebooks/01_subaperture_decomposition.ipynb`: step-by-step demo with intermediate figures (spectrum before and after compensation, windows, subapertures).

## Method notes

- **Complex normalisation.** The paper computes the intensity after the azimuth iFFT, so the FFT acts on complex
  data. The SLC is scaled by `1 / (A_sigma * sqrt(sigma0_CMOD5.N))`, so that its intensity equals sigma0 divided by
  the CMOD5.N reference (called SSR here) while the phase is kept.
- **Hamming compensation** divides the spectrum by a 0.75 Hamming window spanning all azimuth FFT bins. The paper does
  not state the processed Doppler bandwidth; with this choice the spectrum is flat in its centre but its edges stay
  lower (notebook, step 3).
- **Subaperture windows** have length `N // 4` (N = number of azimuth lines), are evenly spaced and do not overlap.
- The CMOD5.N reference uses the incidence angle interpolated from the annotation geolocation grid; the wind is
  constant over the imagette, as in the paper.

## Data

Sentinel-1 WV SLC products are free on the [Copernicus Data Space Ecosystem](https://browser.dataspace.copernicus.eu)
(product type `WV_SLC`). The examples use `S1C_WV_SLC__1SSV_20250907T201714_20250907T202352_004023_007FFA_3CF9`.
Unzip it in `data/`; the code expects the standard .SAFE layout:

```
data/<product>.SAFE/
├── measurement/<image-id>.tiff
└── annotation/
    ├── <image-id>.xml
    └── calibration/calibration-<image-id>.xml
```

`data/` and `outputs/` are not tracked by git.

## References

```bibtex
@article{ristea2022guided,
  title   = {Guided deep learning by subaperture decomposition: ocean patterns from {SAR} imagery},
  author  = {Ristea, Nicolae-Catalin and Anghel, Andrei and Datcu, Mihai and Chapron, Bertrand},
  journal = {arXiv preprint arXiv:2204.04438},
  year    = {2022}
}
```

CMOD5.N implementation: [nansencenter/openwind](https://github.com/nansencenter/openwind/blob/master/openwind/cmod5n.py).

## Author

Quentin Revillon.
