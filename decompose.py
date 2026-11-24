"""Decompose one Sentinel-1 wave-mode SLC imagette into azimuth subapertures.

Usage:
    python decompose.py --safe data/S1C_WV_SLC__1SSV_20250907T201714_20250907T202352_004023_007FFA_3CF9.SAFE \
        --image-id s1c-wv1-slc-vv-20250907t201714-20250907t201717-004023-007ffa-001 --output-dir outputs

Writes to --output-dir:
    <image-id>_subapertures.npy  float32 array (n_looks, ceil(H / factor), ceil(W / factor)), linear SSR intensity
    <image-id>_subapertures.png  SSR image and the subapertures, in dB
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from subaperture.calibration import load_normalized_slc, to_db
from subaperture.decomposition import decimate, decompose, mean_filter


def save_figure(ssr, looks, path):
    """Save the SSR image next to each subaperture, in dB, each stretched to its 2-98 % range."""
    images = [ssr] + list(looks)
    titles = ["SSR (full aperture)"] + [f"Subaperture {k + 1}" for k in range(len(looks))]
    fig, axes = plt.subplots(1, len(images), figsize=(3.2 * len(images), 3.4))
    for ax, image, title in zip(axes, images, titles):
        image_db = to_db(image)
        vmin, vmax = np.percentile(image_db, (2, 98))
        ax.imshow(image_db, vmin=vmin, vmax=vmax, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--safe", required=True, help="Path to the .SAFE product directory")
    parser.add_argument("--image-id", required=True, help="Imagette name without extension")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--looks", type=int, default=4, help="Number of subapertures")
    parser.add_argument("--wind-speed", type=float, default=10.0, help="CMOD5.N reference wind speed, m/s")
    parser.add_argument("--wind-direction", type=float, default=45.0,
                        help="CMOD5.N reference angle between radar azimuth and wind, degrees")
    parser.add_argument("--filter-size", type=int, default=10, help="Moving-average window side, pixels")
    parser.add_argument("--factor", type=int, default=10, help="Decimation factor")
    return parser.parse_args()


def main():
    """Run the decomposition on one imagette and save the array and the figure."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slc = load_normalized_slc(args.safe, args.image_id, args.wind_speed, args.wind_direction)
    looks = decompose(slc, n_looks=args.looks, filter_size=args.filter_size, factor=args.factor)
    np.save(out_dir / f"{args.image_id}_subapertures.npy", looks.astype(np.float32))
    ssr = decimate(mean_filter(np.abs(slc) ** 2, args.filter_size), args.factor)
    save_figure(ssr, looks, out_dir / f"{args.image_id}_subapertures.png")


if __name__ == "__main__":
    main()
