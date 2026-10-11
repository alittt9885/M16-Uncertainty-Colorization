"""
Baseline: M16 three-band composite into the Hubble Palette with a percentile stretch
---------------------------------------------------------------------------------------
This script implements the baseline (no machine learning):
    SII (F673N) -> Red
    Ha  (F656N) -> Green
    OIII(F502N) -> Blue

Steps:
    1. Read the science (SCI) extension from each FITS file
    2. Register (reproject) all three bands onto a common reference grid
    3. Percentile stretch for contrast enhancement
    4. Combine into a three-channel image and save as PNG

Run:
    conda activate astro
    python3 04_baseline_colorization.py

Changes relative to the previous version:
    - Fixed SCI extension selection (now looks specifically for SCI
      first, and only falls back to 2D data if that's not found)
    - Added error handling for missing files
    - Added shape logging before and after reproject
    - Reference band is now a configurable variable
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

DATA_DIR = os.path.expanduser("~/astro_data/M16/mastDownload/HST")
OUTPUT_DIR = os.path.expanduser("~/astro_data/M16/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact file paths (based on what was downloaded)
files = {
    "SII": f"{DATA_DIR}/ick905050/ick905050_drc.fits",
    "HA": f"{DATA_DIR}/iehc96klq/iehc96klq_drz.fits",
    "OIII": f"{DATA_DIR}/ick905030/ick905030_drz.fits",
}

# Reference band for reproject - can be changed here
REFERENCE_BAND = "OIII"


def load_science_data(filepath):
    """Read the data and WCS from the science (SCI) extension of a FITS file.

    First looks specifically for an extension named SCI. If that's not
    found, falls back to the first extension with 2D data (since some
    older HST files may use a different extension name).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with fits.open(filepath) as hdul:
        sci_hdu = None

        # First priority: an extension named exactly SCI
        for hdu in hdul:
            if hdu.name == "SCI":
                sci_hdu = hdu
                break

        # Second priority (fallback): any extension with 2D data
        if sci_hdu is None:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    sci_hdu = hdu
                    print(
                        f"  WARNING: SCI extension not found in {os.path.basename(filepath)}; "
                        f"using extension '{hdu.name}' instead."
                    )
                    break

        if sci_hdu is None:
            raise ValueError(f"No valid science extension found in {filepath}.")

        data = sci_hdu.data.astype(np.float32)
        wcs = WCS(sci_hdu.header)

    return data, wcs


def percentile_stretch(data, low=1, high=99.5):
    """Simple percentile stretch to improve image contrast."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("No finite data available to stretch.")
    vmin, vmax = np.percentile(finite, [low, high])
    if vmax <= vmin:
        raise ValueError("Invalid stretch range (vmax <= vmin).")
    stretched = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    return stretched


def main():
    print("Reading bands...")
    bands = {}
    for name, path in files.items():
        try:
            data, wcs = load_science_data(path)
            bands[name] = (data, wcs)
            print(f"  OK {name}: {data.shape}  <- {path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ERROR reading band {name}: {e}")
            sys.exit(1)

    sii_data, sii_wcs = bands["SII"]
    ha_data, ha_wcs = bands["HA"]
    oiii_data, oiii_wcs = bands["OIII"]

    # ---------------------------------------------------------
    # Registration: treat the reference band as the reference grid
    # and reproject the other two bands onto that same grid
    # ---------------------------------------------------------
    ref_data, ref_wcs = bands[REFERENCE_BAND]
    ref_shape = ref_data.shape
    print(f"\nReference band: {REFERENCE_BAND}  (shape={ref_shape})")

    print("Registering (reprojecting) bands...")
    aligned = {}
    for name, (data, wcs) in bands.items():
        if name == REFERENCE_BAND:
            aligned[name] = data
            continue
        print(f"  Reprojecting {name}: {data.shape} -> {ref_shape}")
        reprojected, footprint = reproject_interp((data, wcs), ref_wcs, shape_out=ref_shape)
        coverage = np.mean(footprint > 0) * 100
        print(f"    Coverage after reproject: {coverage:.1f}%")
        aligned[name] = np.nan_to_num(reprojected)

    sii_aligned = aligned["SII"]
    ha_aligned = aligned["HA"]
    oiii_aligned = np.nan_to_num(aligned["OIII"])

    # ---------------------------------------------------------
    # Crop to the largest rectangle that lies entirely within the
    # valid region (covered by all three bands).
    # Because the Ha field of view is slightly rotated relative to
    # OIII/SII, the true common-coverage region is a rotated
    # quadrilateral rather than an axis-aligned rectangle. If we just
    # took the naive bounding box, its corners would still fall
    # outside the valid region. So we use the classic "maximal
    # rectangle in a binary matrix" algorithm to find a rectangle
    # that is 100% inside the valid region (no invalid pixels at all).
    # ---------------------------------------------------------
    def largest_valid_rectangle(mask):
        """Finds the largest all-True rectangle inside a boolean matrix.
        Standard, well-tested implementation of the classic "Maximal
        Rectangle" problem (using a monotonic stack over each row's
        histogram).
        Returns: (rmin, rmax, cmin, cmax) - an inclusive range.
        """
        n_rows, n_cols = mask.shape
        height = np.zeros(n_cols, dtype=np.int64)
        best_area = 0
        best_rect = (0, 0, 0, 0)

        for r in range(n_rows):
            height = np.where(mask[r], height + 1, 0)
            h_list = height.tolist()
            h_list.append(0)  # sentinel to flush the stack at the end of each row

            stack = [-1]  # column indices; -1 = virtual left boundary
            for c in range(n_cols + 1):
                while h_list[c] < h_list[stack[-1]]:
                    h = h_list[stack.pop()]
                    left = stack[-1] + 1
                    width = c - left
                    area = h * width
                    if area > best_area:
                        best_area = area
                        best_rect = (r - h + 1, r, left, c - 1)
                stack.append(c)

        return best_rect

    print("\nFinding the common (crop) region across all three bands...")
    valid_mask = (sii_aligned != 0) & (ha_aligned != 0) & (oiii_aligned != 0)

    if not valid_mask.any():
        print("  WARNING: no common region found between the three bands; skipping crop.")
    else:
        print("  Searching for the largest fully valid rectangle (may take a few seconds)...")
        rmin, rmax, cmin, cmax = largest_valid_rectangle(valid_mask)
        crop_coverage = valid_mask[rmin:rmax+1, cmin:cmax+1].mean() * 100
        print(f"  Crop region: rows [{rmin}:{rmax}], columns [{cmin}:{cmax}]")
        print(f"  Dimensions after crop: {rmax-rmin+1} x {cmax-cmin+1}")
        print(f"  Percentage of valid pixels in this region: {crop_coverage:.1f}% (should be 100%)")

        if crop_coverage < 99.9:
            print("  WARNING: the rectangle found is not fully valid! Falling back to a simple bounding box.")
            rows_any = np.any(valid_mask, axis=1)
            cols_any = np.any(valid_mask, axis=0)
            rmin, rmax = np.where(rows_any)[0][[0, -1]]
            cmin, cmax = np.where(cols_any)[0][[0, -1]]

        sii_aligned = sii_aligned[rmin:rmax+1, cmin:cmax+1]
        ha_aligned = ha_aligned[rmin:rmax+1, cmin:cmax+1]
        oiii_aligned = oiii_aligned[rmin:rmax+1, cmin:cmax+1]

    # ---------------------------------------------------------
    # Percentile stretch on each band
    # ---------------------------------------------------------
    print("\nApplying percentile stretch...")
    r_channel = percentile_stretch(sii_aligned)   # SII  -> Red
    g_channel = percentile_stretch(ha_aligned)    # Ha   -> Green
    b_channel = percentile_stretch(oiii_aligned)  # OIII -> Blue

    rgb_image = np.dstack([r_channel, g_channel, b_channel])

    # ---------------------------------------------------------
    # Save the output
    # ---------------------------------------------------------
    output_path = os.path.join(OUTPUT_DIR, "m16_hubble_palette_baseline.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image, origin="lower")
    plt.axis("off")
    plt.title("M16 - Hubble Palette (Baseline, percentile-stretch)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\nBaseline image saved to: {output_path}")

    # ---------------------------------------------------------
    # Display the image.
    # On Windows 11, WSLg is enabled by default and plt.show() should
    # open a graphical window directly. If for any reason (e.g. WSLg
    # not enabled) the window doesn't open, the saved file is opened
    # automatically with Windows Explorer instead (fallback).
    # ---------------------------------------------------------
    try:
        plt.show()
    except Exception as e:
        print(f"  WARNING: direct graphical display failed ({e})")
    finally:
        plt.close()

    # fallback: open the file with the default Windows viewer
    try:
        import subprocess
        windows_path = subprocess.check_output(
            ["wslpath", "-w", output_path], text=True
        ).strip()
        subprocess.Popen(["explorer.exe", windows_path])
    except Exception as e:
        print(f"  WARNING: could not auto-open with Explorer ({e})")
        print(f"  You can open this path manually in File Explorer: {output_path}")

    npz_path = os.path.join(OUTPUT_DIR, "m16_bands_aligned.npz")
    np.savez(
        npz_path,
        sii=sii_aligned,
        ha=ha_aligned,
        oiii=oiii_aligned,
    )
    print(f"Raw aligned band arrays saved to: {npz_path}")


if __name__ == "__main__":
    main()
