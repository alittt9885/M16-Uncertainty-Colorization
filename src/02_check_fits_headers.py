"""
Inspect the headers of the FITS files downloaded for M16
----------------------------------------------------------
For each FITS file, this script reads the astronomical filter,
instrument, and exposure time from the header and prints them, so it's
clear which file corresponds to which band (Ha / OIII / SII).

Run:
    conda activate astro
    python3 02_check_fits_headers.py
"""

from astropy.io import fits
import glob
import os

download_dir = os.path.expanduser("~/astro_data/M16")
fits_files = sorted(glob.glob(f"{download_dir}/mastDownload/HST/**/*.fits", recursive=True))

print(f"Number of FITS files found: {len(fits_files)}\n")

for fpath in fits_files:
    with fits.open(fpath) as hdul:
        header = hdul[0].header
        filt = header.get("FILTER", header.get("FILTER1", "unknown"))
        instrument = header.get("INSTRUME", "unknown")
        exptime = header.get("EXPTIME", "unknown")
        target = header.get("TARGNAME", "unknown")
        naxis1 = header.get("NAXIS1", "?")
        naxis2 = header.get("NAXIS2", "?")

        print(f"File: {os.path.basename(fpath)}")
        print(f"  Target: {target}")
        print(f"  Instrument: {instrument}")
        print(f"  Filter: {filt}")
        print(f"  Exposure time: {exptime} s")
        print(f"  Image dimensions: {naxis1} x {naxis2}")
        print("-" * 50)
