"""
Download multi-filter Hubble data of the M16 nebula (Pillars of Creation) from MAST
------------------------------------------------------------------------------------
This script uses astroquery.mast to search for Hubble observations of M16
and download the FITS files for the filters of interest.

Run (inside a conda environment named "astro"):
    conda activate astro
    python3 01_download_oiii_sii.py
"""

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# ---------------------------------------------------------
# 1. Target coordinates: M16 / Eagle Nebula / Pillars of Creation
# ---------------------------------------------------------
target_name = "M16"
coord = SkyCoord.from_name(target_name)
print(f"Coordinates for {target_name}: RA={coord.ra.deg:.4f}, Dec={coord.dec.deg:.4f}")

# ---------------------------------------------------------
# 2. Search for Hubble (HST) observations within a given radius
# ---------------------------------------------------------
print("\nSearching for HST observations...")
obs_table = Observations.query_criteria(
    coordinates=coord,
    radius=0.05 * u.deg,          # search radius, roughly 3 arcmin
    obs_collection="HST",
    dataproduct_type="image",
    instrument_name=["WFC3/UVIS", "WFC3/IR", "ACS/WFC"],
)

print(f"Number of observations found: {len(obs_table)}")

if len(obs_table) == 0:
    print("No observations found. Try a larger search radius or different filters.")
    exit()

# Show a few key columns for manual inspection
cols_to_show = ["obs_id", "instrument_name", "filters", "target_name", "t_exptime"]
print("\nSample of results:")
print(obs_table[cols_to_show][:10])

# ---------------------------------------------------------
# 3. Filter down to the astronomical filters of interest (Ha, OIII, SII)
#    Hubble filters are usually named like F656N (Ha), F502N (OIII),
#    F673N (SII). Here we keep any observation whose filter string
#    contains one of these codes.
# ---------------------------------------------------------
target_filters = ["F656N", "F502N", "F673N", "F658N"]  # Ha, OIII, SII, [NII]
mask = [
    any(f in str(row["filters"]) for f in target_filters)
    for row in obs_table
]
filtered_obs = obs_table[mask]
print(f"\nObservations matching target filters (Ha/OIII/SII): {len(filtered_obs)}")

if len(filtered_obs) == 0:
    print("No observations found with these specific filters.")
    print("Filters available in the full result set:", set(obs_table["filters"]))
    exit()

# ---------------------------------------------------------
# 4. Get the list of data products for these observations
# ---------------------------------------------------------
print("\nFetching the list of downloadable files...")
data_products = Observations.get_product_list(filtered_obs)

# Keep only final, calibrated science-level files
science_products = Observations.filter_products(
    data_products,
    productType="SCIENCE",
    extension="fits",
    calib_level=[3],  # final calibration level
)

print(f"Number of science files available for download: {len(science_products)}")

if len(science_products) == 0:
    print("No calibration-level-3 science files found. Try a different calib_level (e.g. [2, 3]).")
    exit()

# Cap the number of downloads for an initial test run (e.g. first 5 files only)
MAX_DOWNLOADS = 5
science_products = science_products[:MAX_DOWNLOADS]

# ---------------------------------------------------------
# 5. Download the files
# ---------------------------------------------------------
download_dir = os.path.expanduser("~/astro_data/M16")
os.makedirs(download_dir, exist_ok=True)

print(f"\nStarting download of {len(science_products)} file(s) to: {download_dir}")
manifest = Observations.download_products(
    science_products,
    download_dir=download_dir,
)

print("\nDownload complete.")
print(manifest)
