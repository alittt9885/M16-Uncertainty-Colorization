"""
Download the Ha band (F656N) for M16 - completing the three-band set
------------------------------------------------------------------------
Previous observations located OIII (F502N) and SII (F673N).
This script specifically looks for Ha (F656N) from the same
proposal/target.

Run:
    conda activate astro
    python3 03_download_ha.py
"""

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

target_name = "M16"
coord = SkyCoord.from_name(target_name)

print("Searching for HST observations with the Ha (F656N) filter...")
obs_table = Observations.query_criteria(
    coordinates=coord,
    radius=0.05 * u.deg,
    obs_collection="HST",
    dataproduct_type="image",
    instrument_name=["WFC3/UVIS"],
    filters="F656N",
)

print(f"Number of observations found with F656N: {len(obs_table)}")

if len(obs_table) == 0:
    print("No observations found with the F656N filter in this area.")
    print("Suggestion: increase the search radius or drop the instrument_name filter.")
    exit()

cols_to_show = ["obs_id", "instrument_name", "filters", "target_name", "t_exptime"]
print("\nResults:")
print(obs_table[cols_to_show])

# Get the product list and keep only drz (the final, combined/drizzled version)
data_products = Observations.get_product_list(obs_table)
science_products = Observations.filter_products(
    data_products,
    productType="SCIENCE",
    extension="fits",
    calib_level=[3],
)

# Keep only drz files (not drc)
drz_mask = [str(row["productFilename"]).endswith("_drz.fits") for row in science_products]
science_products = science_products[drz_mask]

print(f"\nNumber of drz files available for download: {len(science_products)}")

if len(science_products) == 0:
    print("No drz file found.")
    exit()

# Only download the first 2 files (to avoid excessive data volume)
science_products = science_products[:2]

download_dir = os.path.expanduser("~/astro_data/M16")
os.makedirs(download_dir, exist_ok=True)

print(f"\nStarting download to: {download_dir}")
manifest = Observations.download_products(
    science_products,
    download_dir=download_dir,
)

print("\nHa download complete.")
print(manifest)
