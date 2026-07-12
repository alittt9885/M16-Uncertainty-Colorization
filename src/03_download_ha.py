"""
دانلود باند Ha (F656N) از M16 - تکمیل مجموعه‌ی سه‌باندی
----------------------------------------------------------
مشاهدات قبلی OIII (F502N) و SII (F673N) رو پیدا کردن.
این اسکریپت مشخصاً دنبال Ha (F656N) از همون پروپوزال/هدف می‌گرده.

اجرا:
    conda activate astro
    python3 download_ha_m16.py
"""

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

target_name = "M16"
coord = SkyCoord.from_name(target_name)

print("در حال جستجوی مشاهدات HST با فیلتر Ha (F656N)...")
obs_table = Observations.query_criteria(
    coordinates=coord,
    radius=0.05 * u.deg,
    obs_collection="HST",
    dataproduct_type="image",
    instrument_name=["WFC3/UVIS"],
    filters="F656N",
)

print(f"تعداد مشاهدات یافت‌شده با F656N: {len(obs_table)}")

if len(obs_table) == 0:
    print("هیچ مشاهده‌ای با فیلتر F656N در این ناحیه پیدا نشد.")
    print("پیشنهاد: شعاع جستجو رو زیاد کن یا instrument_name رو بردار.")
    exit()

cols_to_show = ["obs_id", "instrument_name", "filters", "target_name", "t_exptime"]
print("\nنتایج:")
print(obs_table[cols_to_show])

# دریافت محصولات و فیلتر فقط drz (نسخه‌ی نهایی ترکیب‌شده)
data_products = Observations.get_product_list(obs_table)
science_products = Observations.filter_products(
    data_products,
    productType="SCIENCE",
    extension="fits",
    calib_level=[3],
)

# فقط فایل‌های drz رو نگه دار (نه drc)
drz_mask = [str(row["productFilename"]).endswith("_drz.fits") for row in science_products]
science_products = science_products[drz_mask]

print(f"\nتعداد فایل‌های drz برای دانلود: {len(science_products)}")

if len(science_products) == 0:
    print("فایل drz پیدا نشد.")
    exit()

# فقط ۲ فایل اول رو دانلود کن (برای جلوگیری از حجم زیاد)
science_products = science_products[:2]

download_dir = os.path.expanduser("~/astro_data/M16")
os.makedirs(download_dir, exist_ok=True)

print(f"\nشروع دانلود به مسیر: {download_dir}")
manifest = Observations.download_products(
    science_products,
    download_dir=download_dir,
)

print("\n✅ دانلود Ha کامل شد.")
print(manifest)
