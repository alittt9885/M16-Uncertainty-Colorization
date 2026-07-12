"""
دانلود داده‌ی چندفیلتری هابل از سحابی M16 (ستون‌های آفرینش) از MAST
------------------------------------------------------------------
این اسکریپت با astroquery.mast مشاهدات هابل مربوط به M16 رو جستجو
و فایل‌های FITS مربوط به فیلترهای مختلف رو دانلود می‌کنه.

اجرا (داخل محیط conda به اسم astro):
    conda activate astro
    python3 download_m16_mast.py
"""

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# ---------------------------------------------------------
# ۱. مختصات هدف: M16 / Eagle Nebula / Pillars of Creation
# ---------------------------------------------------------
target_name = "M16"
coord = SkyCoord.from_name(target_name)
print(f"مختصات {target_name}: RA={coord.ra.deg:.4f}, Dec={coord.dec.deg:.4f}")

# ---------------------------------------------------------
# ۲. جستجوی مشاهدات هابل (HST) در شعاع مشخص
# ---------------------------------------------------------
print("\nدر حال جستجوی مشاهدات HST...")
obs_table = Observations.query_criteria(
    coordinates=coord,
    radius=0.05 * u.deg,          # شعاع جستجو حدود ۳ دقیقه‌ی قوسی
    obs_collection="HST",
    dataproduct_type="image",
    instrument_name=["WFC3/UVIS", "WFC3/IR", "ACS/WFC"],
)

print(f"تعداد مشاهدات یافت‌شده: {len(obs_table)}")

if len(obs_table) == 0:
    print("هیچ مشاهده‌ای پیدا نشد. شعاع جستجو یا فیلترها رو تغییر بده.")
    exit()

# نمایش چند ستون مهم برای بررسی دستی
cols_to_show = ["obs_id", "instrument_name", "filters", "target_name", "t_exptime"]
print("\nنمونه‌ای از نتایج:")
print(obs_table[cols_to_show][:10])

# ---------------------------------------------------------
# ۳. فیلتر کردن بر اساس فیلترهای نجومی موردنظر (مثال: Ha, OIII, SII)
#    فیلترهای هابل معمولاً با نام‌هایی مثل F656N (Ha), F502N (OIII),
#    F673N (SII) مشخص می‌شن. اینجا هر فیلتری که شامل این کدها باشه نگه می‌داریم.
# ---------------------------------------------------------
target_filters = ["F656N", "F502N", "F673N", "F658N"]  # Ha, OIII, SII, [NII]
mask = [
    any(f in str(row["filters"]) for f in target_filters)
    for row in obs_table
]
filtered_obs = obs_table[mask]
print(f"\nمشاهدات با فیلترهای هدف (Ha/OIII/SII): {len(filtered_obs)}")

if len(filtered_obs) == 0:
    print("هیچ مشاهده‌ای با این فیلترهای خاص پیدا نشد.")
    print("فیلترهای موجود در نتایج کلی:", set(obs_table["filters"]))
    exit()

# ---------------------------------------------------------
# ۴. دریافت لیست محصولات داده (data products) برای این مشاهدات
# ---------------------------------------------------------
print("\nدر حال دریافت لیست فایل‌های قابل دانلود...")
data_products = Observations.get_product_list(filtered_obs)

# فقط فایل‌های علمی سطح نهایی (کالیبره‌شده) رو نگه می‌داریم
science_products = Observations.filter_products(
    data_products,
    productType="SCIENCE",
    extension="fits",
    calib_level=[3],  # سطح کالیبراسیون نهایی
)

print(f"تعداد فایل‌های علمی برای دانلود: {len(science_products)}")

if len(science_products) == 0:
    print("فایل علمی سطح ۳ پیدا نشد. سطح کالیبراسیون رو تغییر بده (مثلاً [2, 3]).")
    exit()

# محدود کردن تعداد دانلود برای تست اولیه (مثلاً فقط ۵ فایل اول)
MAX_DOWNLOADS = 5
science_products = science_products[:MAX_DOWNLOADS]

# ---------------------------------------------------------
# ۵. دانلود فایل‌ها
# ---------------------------------------------------------
download_dir = os.path.expanduser("~/astro_data/M16")
os.makedirs(download_dir, exist_ok=True)

print(f"\nشروع دانلود {len(science_products)} فایل به مسیر: {download_dir}")
manifest = Observations.download_products(
    science_products,
    download_dir=download_dir,
)

print("\n✅ دانلود کامل شد.")
print(manifest)
