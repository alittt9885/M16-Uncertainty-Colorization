"""
بررسی هدر فایل‌های FITS دانلودشده از M16
------------------------------------------
این اسکریپت برای هر فایل FITS، فیلتر نجومی، ابزار، و زمان نوردهی
(exposure time) رو از هدر می‌خونه و چاپ می‌کنه تا مشخص بشه
کدوم فایل مربوط به کدوم باند (Ha / OIII / SII) هست.

اجرا:
    conda activate astro
    python3 check_fits_headers.py
"""

from astropy.io import fits
import glob
import os

download_dir = os.path.expanduser("~/astro_data/M16")
fits_files = sorted(glob.glob(f"{download_dir}/mastDownload/HST/**/*.fits", recursive=True))

print(f"تعداد فایل‌های FITS پیدا‌شده: {len(fits_files)}\n")

for fpath in fits_files:
    with fits.open(fpath) as hdul:
        header = hdul[0].header
        filt = header.get("FILTER", header.get("FILTER1", "نامشخص"))
        instrument = header.get("INSTRUME", "نامشخص")
        exptime = header.get("EXPTIME", "نامشخص")
        target = header.get("TARGNAME", "نامشخص")
        naxis1 = header.get("NAXIS1", "?")
        naxis2 = header.get("NAXIS2", "?")

        print(f"فایل: {os.path.basename(fpath)}")
        print(f"  هدف: {target}")
        print(f"  ابزار: {instrument}")
        print(f"  فیلتر: {filt}")
        print(f"  زمان نوردهی: {exptime} ثانیه")
        print(f"  ابعاد تصویر: {naxis1} x {naxis2}")
        print("-" * 50)
