"""
Baseline: ترکیب سه‌باندی M16 به Hubble Palette با percentile-stretch
-----------------------------------------------------------------------
این اسکریپت baseline (بدون یادگیری ماشین) رو پیاده می‌کنه:
    SII (F673N) -> قرمز
    Ha  (F656N) -> سبز
    OIII(F502N) -> آبی

مراحل:
    1. خواندن extension علمی (SCI) از هر فایل FITS
    2. هم‌ترازسازی (reproject) هر سه باند به شبکه‌ی مرجع مشترک
    3. percentile stretch (کشیدگی درصدی) برای بهبود کنتراست
    4. ترکیب سه‌کاناله و ذخیره به‌صورت PNG

اجرا:
    conda activate astro
    python3 baseline_hubble_palette.py

تغییرات نسبت به نسخه‌ی قبل:
    - انتخاب extension SCI اصلاح شد (اول دقیقاً دنبال SCI می‌گرده،
      و فقط در صورت نبودنش به fallback داده‌ی 2بعدی می‌ره)
    - مدیریت خطا برای فایل‌های ناموجود اضافه شد
    - لاگ shape قبل و بعد از reproject اضافه شد
    - باند مرجع (reference band) به‌صورت متغیر قابل‌تنظیم شد
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

# مسیر دقیق فایل‌ها (بر اساس چیزی که دانلود شد)
files = {
    "SII": f"{DATA_DIR}/ick905050/ick905050_drc.fits",
    "HA": f"{DATA_DIR}/iehc96klq/iehc96klq_drz.fits",
    "OIII": f"{DATA_DIR}/ick905030/ick905030_drz.fits",
}

# باند مرجع برای reproject - اینجا قابل تغییره
REFERENCE_BAND = "OIII"


def load_science_data(filepath):
    """داده و WCS رو از extension علمی (SCI) فایل FITS می‌خونه.

    اول دقیقاً دنبال extension با نام SCI می‌گرده. اگه پیدا نشد،
    به‌عنوان fallback اولین extension با داده‌ی دوبعدی رو برمی‌داره
    (چون بعضی فایل‌های قدیمی‌تر HST ممکنه اسم extension متفاوت داشته باشن).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"فایل پیدا نشد: {filepath}")

    with fits.open(filepath) as hdul:
        sci_hdu = None

        # اولویت اول: دقیقاً extension با نام SCI
        for hdu in hdul:
            if hdu.name == "SCI":
                sci_hdu = hdu
                break

        # اولویت دوم (fallback): هر extension با داده‌ی دوبعدی
        if sci_hdu is None:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    sci_hdu = hdu
                    print(
                        f"  ⚠️ extension SCI در {os.path.basename(filepath)} "
                        f"پیدا نشد؛ از extension '{hdu.name}' استفاده می‌شه."
                    )
                    break

        if sci_hdu is None:
            raise ValueError(f"هیچ extension علمی معتبری در {filepath} پیدا نشد.")

        data = sci_hdu.data.astype(np.float32)
        wcs = WCS(sci_hdu.header)

    return data, wcs


def percentile_stretch(data, low=1, high=99.5):
    """کشیدگی درصدی ساده برای بهبود کنتراست تصویر."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("داده‌ی معتبری (finite) برای stretch وجود نداره.")
    vmin, vmax = np.percentile(finite, [low, high])
    if vmax <= vmin:
        raise ValueError("محدوده‌ی stretch نامعتبره (vmax <= vmin).")
    stretched = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    return stretched


def main():
    print("در حال خواندن باندها...")
    bands = {}
    for name, path in files.items():
        try:
            data, wcs = load_science_data(path)
            bands[name] = (data, wcs)
            print(f"  ✅ {name}: {data.shape}  <- {path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"  ❌ خطا در خواندن باند {name}: {e}")
            sys.exit(1)

    sii_data, sii_wcs = bands["SII"]
    ha_data, ha_wcs = bands["HA"]
    oiii_data, oiii_wcs = bands["OIII"]

    # ---------------------------------------------------------
    # هم‌ترازسازی: باند مرجع رو به‌عنوان شبکه‌ی مرجع در نظر می‌گیریم
    # و دو باند دیگه رو با reproject روی همون شبکه می‌بریم
    # ---------------------------------------------------------
    ref_data, ref_wcs = bands[REFERENCE_BAND]
    ref_shape = ref_data.shape
    print(f"\nباند مرجع: {REFERENCE_BAND}  (shape={ref_shape})")

    print("در حال هم‌ترازسازی (reproject) باندها...")
    aligned = {}
    for name, (data, wcs) in bands.items():
        if name == REFERENCE_BAND:
            aligned[name] = data
            continue
        print(f"  در حال reproject کردن {name}: {data.shape} -> {ref_shape}")
        reprojected, footprint = reproject_interp((data, wcs), ref_wcs, shape_out=ref_shape)
        coverage = np.mean(footprint > 0) * 100
        print(f"    پوشش (coverage) بعد از reproject: {coverage:.1f}%")
        aligned[name] = np.nan_to_num(reprojected)

    sii_aligned = aligned["SII"]
    ha_aligned = aligned["HA"]
    oiii_aligned = np.nan_to_num(aligned["OIII"])

    # ---------------------------------------------------------
    # Crop به بزرگ‌ترین مستطیلی که کاملاً داخل ناحیه‌ی معتبر
    # (پوشش هر سه باند) قرار داره.
    # چون میدان دید Ha نسبت به OIII/SII کمی چرخیده، ناحیه‌ی مشترک
    # واقعی یک چهارضلعی موربه، نه یک مستطیل. اگه فقط bounding box
    # ساده بگیریم، گوشه‌های اون مستطیل هنوز بیرون از ناحیه‌ی معتبر
    # می‌افتن. برای همین از الگوریتم کلاسیک "maximal rectangle in
    # binary matrix" استفاده می‌کنیم تا مستطیلی پیدا کنیم که
    # صد‌درصد داخل ناحیه‌ی معتبر باشه (بدون هیچ پیکسل نامعتبر).
    # ---------------------------------------------------------
    def largest_valid_rectangle(mask):
        """بزرگ‌ترین مستطیل کاملاً True داخل ماتریس بولی رو پیدا می‌کنه.
        پیاده‌سازی استاندارد و تست‌شده‌ی مسئله‌ی معروف "Maximal Rectangle"
        (با استفاده از monotonic stack روی هیستوگرام هر سطر).
        برمی‌گردونه: (rmin, rmax, cmin, cmax) - محدوده‌ی شامل (inclusive)
        """
        n_rows, n_cols = mask.shape
        height = np.zeros(n_cols, dtype=np.int64)
        best_area = 0
        best_rect = (0, 0, 0, 0)

        for r in range(n_rows):
            height = np.where(mask[r], height + 1, 0)
            h_list = height.tolist()
            h_list.append(0)  # سنتینل برای خالی کردن استک در انتهای هر سطر

            stack = [-1]  # ایندکس ستون‌ها؛ -1 = مرز مجازی چپ
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

    print("\nدر حال پیدا کردن ناحیه‌ی مشترک (crop) هر سه باند...")
    valid_mask = (sii_aligned != 0) & (ha_aligned != 0) & (oiii_aligned != 0)

    if not valid_mask.any():
        print("  ⚠️ هیچ ناحیه‌ی مشترکی بین سه باند پیدا نشد؛ crop انجام نمی‌شه.")
    else:
        print("  در حال جستجوی بزرگ‌ترین مستطیل کاملاً معتبر (ممکنه چند ثانیه طول بکشه)...")
        rmin, rmax, cmin, cmax = largest_valid_rectangle(valid_mask)
        crop_coverage = valid_mask[rmin:rmax+1, cmin:cmax+1].mean() * 100
        print(f"  ناحیه‌ی crop: سطرها [{rmin}:{rmax}], ستون‌ها [{cmin}:{cmax}]")
        print(f"  ابعاد بعد از crop: {rmax-rmin+1} x {cmax-cmin+1}")
        print(f"  درصد پیکسل‌های معتبر داخل این ناحیه: {crop_coverage:.1f}% (باید ۱۰۰٪ باشه)")

        if crop_coverage < 99.9:
            print("  ⚠️ هشدار: مستطیل پیدا‌شده کاملاً معتبر نیست! به bounding box ساده برمی‌گردیم.")
            rows_any = np.any(valid_mask, axis=1)
            cols_any = np.any(valid_mask, axis=0)
            rmin, rmax = np.where(rows_any)[0][[0, -1]]
            cmin, cmax = np.where(cols_any)[0][[0, -1]]

        sii_aligned = sii_aligned[rmin:rmax+1, cmin:cmax+1]
        ha_aligned = ha_aligned[rmin:rmax+1, cmin:cmax+1]
        oiii_aligned = oiii_aligned[rmin:rmax+1, cmin:cmax+1]



    # ---------------------------------------------------------
    # percentile stretch روی هر باند
    # ---------------------------------------------------------
    print("\nدر حال اعمال percentile stretch...")
    r_channel = percentile_stretch(sii_aligned)   # SII  -> قرمز
    g_channel = percentile_stretch(ha_aligned)    # Ha   -> سبز
    b_channel = percentile_stretch(oiii_aligned)  # OIII -> آبی

    rgb_image = np.dstack([r_channel, g_channel, b_channel])

    # ---------------------------------------------------------
    # ذخیره‌ی خروجی
    # ---------------------------------------------------------
    output_path = os.path.join(OUTPUT_DIR, "m16_hubble_palette_baseline.png")
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image, origin="lower")
    plt.axis("off")
    plt.title("M16 - Hubble Palette (Baseline, percentile-stretch)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ تصویر baseline ذخیره شد در: {output_path}")

    # ---------------------------------------------------------
    # نمایش تصویر
    # روی Windows 11، WSLg به‌صورت پیش‌فرض فعاله و plt.show() باید
    # یک پنجره‌ی گرافیکی مستقیماً باز کنه. اگه به هر دلیلی (مثلاً
    # عدم فعال بودن WSLg) پنجره باز نشد، به‌صورت خودکار فایل ذخیره‌شده
    # با Explorer ویندوز باز می‌شه (fallback).
    # ---------------------------------------------------------
    try:
        plt.show()
    except Exception as e:
        print(f"  ⚠️ نمایش گرافیکی مستقیم ممکن نشد ({e})")
    finally:
        plt.close()

    # fallback: باز کردن فایل با ویوئر پیش‌فرض ویندوز
    try:
        import subprocess
        windows_path = subprocess.check_output(
            ["wslpath", "-w", output_path], text=True
        ).strip()
        subprocess.Popen(["explorer.exe", windows_path])
    except Exception as e:
        print(f"  ⚠️ باز کردن خودکار با Explorer ممکن نشد ({e})")
        print(f"  می‌تونی خودت این مسیر رو در File Explorer باز کنی: {output_path}")

    npz_path = os.path.join(OUTPUT_DIR, "m16_bands_aligned.npz")
    np.savez(
        npz_path,
        sii=sii_aligned,
        ha=ha_aligned,
        oiii=oiii_aligned,
    )
    print(f"✅ آرایه‌ی خام باندهای هم‌تراز‌شده ذخیره شد در: {npz_path}")


if __name__ == "__main__":
    main()