"""
انتخاب قطعه‌های نماینده (patches) برای نظرسنجی ادراکی
---------------------------------------------------------
از تصویر نهایی، به تعداد N_PATCHES قطعه انتخاب می‌کنیم که سطوح
مختلف عدم‌قطعیت (از کم تا زیاد) رو پوشش بدن. هر قطعه به‌صورت
PNG کوچیک (base64) ذخیره می‌شه، همراه با «امتیاز عدم‌قطعیت واقعی»
که مخفی می‌مونه و بعداً برای همبستگی با نظر انسان‌ها استفاده می‌شه.

خروجی: یک فایل survey_patches.json که شامل تصویر base64 و
metadata هر قطعه‌ست (بدون افشای امتیاز عدم‌قطعیت به کاربر نهایی؛
اون فقط برای تحلیل بعدی نگه داشته می‌شه).

اجرا:
    conda activate astro
    python3 generate_survey_patches.py
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image

RESULT_PATH = os.path.expanduser("~/astro_data/M16/output/m16_uncertainty_result.npz")
OUTPUT_DIR = os.path.expanduser("~/astro_data/M16/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_SIZE = 180        # اندازه‌ی قطعه‌ی نمایش‌داده‌شده به کاربر (پیکسل)
N_PATCHES = 16           # تعداد کل قطعه‌ها
MARGIN = 20              # حاشیه‌ای که از لبه‌های تصویر رد می‌شیم (برای جلوگیری از artifact لبه)


def array_to_base64_png(arr):
    """آرایه‌ی float [0,1] با شکل (H,W,3) رو به base64 PNG تبدیل می‌کنه."""
    img_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode="RGB")
    # چون origin تصویر اصلی lower بود (مبدأ پایین)، برای نمایش صحیح در HTML/PNG استاندارد باید flip بشه
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    print("در حال بارگذاری نتایج MC-Dropout...")
    data = np.load(RESULT_PATH)
    mean_rgb = data["mean_rgb"]          # (3, H, W)
    uncertainty = data["uncertainty"]    # (3, H, W)

    mean_rgb_hwc = np.transpose(mean_rgb, (1, 2, 0))  # (H, W, 3)
    uncertainty_gray = uncertainty.mean(axis=0)        # (H, W)

    H, W = uncertainty_gray.shape
    ps = PATCH_SIZE

    # میانگین عدم‌قطعیت هر پنجره‌ی ممکن رو با یک کانولوشن ساده (میانگین متحرک) حساب می‌کنیم
    # تا بتونیم بهترین موقعیت‌ها رو برای سطوح مختلف عدم‌قطعیت پیدا کنیم
    valid_r = range(MARGIN, H - ps - MARGIN, 20)
    valid_c = range(MARGIN, W - ps - MARGIN, 20)

    print("در حال محاسبه‌ی عدم‌قطعیت میانگین برای موقعیت‌های کاندید...")
    candidates = []
    for r in valid_r:
        for c in valid_c:
            patch_unc = uncertainty_gray[r:r + ps, c:c + ps].mean()
            candidates.append((patch_unc, r, c))

    candidates.sort(key=lambda x: x[0])  # مرتب‌سازی از کم‌ترین تا بیشترین عدم‌قطعیت
    n_cand = len(candidates)
    print(f"تعداد موقعیت‌های کاندید: {n_cand}")

    # انتخاب N_PATCHES موقعیت پخش‌شده به‌طور یکنواخت روی طیف عدم‌قطعیت (کم تا زیاد)
    indices = np.linspace(0, n_cand - 1, N_PATCHES).astype(int)
    selected = [candidates[i] for i in indices]

    # حذف موقعیت‌های خیلی نزدیک به هم (برای تنوع بصری بیشتر)
    final_selected = []
    min_dist = ps  # حداقل فاصله بین مرکز پچ‌ها
    for unc, r, c in selected:
        too_close = any(
            abs(r - fr) < min_dist and abs(c - fc) < min_dist
            for _, fr, fc in final_selected
        )
        if not too_close:
            final_selected.append((unc, r, c))

    print(f"تعداد قطعه‌های نهایی انتخاب‌شده: {len(final_selected)}")

    # ---------------------------------------------------------
    # ساخت خروجی JSON (تصویر base64 + امتیاز واقعی، به‌صورت جدا)
    # ---------------------------------------------------------
    patches_for_survey = []   # این بخش به کاربر نهایی (رأی‌دهنده) نشون داده می‌شه - بدون امتیاز
    ground_truth = {}         # این بخش فقط برای تحلیل بعدی نگه داشته می‌شه

    # ترتیب رو تصادفی می‌کنیم تا کاربر نتونه از روی ترتیب حدس بزنه
    rng = np.random.default_rng(42)
    order = rng.permutation(len(final_selected))

    for display_idx, orig_idx in enumerate(order):
        unc, r, c = final_selected[orig_idx]
        patch_rgb = mean_rgb_hwc[r:r + ps, c:c + ps]
        b64 = array_to_base64_png(patch_rgb)
        patch_id = f"patch_{display_idx:02d}"

        patches_for_survey.append({
            "id": patch_id,
            "image_base64": b64,
        })
        ground_truth[patch_id] = {
            "mean_uncertainty": float(unc),
            "row": int(r),
            "col": int(c),
        }

    survey_path = os.path.join(OUTPUT_DIR, "survey_patches.json")
    with open(survey_path, "w", encoding="utf-8") as f:
        json.dump(patches_for_survey, f)

    gt_path = os.path.join(OUTPUT_DIR, "survey_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\n✅ فایل نظرسنجی (بدون امتیاز، برای رأی‌دهنده‌ها) ذخیره شد در:\n   {survey_path}")
    print(f"✅ فایل ground truth (فقط برای تحلیل خودت) ذخیره شد در:\n   {gt_path}")
    print(f"\nحجم فایل نظرسنجی: {os.path.getsize(survey_path) / 1024:.1f} KB")
    print("\nاین فایل survey_patches.json رو آپلود کن تا نظرسنجی تعاملی HTML رو برات بسازم.")


if __name__ == "__main__":
    main()
