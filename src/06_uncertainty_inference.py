"""
ساخت نقشه‌ی عدم‌قطعیت (Uncertainty Map) با MC-Dropout روی کل تصویر M16
-----------------------------------------------------------------------
مدل آموزش‌دیده رو روی کل تصویر (نه فقط پچ‌های کوچیک) اجرا می‌کنیم.
چون تصویر بزرگه، به پچ‌های هم‌پوشان (overlapping tiles) تقسیم می‌شه،
هر پچ با N بار forward pass (با Dropout فعال) پردازش می‌شه، و در آخر
همه‌ی پچ‌ها با میانگین‌گیری در نواحی هم‌پوشان کنار هم چیده می‌شن.

خروجی:
    - mean_rgb: میانگین N پیش‌بینی -> تصویر رنگی نهایی
    - uncertainty_map: انحراف‌معیار N پیش‌بینی -> نقشه‌ی عدم‌قطعیت

اجرا:
    conda activate astro
    python3 mc_dropout_inference.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DATA_PATH = os.path.expanduser("~/astro_data/M16/output/m16_bands_aligned.npz")
CKPT_PATH = os.path.expanduser("~/astro_data/M16/checkpoints/uncertain_colorizer.pt")
OUTPUT_DIR = os.path.expanduser("~/astro_data/M16/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TILE_SIZE = 256
STRIDE = 192          # هم‌پوشانی بین پچ‌ها (TILE_SIZE - STRIDE = مقدار overlap)
N_PASSES = 30          # تعداد forward pass برای هر پچ (MC-Dropout)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def percentile_stretch(data, low=1, high=99.5):
    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, [low, high])
    return np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1).astype(np.float32)


class UncertainColorizer(nn.Module):
    def __init__(self, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(p_drop),
            nn.Conv2d(32, 3, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_tile_coords(H, W, tile_size, stride):
    """مختصات شروع پچ‌ها رو تولید می‌کنه، طوری که کل تصویر (از جمله لبه‌ها) پوشش داده بشه."""
    rs = list(range(0, H - tile_size + 1, stride))
    if rs[-1] != H - tile_size:
        rs.append(H - tile_size)
    cs = list(range(0, W - tile_size + 1, stride))
    if cs[-1] != W - tile_size:
        cs.append(W - tile_size)
    return rs, cs


def main():
    print(f"دستگاه استفاده‌شده: {DEVICE}")
    print("در حال بارگذاری داده و مدل...")

    npz = np.load(DATA_PATH)
    sii = percentile_stretch(npz["sii"])
    ha = percentile_stretch(npz["ha"])
    oiii = percentile_stretch(npz["oiii"])
    bands = np.stack([sii, ha, oiii], axis=0)  # (3, H, W)
    H, W = sii.shape
    print(f"ابعاد تصویر کامل: {H} x {W}")

    model = UncertainColorizer(p_drop=0.3).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    # نکته‌ی کلیدی MC-Dropout: مدل رو در حالت train نگه می‌داریم
    # تا Dropout حتی در inference هم فعال بمونه.
    model.train()

    # بافرهای انباشت برای میانگین‌گیری وزن‌دار در نواحی هم‌پوشان
    sum_mean = np.zeros((3, H, W), dtype=np.float32)
    sum_sq = np.zeros((3, H, W), dtype=np.float32)   # برای محاسبه‌ی واریانس کلی
    count = np.zeros((1, H, W), dtype=np.float32)

    rs, cs = get_tile_coords(H, W, TILE_SIZE, STRIDE)
    total_tiles = len(rs) * len(cs)
    print(f"تعداد کل پچ‌ها: {total_tiles}  (هر پچ {N_PASSES} بار forward pass می‌شه)")

    tile_idx = 0
    with torch.no_grad():
        for r in rs:
            for c in cs:
                tile_idx += 1
                patch = bands[:, r:r + TILE_SIZE, c:c + TILE_SIZE]
                x = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)  # (1,3,H,W)

                preds = torch.stack([model(x)[0] for _ in range(N_PASSES)])  # (N,3,H,W)
                mean_pred = preds.mean(dim=0).cpu().numpy()
                sq_pred = (preds ** 2).mean(dim=0).cpu().numpy()

                sum_mean[:, r:r + TILE_SIZE, c:c + TILE_SIZE] += mean_pred
                sum_sq[:, r:r + TILE_SIZE, c:c + TILE_SIZE] += sq_pred
                count[:, r:r + TILE_SIZE, c:c + TILE_SIZE] += 1

                if tile_idx % 10 == 0 or tile_idx == total_tiles:
                    print(f"  پچ {tile_idx}/{total_tiles} پردازش شد...")

    # میانگین‌گیری نهایی روی نواحی هم‌پوشان
    mean_rgb = sum_mean / count
    mean_sq = sum_sq / count
    # Var(X) = E[X^2] - (E[X])^2  (تخمین واریانس کلی بین پچ‌های هم‌پوشان)
    variance = np.clip(mean_sq - mean_rgb ** 2, 0, None)
    uncertainty_map = np.sqrt(variance)  # انحراف معیار = عدم‌قطعیت

    # تبدیل به فرمت (H, W, 3) برای نمایش
    mean_rgb_img = np.transpose(mean_rgb, (1, 2, 0))
    uncertainty_gray = uncertainty_map.mean(axis=0)  # میانگین عدم‌قطعیت روی ۳ کانال

    # ---------------------------------------------------------
    # ذخیره و نمایش نتایج
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(np.clip(mean_rgb_img, 0, 1), origin="lower")
    axes[0].set_title("میانگین پیش‌بینی مدل (MC-Dropout mean)")
    axes[0].axis("off")

    im = axes[1].imshow(uncertainty_gray, origin="lower", cmap="inferno")
    axes[1].set_title("نقشه‌ی عدم‌قطعیت (Uncertainty Map)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "m16_mc_dropout_uncertainty.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ تصویر نهایی (میانگین + عدم‌قطعیت) ذخیره شد در: {output_path}")

    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close()

    try:
        import subprocess
        windows_path = subprocess.check_output(
            ["wslpath", "-w", output_path], text=True
        ).strip()
        subprocess.Popen(["explorer.exe", windows_path])
    except Exception as e:
        print(f"  ⚠️ باز کردن خودکار با Explorer ممکن نشد ({e})")

    npz_out = os.path.join(OUTPUT_DIR, "m16_uncertainty_result.npz")
    np.savez(npz_out, mean_rgb=mean_rgb, uncertainty=uncertainty_map)
    print(f"✅ آرایه‌ی خام (mean + uncertainty) ذخیره شد در: {npz_out}")


if __name__ == "__main__":
    main()
