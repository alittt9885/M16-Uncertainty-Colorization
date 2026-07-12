"""
آموزش مدل MC-Dropout برای رنگ‌آمیزی/بازسازی سه‌باندی M16
------------------------------------------------------------
ایده: چون «رنگ درست» واقعی نداریم، مدل رو طوری آموزش می‌دیم که:
    - ورودی: نسخه‌ی نویزی/ناقص باندها (گاهی یک باند کامل حذف می‌شه،
      گاهی نویز گاوسی اضافه می‌شه) — این شبیه‌سازی شرایط واقعی
      داده‌های نجومیه (میدان دید ناقص، نویز سنسور و غیره)
    - هدف (target): نسخه‌ی تمیز و کامل سه‌باندی

مدل با لایه‌های Dropout2d ساخته شده که حتی در inference هم فعال
می‌مونن (MC-Dropout) تا با چند forward pass بشه واریانس (=عدم‌قطعیت)
رو حساب کرد.

اجرا:
    conda activate astro
    python3 train_mc_dropout.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DATA_PATH = os.path.expanduser("~/astro_data/M16/output/m16_bands_aligned.npz")
CKPT_DIR = os.path.expanduser("~/astro_data/M16/checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

PATCH_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
N_PATCHES_PER_EPOCH = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def percentile_stretch(data, low=1, high=99.5):
    """کشیدگی درصدی برای نگاشت به بازه‌ی [0,1]."""
    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, [low, high])
    return np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1).astype(np.float32)


class PatchDataset(Dataset):
    """پچ‌های تصادفی از تصویر بزرگ، با نسخه‌ی نویزی/ناقص به‌عنوان ورودی
    و نسخه‌ی تمیز به‌عنوان هدف."""

    def __init__(self, sii, ha, oiii, patch_size=128, n_patches=4000):
        self.bands = np.stack([sii, ha, oiii], axis=0)  # (3, H, W)
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.H, self.W = sii.shape

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        ps = self.patch_size
        r = np.random.randint(0, self.H - ps)
        c = np.random.randint(0, self.W - ps)
        clean = self.bands[:, r:r + ps, c:c + ps].copy()

        x = clean.copy()
        # با احتمال ۴۰٪ یکی از باندها رو کامل حذف کن (شبیه‌سازی داده‌ی گمشده)
        if np.random.rand() < 0.4:
            ch = np.random.randint(0, 3)
            x[ch] = 0.0
        # نویز گاوسی خفیف روی همه‌ی باندها
        noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
        x = np.clip(x + noise, 0, 1)

        return torch.from_numpy(x), torch.from_numpy(clean)


class UncertainColorizer(nn.Module):
    """شبکه‌ی کوچک با Dropout2d برای بازسازی سه‌باندی + تخمین عدم‌قطعیت."""

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


def main():
    print(f"دستگاه استفاده‌شده: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nدر حال بارگذاری داده...")
    npz = np.load(DATA_PATH)
    sii = percentile_stretch(npz["sii"])
    ha = percentile_stretch(npz["ha"])
    oiii = percentile_stretch(npz["oiii"])
    print(f"ابعاد تصویر: {sii.shape}")

    dataset = PatchDataset(sii, ha, oiii, patch_size=PATCH_SIZE, n_patches=N_PATCHES_PER_EPOCH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = UncertainColorizer(p_drop=0.3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\nشروع آموزش...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch:2d}/{EPOCHS} - loss: {avg_loss:.5f}")

    ckpt_path = os.path.join(CKPT_DIR, "uncertain_colorizer.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ مدل آموزش‌دیده ذخیره شد در: {ckpt_path}")


if __name__ == "__main__":
    main()
