"""
Train the MC-Dropout model for M16 three-band colorization/reconstruction
------------------------------------------------------------------------------
Idea: since we don't have a real "ground-truth color," we train the model
so that:
    - Input: a noisy/incomplete version of the bands (sometimes one whole
      band is dropped, sometimes Gaussian noise is added) -- this
      simulates real astronomical data conditions (incomplete field of
      view, sensor noise, etc.)
    - Target: the clean, complete three-band version

The model is built with Dropout2d layers that stay active even at
inference time (MC-Dropout), so that running several forward passes lets
us compute the variance (= uncertainty).

Run:
    conda activate astro
    python3 05_train_mc_dropout.py
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
    """Percentile stretch to map values into the [0,1] range."""
    finite = data[np.isfinite(data)]
    vmin, vmax = np.percentile(finite, [low, high])
    return np.clip((data - vmin) / (vmax - vmin + 1e-8), 0, 1).astype(np.float32)


class PatchDataset(Dataset):
    """Random patches from the full image, with a noisy/incomplete version
    as input and the clean version as the target."""

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
        # With probability 0.4, zero out one whole band (simulates missing data)
        if np.random.rand() < 0.4:
            ch = np.random.randint(0, 3)
            x[ch] = 0.0
        # Light Gaussian noise on all bands
        noise = np.random.normal(0, 0.05, size=x.shape).astype(np.float32)
        x = np.clip(x + noise, 0, 1)

        return torch.from_numpy(x), torch.from_numpy(clean)


class UncertainColorizer(nn.Module):
    """Small network with Dropout2d for three-band reconstruction + uncertainty estimation."""

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
    print(f"Device in use: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    npz = np.load(DATA_PATH)
    sii = percentile_stretch(npz["sii"])
    ha = percentile_stretch(npz["ha"])
    oiii = percentile_stretch(npz["oiii"])
    print(f"Image dimensions: {sii.shape}")

    dataset = PatchDataset(sii, ha, oiii, patch_size=PATCH_SIZE, n_patches=N_PATCHES_PER_EPOCH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = UncertainColorizer(p_drop=0.3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("\nStarting training...")
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
    print(f"\nTrained model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
