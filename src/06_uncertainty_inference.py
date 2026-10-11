"""
Build the uncertainty map with MC-Dropout over the full M16 image
---------------------------------------------------------------------
Run the trained model over the full image (not just small patches).
Because the image is large, it is split into overlapping tiles; each
tile is processed with N forward passes (with Dropout active), and all
tiles are finally stitched back together by averaging in the
overlapping regions.

Output:
    - mean_rgb: mean of N predictions -> final color image
    - uncertainty_map: std. dev. of N predictions -> uncertainty map

Run:
    conda activate astro
    python3 06_uncertainty_inference.py
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
STRIDE = 192          # overlap between tiles (TILE_SIZE - STRIDE = overlap amount)
N_PASSES = 30          # number of forward passes per tile (MC-Dropout)
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
    """Generate tile start coordinates so the whole image, including the
    edges, is covered."""
    rs = list(range(0, H - tile_size + 1, stride))
    if rs[-1] != H - tile_size:
        rs.append(H - tile_size)
    cs = list(range(0, W - tile_size + 1, stride))
    if cs[-1] != W - tile_size:
        cs.append(W - tile_size)
    return rs, cs


def main():
    print(f"Device in use: {DEVICE}")
    print("Loading data and model...")

    npz = np.load(DATA_PATH)
    sii = percentile_stretch(npz["sii"])
    ha = percentile_stretch(npz["ha"])
    oiii = percentile_stretch(npz["oiii"])
    bands = np.stack([sii, ha, oiii], axis=0)  # (3, H, W)
    H, W = sii.shape
    print(f"Full image size: {H} x {W}")

    model = UncertainColorizer(p_drop=0.3).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    # Key point for MC-Dropout: keep the model in train() mode so that
    # Dropout stays active even at inference time.
    model.train()

    # Accumulator buffers for weighted averaging in overlapping regions
    sum_mean = np.zeros((3, H, W), dtype=np.float32)
    sum_sq = np.zeros((3, H, W), dtype=np.float32)   # for overall variance
    count = np.zeros((1, H, W), dtype=np.float32)

    rs, cs = get_tile_coords(H, W, TILE_SIZE, STRIDE)
    total_tiles = len(rs) * len(cs)
    print(f"Total tiles: {total_tiles}  ({N_PASSES} forward passes per tile)")

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
                    print(f"  Tile {tile_idx}/{total_tiles} processed...")

    # Final averaging over the overlapping regions
    mean_rgb = sum_mean / count
    mean_sq = sum_sq / count
    # Var(X) = E[X^2] - (E[X])^2 (overall variance estimate across overlapping tiles)
    variance = np.clip(mean_sq - mean_rgb ** 2, 0, None)
    uncertainty_map = np.sqrt(variance)  # standard deviation = uncertainty

    # Convert to (H, W, 3) format for display
    mean_rgb_img = np.transpose(mean_rgb, (1, 2, 0))
    uncertainty_gray = uncertainty_map.mean(axis=0)  # average uncertainty across the 3 channels

    # ---------------------------------------------------------
    # Save and display the results
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    axes[0].imshow(np.clip(mean_rgb_img, 0, 1), origin="lower")
    axes[0].set_title("Mean model prediction (MC-Dropout mean)")
    axes[0].axis("off")

    im = axes[1].imshow(uncertainty_gray, origin="lower", cmap="inferno")
    axes[1].set_title("Uncertainty map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "m16_mc_dropout_uncertainty.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"\n\u2705 Final image (mean + uncertainty) saved to: {output_path}")

    # --- Added for the paper figure: also save each panel as its own
    # standalone file, since the RNAAS/preprint figure needs the example
    # composite and the uncertainty map as separate panel images. ---
    fig_a, ax_a = plt.subplots(figsize=(7, 7))
    ax_a.imshow(np.clip(mean_rgb_img, 0, 1), origin="lower")
    ax_a.axis("off")
    fig_a.savefig(os.path.join(OUTPUT_DIR, "example_patch.pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7, 7))
    im_b = ax_b.imshow(uncertainty_gray, origin="lower", cmap="inferno")
    ax_b.axis("off")
    fig_b.savefig(os.path.join(OUTPUT_DIR, "example_uncertainty_map.pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig_b)
    print(f"\u2705 Standalone panels saved: example_patch.pdf, example_uncertainty_map.pdf (in {OUTPUT_DIR})")

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
        print(f"  \u26a0\ufe0f Could not auto-open with Explorer ({e})")

    npz_out = os.path.join(OUTPUT_DIR, "m16_uncertainty_result.npz")
    np.savez(npz_out, mean_rgb=mean_rgb, uncertainty=uncertainty_map)
    print(f"\u2705 Raw array (mean + uncertainty) saved to: {npz_out}")


if __name__ == "__main__":
    main()
