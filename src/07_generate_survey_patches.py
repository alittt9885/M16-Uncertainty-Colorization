"""
Select representative patches for the perceptual survey
---------------------------------------------------------
From the final image, select N_PATCHES patches that span different
levels of uncertainty (from low to high). Each patch is saved as a
small base64-encoded PNG, together with a "true uncertainty score"
that is kept hidden and used later to correlate with human ratings.

Output: a survey_patches.json file containing the base64 image and
metadata for each patch (without exposing the uncertainty score to
the end user; that is kept only for later analysis).

Run:
    conda activate astro
    python3 07_generate_survey_patches.py
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

PATCH_SIZE = 180        # size of the patch shown to the user (pixels)
N_PATCHES = 16           # total number of patches
MARGIN = 20              # margin to skip near the image edges (avoids edge artifacts)


def array_to_base64_png(arr):
    """Convert a float [0,1] array of shape (H,W,3) to a base64 PNG."""
    img_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode="RGB")
    # The original image has a "lower" origin, so it must be flipped for
    # correct display in standard HTML/PNG viewers.
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    print("Loading MC-Dropout results...")
    data = np.load(RESULT_PATH)
    mean_rgb = data["mean_rgb"]          # (3, H, W)
    uncertainty = data["uncertainty"]    # (3, H, W)

    mean_rgb_hwc = np.transpose(mean_rgb, (1, 2, 0))  # (H, W, 3)
    uncertainty_gray = uncertainty.mean(axis=0)        # (H, W)

    H, W = uncertainty_gray.shape
    ps = PATCH_SIZE

    # Compute the mean uncertainty of every candidate window with a
    # simple moving-average style scan, to find good positions across
    # the uncertainty spectrum.
    valid_r = range(MARGIN, H - ps - MARGIN, 20)
    valid_c = range(MARGIN, W - ps - MARGIN, 20)

    print("Computing mean uncertainty for candidate positions...")
    candidates = []
    for r in valid_r:
        for c in valid_c:
            patch_unc = uncertainty_gray[r:r + ps, c:c + ps].mean()
            candidates.append((patch_unc, r, c))

    candidates.sort(key=lambda x: x[0])  # sort from lowest to highest uncertainty
    n_cand = len(candidates)
    print(f"Number of candidate positions: {n_cand}")

    # Select N_PATCHES positions evenly spread across the uncertainty
    # range (low to high)
    indices = np.linspace(0, n_cand - 1, N_PATCHES).astype(int)
    selected = [candidates[i] for i in indices]

    # Drop positions that are too close to each other (for more visual diversity)
    final_selected = []
    min_dist = ps  # minimum distance between patch centers
    for unc, r, c in selected:
        too_close = any(
            abs(r - fr) < min_dist and abs(c - fc) < min_dist
            for _, fr, fc in final_selected
        )
        if not too_close:
            final_selected.append((unc, r, c))

    print(f"Final number of selected patches: {len(final_selected)}")

    # ---------------------------------------------------------
    # Build the JSON output (base64 image + true score, kept separate)
    # ---------------------------------------------------------
    patches_for_survey = []   # shown to the end user (rater) -- no score included
    ground_truth = {}         # kept only for later analysis

    # Randomize the order so the user can't guess anything from it
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

    print(f"\n\u2705 Survey file (no scores, for raters) saved to:\n   {survey_path}")
    print(f"\u2705 Ground-truth file (for your own analysis only) saved to:\n   {gt_path}")
    print(f"\nSurvey file size: {os.path.getsize(survey_path) / 1024:.1f} KB")
    print("\nUpload this survey_patches.json file to build the interactive HTML survey.")


if __name__ == "__main__":
    main()
