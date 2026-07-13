# Uncertainty-Aware Colorization of Astronomical Images

Monte Carlo Dropout for pixel-wise uncertainty estimation in narrowband-to-RGB
colorization of Hubble Space Telescope imagery, validated against human
perceptual judgment.

**Author:** Ali Teymouri
**Target object:** M16 (Eagle Nebula / Pillars of Creation)
**Data source:** Hubble Space Telescope, via MAST (STScI)

---

## Overview

Traditional astronomical "Hubble Palette" colorization maps narrowband
filters (SII → red, Hα → green, OIII → blue) through a fixed, deterministic
stretch. This project instead trains a small convolutional network with
Monte Carlo Dropout (Gal & Ghahramani, 2016) to reconstruct clean three-band
imagery from noisy/incomplete input, then uses repeated stochastic forward
passes at inference time to produce, alongside the color image, a **per-pixel
uncertainty map**. Regions of missing data, cosmic-ray contamination, or
saturation are expected to show elevated uncertainty.

The resulting uncertainty map is validated against human judgment through a
small-scale perceptual survey in which raters score the "naturalness /
trustworthiness" of isolated image patches without seeing the model's
internal confidence.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── 01_download_oiii_sii.py      # MAST query + download (OIII, SII bands)
│   ├── 02_check_fits_headers.py     # Verify filter/instrument metadata
│   ├── 03_download_ha.py            # MAST query + download (Hα band)
│   ├── 04_baseline_colorization.py  # Reproject, crop, percentile-stretch baseline
│   ├── 05_train_mc_dropout.py       # MC-Dropout CNN training
│   ├── 06_uncertainty_inference.py  # Tiled MC-Dropout inference + uncertainty map
│   └── 07_generate_survey_patches.py# Patch selection for perceptual survey
├── survey/
│   └── uncertainty_survey.html      # Self-contained interactive perceptual survey
└── paper/
    ├── paper.tex                    # Manuscript source (LaTeX)
    └── paper.pdf                    # Compiled manuscript
```

## Pipeline

1. **Environment**: WSL2 + Ubuntu + Miniconda + PyTorch (CUDA) + astropy/astroquery/reproject
2. **Data acquisition**: query MAST for HST/WFC3 narrowband observations of M16
3. **Registration**: `reproject` all bands onto a common WCS grid; crop to the
   largest fully-valid rectangle (maximal-rectangle-in-binary-matrix algorithm)
4. **Baseline**: deterministic percentile-stretch Hubble Palette composite
5. **MC-Dropout model**: small CNN trained to reconstruct clean 3-band
   patches from inputs with simulated missing channels and Gaussian noise
6. **Inference**: tiled, overlapping-window inference with N=30 stochastic
   forward passes; mean = color estimate, std = uncertainty map
7. **Perceptual validation**: web-based survey correlating human "naturalness"
   ratings with model uncertainty (Spearman correlation)

## Requirements

See `requirements.txt`. GPU with CUDA support recommended (developed on an
NVIDIA RTX 4060 laptop GPU).

```bash
conda create -n astro python=3.10 -y
conda activate astro
pip install -r requirements.txt
```

## Status

- [x] Environment setup
- [x] Data acquisition
- [x] Baseline colorization
- [x] MC-Dropout model + training
- [x] Uncertainty map generation
- [ ] Perceptual survey — data collection in progress
- [ ] Novelty check (arXiv / ADS)
- [ ] Manuscript submission

## Citation

If you use this code, please cite the accompanying manuscript (see
`paper/paper.tex`; a formal citation will be added once a DOI/arXiv ID is
assigned).

## Acknowledgments

Based on observations made with the NASA/ESA Hubble Space Telescope,
obtained from the data archive at the Space Telescope Science Institute
(STScI). Data retrieved via the Mikulski Archive for Space Telescopes (MAST).

## License

MIT — see `LICENSE`.
