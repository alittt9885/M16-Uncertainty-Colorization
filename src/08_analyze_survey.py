"""
Analyze the perceptual survey results and produce the correlation figure
for the paper (Figure 1: naturalness vs. uncertainty, disagreement vs.
uncertainty).

Reads the per-patch model uncertainty (ground_truth, from
06_uncertainty_inference.py) and the raw rater responses (collected via
survey/uncertainty_survey.html), computes Spearman/Pearson correlations,
and saves a two-panel scatter plot.

NOTE: rater identities are anonymized here (Rater_1, Rater_2, ...) rather
than using real names, since this script -- and its output -- are public
(linked directly from the paper). Keep a SEPARATE, non-public mapping from
Rater_N back to real names/consent records if you need one for your own
records; never commit that mapping to the public repo.

Run:
    python 08_analyze_survey.py
"""
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Per-patch model uncertainty (mean MC-Dropout std over the patch),
# produced by 06_uncertainty_inference.py / 07_generate_survey_patches.py
ground_truth = {
    "patch_00": 0.029136428609490395, "patch_01": 0.02991010621190071,
    "patch_02": 0.031623475253582, "patch_03": 0.02547457441687584,
    "patch_04": 0.019817806780338287, "patch_05": 0.03490735962986946,
    "patch_06": 0.0917506292462349, "patch_07": 0.033065784722566605,
    "patch_08": 0.027572568506002426, "patch_09": 0.02450764924287796,
    "patch_10": 0.026504794135689735, "patch_11": 0.03775689750909805,
    "patch_12": 0.02223319187760353, "patch_13": 0.03073294647037983,
}

# Raw per-rater "naturalness" scores (1-5) from the browser survey.
# Anonymized: original file had real first/last names here.
raters = [
    {"rater_name": "Rater_1", "responses": {"patch_00":3,"patch_01":3,"patch_02":4,"patch_03":5,"patch_04":3,"patch_05":3,"patch_06":4,"patch_07":4,"patch_08":4,"patch_09":3,"patch_10":3,"patch_11":3,"patch_12":3,"patch_13":3}},
    {"rater_name": "Rater_2", "responses": {"patch_00":2,"patch_01":2,"patch_02":2,"patch_03":2,"patch_04":3,"patch_05":2,"patch_06":4,"patch_08":4,"patch_09":2,"patch_10":4,"patch_11":4,"patch_12":4,"patch_13":4}},
    {"rater_name": "Rater_3", "responses": {"patch_00":2,"patch_01":2,"patch_02":3,"patch_03":4,"patch_04":2,"patch_05":2,"patch_06":1,"patch_07":4,"patch_08":4,"patch_09":2,"patch_10":2,"patch_11":2,"patch_12":3,"patch_13":2}},
    {"rater_name": "Rater_4", "responses": {"patch_00":4,"patch_01":4,"patch_02":5,"patch_03":3,"patch_04":4,"patch_05":2,"patch_06":2,"patch_07":3,"patch_08":2,"patch_09":3,"patch_10":4,"patch_11":1,"patch_12":4,"patch_13":2}},
    {"rater_name": "Rater_5", "responses": {"patch_00":2,"patch_01":3,"patch_02":2,"patch_03":4,"patch_04":3,"patch_05":1,"patch_06":4,"patch_07":5,"patch_08":4,"patch_09":3,"patch_10":2,"patch_11":3,"patch_12":4,"patch_13":5}},
    {"rater_name": "Rater_6", "responses": {"patch_00":3,"patch_01":3,"patch_02":4,"patch_03":4,"patch_04":3,"patch_05":2,"patch_06":3,"patch_07":3,"patch_08":4,"patch_09":3,"patch_10":3,"patch_11":3,"patch_12":3,"patch_13":3}},
    {"rater_name": "Rater_7", "responses": {"patch_00":4,"patch_01":4,"patch_02":3,"patch_03":4,"patch_04":4,"patch_05":3,"patch_06":3,"patch_07":5,"patch_08":5,"patch_09":3,"patch_10":4,"patch_11":3,"patch_12":3,"patch_13":3}},
    {"rater_name": "Rater_8", "responses": {"patch_00":4,"patch_01":4,"patch_02":4,"patch_03":5,"patch_04":4,"patch_05":3,"patch_06":3,"patch_07":5,"patch_08":4,"patch_09":3,"patch_10":4,"patch_11":3,"patch_12":4,"patch_13":2}},
    {"rater_name": "Rater_9", "responses": {"patch_00":2,"patch_01":4,"patch_02":4,"patch_03":5,"patch_04":2,"patch_05":4,"patch_06":2,"patch_07":5,"patch_08":4,"patch_09":2,"patch_10":2,"patch_11":4,"patch_12":2,"patch_13":2}},
]
# NOTE: Rater_2's response dict is missing "patch_07" in the source data
# (13 entries instead of 14) -- this is a pre-existing gap in the raw
# data, not something I've filled in. Confirm this is expected (e.g. that
# rater skipped one patch) before treating per-patch n as always 9.

patch_ids = list(ground_truth.keys())
mean_scores, std_scores, n_votes = {}, {}, {}
for pid in patch_ids:
    vals = [r["responses"][pid] for r in raters if pid in r["responses"]]
    mean_scores[pid] = np.mean(vals)
    std_scores[pid] = np.std(vals)
    n_votes[pid] = len(vals)

print(f"{'patch':10s} {'model_uncertainty':>18s} {'mean_rating':>12s} {'std_rating':>11s} {'n':>4s}")
print("-" * 62)
for pid in sorted(patch_ids, key=lambda p: ground_truth[p]):
    print(f"{pid:10s} {ground_truth[pid]:18.4f} {mean_scores[pid]:12.2f} {std_scores[pid]:11.2f} {n_votes[pid]:4d}")

unc_vals = np.array([ground_truth[p] for p in patch_ids])
score_vals = np.array([mean_scores[p] for p in patch_ids])
std_vals = np.array([std_scores[p] for p in patch_ids])

rho, pval = stats.spearmanr(unc_vals, score_vals)
pearson_r, pearson_p = stats.pearsonr(unc_vals, score_vals)

# Correlation between model uncertainty and inter-rater disagreement
# (not just the mean rating)
rho_disagree, pval_disagree = stats.spearmanr(unc_vals, std_vals)

print("\n" + "=" * 60)
print(f"Number of raters: {len(raters)}")
print(f"Spearman correlation (mean rating):        rho={rho:.3f}, p={pval:.4f}")
print(f"Pearson correlation (mean rating):          r={pearson_r:.3f}, p={pearson_p:.4f}")
print(f"Spearman correlation (inter-rater disagreement): rho={rho_disagree:.3f}, p={pval_disagree:.4f}")
print("=" * 60)

sig = "significant (p<0.05)" if pval < 0.05 else "not significant"
sig2 = "significant (p<0.05)" if pval_disagree < 0.05 else "not significant"
print(f"Result 1 (mean rating):          {sig}")
print(f"Result 2 (inter-rater disagreement): {sig2}")

# ---------------- Scatter plots for the paper (English labels) ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].scatter(unc_vals, score_vals, s=90, color="#d98255", edgecolor="#1a1a2e", linewidth=1.2, zorder=3)
for pid in patch_ids:
    axes[0].annotate(pid.replace("patch_", "p"), (ground_truth[pid], mean_scores[pid]),
                      fontsize=8, xytext=(4, 4), textcoords="offset points", color="#555")
z = np.polyfit(unc_vals, score_vals, 1)
xs = np.linspace(unc_vals.min(), unc_vals.max(), 50)
axes[0].plot(xs, np.polyval(z, xs), "--", color="#4fd1c5", linewidth=2, zorder=2)
axes[0].set_xlabel("Model uncertainty (mean std)")
axes[0].set_ylabel("Mean human naturalness rating (1-5)")
axes[0].set_title(f"Spearman $\\rho$={rho:.2f} (p={pval:.2f}), n={len(raters)} raters")
axes[0].grid(alpha=0.25)

axes[1].scatter(unc_vals, std_vals, s=90, color="#4fd1c5", edgecolor="#1a1a2e", linewidth=1.2, zorder=3)
for pid in patch_ids:
    axes[1].annotate(pid.replace("patch_", "p"), (ground_truth[pid], std_scores[pid]),
                      fontsize=8, xytext=(4, 4), textcoords="offset points", color="#555")
z2 = np.polyfit(unc_vals, std_vals, 1)
axes[1].plot(xs, np.polyval(z2, xs), "--", color="#d98255", linewidth=2, zorder=2)
axes[1].set_xlabel("Model uncertainty (mean std)")
axes[1].set_ylabel("Std. dev. of human ratings (disagreement)")
axes[1].set_title(f"Spearman $\\rho$={rho_disagree:.2f} (p={pval_disagree:.2f})")
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("disagreement_vs_uncertainty_combined.pdf", bbox_inches="tight")
plt.savefig("disagreement_vs_uncertainty_combined.png", dpi=180, bbox_inches="tight")
print("\nFigure saved: disagreement_vs_uncertainty_combined.{pdf,png}")

# Save results for use when updating the paper
results = {
    "n_raters": len(raters),
    "spearman_rho": float(rho), "spearman_p": float(pval),
    "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
    "spearman_disagreement_rho": float(rho_disagree), "spearman_disagreement_p": float(pval_disagree),
    "per_patch": {pid: {"uncertainty": ground_truth[pid], "mean_score": mean_scores[pid],
                         "std_score": std_scores[pid], "n": n_votes[pid]} for pid in patch_ids}
}
with open("survey_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
