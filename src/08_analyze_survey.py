import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

ground_truth = {
    "patch_00": 0.029136428609490395, "patch_01": 0.02991010621190071,
    "patch_02": 0.031623475253582, "patch_03": 0.02547457441687584,
    "patch_04": 0.019817806780338287, "patch_05": 0.03490735962986946,
    "patch_06": 0.0917506292462349, "patch_07": 0.033065784722566605,
    "patch_08": 0.027572568506002426, "patch_09": 0.02450764924287796,
    "patch_10": 0.026504794135689735, "patch_11": 0.03775689750909805,
    "patch_12": 0.02223319187760353, "patch_13": 0.03073294647037983,
}

raters = [
    {"rater_name": "الهه", "responses": {"patch_00":3,"patch_01":3,"patch_02":4,"patch_03":5,"patch_04":3,"patch_05":3,"patch_06":4,"patch_07":4,"patch_08":4,"patch_09":3,"patch_10":3,"patch_11":3,"patch_12":3,"patch_13":3}},
    {"rater_name": "ناصر تیموری", "responses": {"patch_00":2,"patch_01":2,"patch_02":2,"patch_03":2,"patch_04":3,"patch_05":2,"patch_06":4,"patch_08":4,"patch_09":2,"patch_10":4,"patch_11":4,"patch_12":4,"patch_13":4}},
    {"rater_name": "شایان", "responses": {"patch_00":2,"patch_01":2,"patch_02":3,"patch_03":4,"patch_04":2,"patch_05":2,"patch_06":1,"patch_07":4,"patch_08":4,"patch_09":2,"patch_10":2,"patch_11":2,"patch_12":3,"patch_13":2}},
    {"rater_name": "عرفان", "responses": {"patch_00":4,"patch_01":4,"patch_02":5,"patch_03":3,"patch_04":4,"patch_05":2,"patch_06":2,"patch_07":3,"patch_08":2,"patch_09":3,"patch_10":4,"patch_11":1,"patch_12":4,"patch_13":2}},
    {"rater_name": "ناشناس-۱", "responses": {"patch_00":2,"patch_01":3,"patch_02":2,"patch_03":4,"patch_04":3,"patch_05":1,"patch_06":4,"patch_07":5,"patch_08":4,"patch_09":3,"patch_10":2,"patch_11":3,"patch_12":4,"patch_13":5}},
    {"rater_name": "عباس", "responses": {"patch_00":3,"patch_01":3,"patch_02":4,"patch_03":4,"patch_04":3,"patch_05":2,"patch_06":3,"patch_07":3,"patch_08":4,"patch_09":3,"patch_10":3,"patch_11":3,"patch_12":3,"patch_13":3}},
    {"rater_name": "اصغر", "responses": {"patch_00":4,"patch_01":4,"patch_02":3,"patch_03":4,"patch_04":4,"patch_05":3,"patch_06":3,"patch_07":5,"patch_08":5,"patch_09":3,"patch_10":4,"patch_11":3,"patch_12":3,"patch_13":3}},
    {"rater_name": "اکبر", "responses": {"patch_00":4,"patch_01":4,"patch_02":4,"patch_03":5,"patch_04":4,"patch_05":3,"patch_06":3,"patch_07":5,"patch_08":4,"patch_09":3,"patch_10":4,"patch_11":3,"patch_12":4,"patch_13":2}},
    {"rater_name": "ناشناس-۲", "responses": {"patch_00":2,"patch_01":4,"patch_02":4,"patch_03":5,"patch_04":2,"patch_05":4,"patch_06":2,"patch_07":5,"patch_08":4,"patch_09":2,"patch_10":2,"patch_11":4,"patch_12":2,"patch_13":2}},
]

patch_ids = list(ground_truth.keys())
mean_scores, std_scores, n_votes = {}, {}, {}
for pid in patch_ids:
    vals = [r["responses"][pid] for r in raters if pid in r["responses"]]
    mean_scores[pid] = np.mean(vals)
    std_scores[pid] = np.std(vals)
    n_votes[pid] = len(vals)

print(f"{'قطعه':10s} {'عدم‌قطعیت مدل':>15s} {'میانگین امتیاز':>15s} {'انحراف‌معیار':>13s} {'n':>4s}")
print("-"*62)
for pid in sorted(patch_ids, key=lambda p: ground_truth[p]):
    print(f"{pid:10s} {ground_truth[pid]:15.4f} {mean_scores[pid]:15.2f} {std_scores[pid]:13.2f} {n_votes[pid]:4d}")

unc_vals = np.array([ground_truth[p] for p in patch_ids])
score_vals = np.array([mean_scores[p] for p in patch_ids])
std_vals = np.array([std_scores[p] for p in patch_ids])

rho, pval = stats.spearmanr(unc_vals, score_vals)
pearson_r, pearson_p = stats.pearsonr(unc_vals, score_vals)

# همبستگی عدم‌قطعیت مدل با میزان اختلاف‌نظر انسان‌ها (نه فقط میانگین امتیاز)
rho_disagree, pval_disagree = stats.spearmanr(unc_vals, std_vals)

print("\n" + "="*60)
print(f"تعداد رأی‌دهنده‌ها: {len(raters)}")
print(f"همبستگی اسپیرمن (میانگین امتیاز): rho={rho:.3f}, p={pval:.4f}")
print(f"همبستگی پیرسون (میانگین امتیاز):  r={pearson_r:.3f}, p={pearson_p:.4f}")
print(f"همبستگی عدم‌قطعیت مدل با پراکندگی نظرات: rho={rho_disagree:.3f}, p={pval_disagree:.4f}")
print("="*60)

sig = "✅ معنادار (p<0.05)" if pval < 0.05 else "⚠️ معنادار نیست"
sig2 = "✅ معنادار (p<0.05)" if pval_disagree < 0.05 else "⚠️ معنادار نیست"
print(f"نتیجه ۱ (امتیاز میانگین): {sig}")
print(f"نتیجه ۲ (پراکندگی نظرات): {sig2}")

# ---------------- نمودار پراکندگی برای مقاله ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].scatter(unc_vals, score_vals, s=90, color="#d98255", edgecolor="#1a1a2e", linewidth=1.2, zorder=3)
for pid in patch_ids:
    axes[0].annotate(pid.replace("patch_", "p"), (ground_truth[pid], mean_scores[pid]),
                      fontsize=8, xytext=(4,4), textcoords="offset points", color="#555")
z = np.polyfit(unc_vals, score_vals, 1)
xs = np.linspace(unc_vals.min(), unc_vals.max(), 50)
axes[0].plot(xs, np.polyval(z, xs), "--", color="#4fd1c5", linewidth=2, zorder=2)
axes[0].set_xlabel("عدم‌قطعیت مدل (میانگین std)")
axes[0].set_ylabel("میانگین امتیاز انسانی (۱-۵)")
axes[0].set_title(f"Spearman ρ={rho:.2f} (p={pval:.2f}), n={len(raters)} raters")
axes[0].grid(alpha=0.25)

axes[1].scatter(unc_vals, std_vals, s=90, color="#4fd1c5", edgecolor="#1a1a2e", linewidth=1.2, zorder=3)
for pid in patch_ids:
    axes[1].annotate(pid.replace("patch_", "p"), (ground_truth[pid], std_scores[pid]),
                      fontsize=8, xytext=(4,4), textcoords="offset points", color="#555")
z2 = np.polyfit(unc_vals, std_vals, 1)
axes[1].plot(xs, np.polyval(z2, xs), "--", color="#d98255", linewidth=2, zorder=2)
axes[1].set_xlabel("عدم‌قطعیت مدل (میانگین std)")
axes[1].set_ylabel("انحراف‌معیار امتیاز انسانی (اختلاف‌نظر)")
axes[1].set_title(f"Spearman ρ={rho_disagree:.2f} (p={pval_disagree:.2f})")
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("/home/claude/survey_correlation.png", dpi=180, bbox_inches="tight")
print("\nنمودار ذخیره شد: /home/claude/survey_correlation.png")

# ذخیره‌ی نتایج برای استفاده در آپدیت مقاله
results = {
    "n_raters": len(raters),
    "spearman_rho": float(rho), "spearman_p": float(pval),
    "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
    "spearman_disagreement_rho": float(rho_disagree), "spearman_disagreement_p": float(pval_disagree),
    "per_patch": {pid: {"uncertainty": ground_truth[pid], "mean_score": mean_scores[pid],
                          "std_score": std_scores[pid], "n": n_votes[pid]} for pid in patch_ids}
}
with open("/home/claude/survey_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
