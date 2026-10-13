"""04 - First exploratory pass on the merged table.

  a) Dominant SWIR mineral per interval and per-hole mineral profile
  b) Spearman correlation of each mineral / spectral index with Au and pathfinders
  c) Mineral averages in gold-bearing (Au >= 0.5 ppm) vs background (< 0.05 ppm) intervals
Low-confidence intervals (QC flag) are excluded.
Outputs: outputs/04_hole_mineral_profile.csv, outputs/04_mineral_vs_gold.csv
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
d = pd.read_csv(OUT / "03_merged_intervals.csv", low_memory=False)
d = d[~d["QC_low_confidence"]]

swir = [c for c in d if c.startswith("SWIR_") and c != "SWIR_Invalid"]
tir = [c for c in d if c.startswith("TIR_") and c != "TIR_Invalid"]
indices = ["2200W_AlOH", "2200D_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "8625D_Quartz", "11300D_Carbonate"]
indices = [c for c in indices if c in d]

# a) dominant mineral
d["Dominant_SWIR"] = d[swir].idxmax(axis=1).str.replace("SWIR_", "")
print("Dominant SWIR mineral across all intervals:\n", d["Dominant_SWIR"].value_counts(normalize=True).round(3).head(8).to_string())
profile = d.groupby("Hole")[swir + tir].mean().round(3)
profile.to_csv(OUT / "04_hole_mineral_profile.csv")
print("\nMain SWIR minerals by hole (mean proportion):")
print(profile[["SWIR_White-Mica", "SWIR_Chlorite", "SWIR_Kaolin", "SWIR_Carbonate", "SWIR_Amphibole", "SWIR_Epidote"]].to_string())

# b) correlations with gold & pathfinders
feats = swir + tir + indices
targets = [c for c in ["Au_PPM", "As_PPM", "S_PPM", "Cu_PPM"] if c in d]
corr = d[feats + targets].corr(method="spearman").loc[feats, targets].round(2)

# c) gold vs background
gold = d[d["Au_PPM"] >= 0.5]; bg = d[d["Au_PPM"] < 0.05]
comp = pd.DataFrame({"mean_gold_intervals": gold[feats].mean(), "mean_background": bg[feats].mean()}).round(3)
comp["difference"] = (comp["mean_gold_intervals"] - comp["mean_background"]).round(3)
res = corr.join(comp).sort_values("Au_PPM")
res.to_csv(OUT / "04_mineral_vs_gold.csv")

print(f"\nGold intervals (Au>=0.5): {len(gold)} | background (Au<0.05): {len(bg)}")
print("\nStrongest relationships with Au (Spearman), with gold vs background means:")
show = res.reindex(res["Au_PPM"].abs().sort_values(ascending=False).index).head(12)
print(show[["Au_PPM", "As_PPM", "S_PPM", "mean_gold_intervals", "mean_background"]].to_string())
