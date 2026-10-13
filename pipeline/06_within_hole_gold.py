"""06 - Do the mineral-gold relationships hold WITHIN each hole?

Pooling all holes mixes 'this hole is rich' with 'this mineral is linked to gold'.
Here every comparison is made inside a hole, then summarised across holes.

  a) Per hole: Spearman correlation of each feature with Au; count how many holes agree in sign.
  b) Per hole: mean of feature in gold intervals (Au >= 0.5 ppm) minus background (Au < 0.05 ppm).
  c) Cluster 'lift': observed gold intervals in a cluster vs expected if each hole's
     gold rate applied evenly (lift > 1 = cluster is richer than its holes' average).
Only holes with >= 8 gold intervals and >= 20 background intervals are used in (a, b).
Outputs: outputs/06_within_hole_feature_effects.csv, outputs/06_cluster_gold_lift.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
d = d[d["Au_PPM"].notna()].copy()
d["gold"] = d["Au_PPM"] >= 0.5
d["bg"] = d["Au_PPM"] < 0.05

feats = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].mean() > 0.005]
feats += [c for c in ["2200W_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "8625D_Quartz"] if c in d]
# Rare but geologically relevant minerals (tested even though they average < 0.5 %)
feats += [c for c in ["SWIR_Tourmaline"] if c in d and c not in feats]

counts = d.groupby("Hole").agg(n=("Au_PPM", "size"), gold=("gold", "sum"), bg=("bg", "sum"))
usable = counts[(counts["gold"] >= 8) & (counts["bg"] >= 20)].index.tolist()
print("Gold / background interval counts per hole:\n", counts.to_string())
print("\nHoles usable for within-hole tests:", usable)

def within_hole_effects(usable):
  rows = []
  for f in feats:
    rhos, diffs = [], []
    for h in usable:
        g = d[d["Hole"] == h]
        if g[f].nunique() > 2:
            rhos.append(g[f].corr(g["Au_PPM"], method="spearman"))
            diffs.append(g.loc[g["gold"], f].mean() - g.loc[g["bg"], f].mean())
    rhos = np.array(rhos); diffs = np.array(diffs)
    rows.append({"feature": f, "holes_tested": len(rhos),
                 "median_rho": np.nanmedian(rhos), "holes_positive": int((rhos > 0.1).sum()),
                 "holes_negative": int((rhos < -0.1).sum()), "median_gold_minus_bg": np.nanmedian(diffs)})
  return pd.DataFrame(rows).set_index("feature")

eff = within_hole_effects(usable)
eff["consistency"] = (eff[["holes_positive", "holes_negative"]].max(axis=1) / eff["holes_tested"]).round(2)
eff = eff.round(3).sort_values("median_rho")
eff.to_csv(OUT / "06_within_hole_feature_effects.csv")
print("\nWithin-hole relationships with Au (rho > 0.1 = positive, < -0.1 = negative):")
print(eff.reindex(eff["median_rho"].abs().sort_values(ascending=False).index).head(12).to_string())
print("\nTourmaline (rare mineral, checked explicitly):")
print(eff.loc[["SWIR_Tourmaline"]].to_string())
print("\nTourmaline present (>0) in gold vs background intervals, per usable hole:")
for h in usable:
    g = d[d["Hole"] == h]
    print(f"  {h:14s} gold {(g.loc[g.gold,'SWIR_Tourmaline']>0).mean():.0%}  background {(g.loc[g.bg,'SWIR_Tourmaline']>0).mean():.0%}")

# Sensitivity: does anything change without MRDH400 (geologist-refined results)?
no_m = within_hole_effects([h for h in usable if h != "MRDH400"])
top = ["SWIR_White-Mica", "2200W_AlOH", "SWIR_Carbonate", "SWIR_Chlorite", "SWIR_Tourmaline"]
cmp_ = pd.DataFrame({"median_rho_all": eff.loc[top, "median_rho"],
                     "median_rho_without_MRDH400": no_m.loc[top, "median_rho"].round(3),
                     "pos/neg_all": eff.loc[top, "holes_positive"].astype(str) + "/" + eff.loc[top, "holes_negative"].astype(str) + " of " + eff.loc[top, "holes_tested"].astype(str),
                     "pos/neg_without": no_m.loc[top, "holes_positive"].astype(str) + "/" + no_m.loc[top, "holes_negative"].astype(str) + " of " + no_m.loc[top, "holes_tested"].astype(str)})
print("\nSensitivity check - with vs without MRDH400:\n", cmp_.to_string())

# c) cluster lift, controlling for hole
hole_rate = d.groupby("Hole")["gold"].mean()
d["expected_gold"] = d["Hole"].map(hole_rate)
lift = d.groupby("Cluster").agg(n=("gold", "size"), observed=("gold", "sum"), expected=("expected_gold", "sum"),
                                n_holes_with_gold=("Hole", lambda s: d.loc[s.index][d.loc[s.index, "gold"]]["Hole"].nunique()))
lift["lift"] = (lift["observed"] / lift["expected"]).round(2)
lift["expected"] = lift["expected"].round(1)
lift.to_csv(OUT / "06_cluster_gold_lift.csv")
print("\nCluster gold lift after accounting for each hole's own gold rate:\n", lift.sort_values("lift", ascending=False).to_string())
