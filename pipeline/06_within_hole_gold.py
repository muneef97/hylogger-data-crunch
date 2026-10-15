"""06 - Do the mineral-gold relationships hold WITHIN each hole?

Pooling all holes mixes 'this hole is rich' with 'this mineral is linked to gold'.
Here every comparison is made inside a hole, then summarised across holes.

  a) Per hole: Spearman correlation of each feature with Au; count how many holes agree in sign.
  b) Per hole: mean of feature in gold intervals (Au >= 0.5 ppm) minus background (Au < 0.05 ppm).
  c) Cluster 'lift': observed gold intervals in a cluster vs expected if each hole's
     gold rate applied evenly (lift > 1 = cluster is richer than its holes' average).
Only holes with >= 8 gold intervals and >= 20 background intervals are used in (a, b): with the
starter holes that is just 3 holes, so (a, b) are exploratory, not a general rule.
Lift uncertainty: 90% interval from a block bootstrap (10 m depth blocks resampled within each hole),
because neighbouring metres are not independent.
Features are chosen on ALL intervals (same basis as script 05), then tests use assayed metres only.
Outputs: outputs/06_within_hole_feature_effects.csv, outputs/06_cluster_gold_lift.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
d_all = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
feats = [c for c in d_all if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d_all[c].mean() > 0.005]
feats += [c for c in ["2200W_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "8625D_Quartz"] if c in d_all]
# Rare but geologically relevant minerals (tested even though they average < 0.5 %)
feats += [c for c in ["SWIR_Tourmaline"] if c in d_all and c not in feats]

d = d_all[d_all["Au_PPM"].notna()].copy()        # gold tests: assayed metres only
d["gold"] = d["Au_PPM"] >= 0.5
d["bg"] = d["Au_PPM"] < 0.05

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

# Per-hole detail for the headline features (a median of 3 holes hides a lot)
top = ["SWIR_Carbonate", "SWIR_White-Mica", "2200W_AlOH", "SWIR_Chlorite", "SWIR_Tourmaline"]
per_hole = pd.DataFrame({h: [d[d.Hole == h][f].corr(d[d.Hole == h]["Au_PPM"], method="spearman") for f in top]
                         for h in usable}, index=top).round(2)
print("\nSpearman rho with Au in each usable hole:\n", per_hole.to_string())

# c) cluster lift, controlling for hole
hole_rate = d.groupby("Hole")["gold"].mean()
d["expected_gold"] = d["Hole"].map(hole_rate)
lift = d.groupby("Cluster").agg(n=("gold", "size"), observed=("gold", "sum"), expected=("expected_gold", "sum"),
                                n_holes_with_gold=("Hole", lambda s: d.loc[s.index][d.loc[s.index, "gold"]]["Hole"].nunique()))
lift["lift"] = (lift["observed"] / lift["expected"]).round(2)
lift["expected"] = lift["expected"].round(1)
# block bootstrap: resample 10 m depth blocks within each hole, recompute lift
rng = np.random.default_rng(0)
d["block"] = d["Hole"] + "_" + (d["Mid_m"] // 10).astype(int).astype(str)
blocks = {h: g["block"].unique() for h, g in d.groupby("Hole")}
by_block = {b: g for b, g in d.groupby("block")}
boot = []
for _ in range(500):
    s_ = pd.concat([by_block[b] for h, bl in blocks.items() for b in rng.choice(bl, len(bl))])
    rate = s_.groupby("Hole")["gold"].mean()
    s_ = s_.assign(exp=s_["Hole"].map(rate))
    g_ = s_.groupby("Cluster").agg(o=("gold", "sum"), e=("exp", "sum"))
    boot.append(g_["o"] / g_["e"].replace(0, np.nan))
boot = pd.concat(boot, axis=1)
lift["lift_90ci_low"] = boot.quantile(0.05, axis=1).round(2)
lift["lift_90ci_high"] = boot.quantile(0.95, axis=1).round(2)
lift.to_csv(OUT / "06_cluster_gold_lift.csv")
print("\nCluster gold lift after accounting for each hole's own gold rate:\n", lift.sort_values("lift", ascending=False).to_string())
