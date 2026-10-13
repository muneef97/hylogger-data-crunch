"""05 - PCA + clustering of HyLogger intervals into mineral 'assemblages'.

Features: SWIR + TIR mineral proportions and spectral indices (wavelength positions/depths).
  * Low-confidence intervals (QC flag) excluded.
  * Spectral indices missing where the mineral is absent -> filled with the column median.
  * All features standardised (z-scores).
Clustering: KMeans, k chosen by silhouette score over k = 4..10 (all intervals scored;
when scores are within 0.005 of the best, the smallest k is chosen).
MRDH400 (geologist-refined results only, not comparable) is excluded when FITTING
the scaler/PCA/KMeans, then its intervals are assigned to the nearest cluster.
Outputs: outputs/05_intervals_with_clusters.csv, outputs/05_cluster_profiles.csv, outputs/05_pca_loadings.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = Path(__file__).resolve().parents[1] / "outputs"
d = pd.read_csv(OUT / "03_merged_intervals.csv", low_memory=False)
d = d[~d["QC_low_confidence"]].reset_index(drop=True)

minerals = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c]
minerals = [c for c in minerals if d[c].mean() > 0.005]          # keep minerals with meaningful presence
indices = [c for c in ["2200W_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "8625D_Quartz", "VIS-SWIR_Albedo"] if c in d]
X = d[minerals + indices].copy()
X[indices] = X[indices].fillna(X[indices].median())
FIT_EXCLUDE = ["MRDH400"]
fit_mask = ~d["Hole"].isin(FIT_EXCLUDE).values
scaler = StandardScaler().fit(X[fit_mask])
Xs = scaler.transform(X)
Xf = Xs[fit_mask]

# ---- PCA ----
pca = PCA(n_components=6, random_state=0).fit(Xf)
pcs = pca.transform(Xs)
for i in range(3):
    d[f"PC{i+1}"] = pcs[:, i]
load = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"]).round(2)
load.to_csv(OUT / "05_pca_loadings.csv")
print("Features used:", len(X.columns))
print("PCA explained variance (PC1..PC6):", pca.explained_variance_ratio_.round(3).tolist(),
      f"-> cumulative {pca.explained_variance_ratio_.sum():.2f}")
for pc in ["PC1", "PC2"]:
    s = load[pc].sort_values()
    print(f"{pc}: negative = {list(s.index[:3])} | positive = {list(s.index[-3:])}")

# ---- choose k ----
scores = {}
for k in range(4, 11):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xf)
    scores[k] = silhouette_score(Xf, lab)   # all intervals (no random sampling) -> stable choice of k
print("\nSilhouette by k:", {k: round(v, 3) for k, v in scores.items()})
# Practically-tied scores -> prefer the simpler model: smallest k within TOL of the best score.
TOL = 0.005
best_k = min(k for k, v in scores.items() if v >= max(scores.values()) - TOL)
km = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit(Xf)
d["Cluster"] = km.predict(Xs)
# MRDH400 has no TIR data (TIR columns are 0, not 'absent'), so assign it to the nearest
# cluster centre using SWIR-derived features only.
swir_feats = [i for i, c in enumerate(X.columns) if c.startswith("SWIR_") or c in
              ["2200W_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "VIS-SWIR_Albedo"]]
nf = ~fit_mask
dist = ((Xs[nf][:, None, swir_feats] - km.cluster_centers_[None, :, swir_feats]) ** 2).sum(axis=2)
d["Cluster"] = d["Cluster"].astype("int64")
d.loc[nf, "Cluster"] = dist.argmin(axis=1).astype("int64")
print("MRDH400 assigned with SWIR-only features:", d.loc[nf, "Cluster"].value_counts().to_dict())
print("Chosen k =", best_k)

# ---- describe clusters ----
def top_minerals(row, n=3):
    r = row[minerals].sort_values(ascending=False)
    return ", ".join(f"{c} {v:.2f}" for c, v in r.head(n).items() if v > 0.03)

prof = d.groupby("Cluster")[minerals + indices].mean()
summary = pd.DataFrame({
    "n_intervals": d.groupby("Cluster").size(),
    "top_minerals": prof.apply(top_minerals, axis=1),
    "2200W_AlOH": prof["2200W_AlOH"].round(1),
    "median_Au_ppm": d.groupby("Cluster")["Au_PPM"].median().round(3),
    "pct_Au_ge_0.5": d.groupby("Cluster")["Au_PPM"].apply(lambda s: (s.dropna() >= 0.5).mean() * 100).round(1),   # assayed metres only
    "n_holes": d.groupby("Cluster")["Hole"].nunique(),
    "main_hole": d.groupby("Cluster")["Hole"].agg(lambda s: f"{s.value_counts().index[0]} ({s.value_counts(normalize=True).iloc[0]:.0%})"),
})
prof.round(3).join(summary[["n_intervals", "median_Au_ppm", "pct_Au_ge_0.5"]]).to_csv(OUT / "05_cluster_profiles.csv")
d.to_csv(OUT / "05_intervals_with_clusters.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print("\nCluster summary:\n", summary.to_string())
print("\nCluster share within each hole (%):\n", (pd.crosstab(d["Hole"], d["Cluster"], normalize="index") * 100).round(0).astype(int).to_string())
