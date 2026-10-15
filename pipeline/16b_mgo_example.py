"""16b - Q5 in one picture: MgO predicted from HyLogger minerals vs the lab assay, for a company the model never saw.

Same model and features as script 16 (PLS, number of components chosen inside the training data,
log10 MgO), held out by COMPANY with 10-fold GroupKFold. Left: every assayed metre, predicted vs
measured, with the 1:1 line. Right: one example hole down its depth. The example hole is chosen by a
fixed rule, not by its score: among holes with >= 100 assayed metres, the one whose MgO varies most
(ratio of its 95th to 5th percentile) - a hole that crosses several rock types, so the picture shows
whether the model follows big changes. Its own correlation is printed (it is not the best hole).
Plain-language numbers printed: R2 (same as 'skill' in 16), correlation, and the typical error
(median of |predicted - measured| / measured), plus the share of metres within a factor of 1.5.
Output: outputs/plots/16b_mgo_example.png, outputs/16b_mgo_predictions.csv
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import OUT

warnings.filterwarnings("ignore")
PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
T = "MgO_PCT"
d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
feats = pd.read_csv(OUT / "05_pca_loadings.csv", index_col=0).index.tolist()
company = pd.read_csv(OUT / "02_holes.csv").set_index("Hole")["Company"]
s = d[d[T].notna()].copy()
s = s[s.groupby("Hole")["Hole"].transform("size") >= 30]
half_min = s[T].where(s[T] > 0).groupby(s["Hole"]).transform("min") / 2          # as in script 16
s[T] = np.where(s[T] > 0, s[T], half_min.fillna(s[T][s[T] > 0].min() / 2))
s["Company_grp"] = s["Hole"].map(company).fillna("hole:" + s["Hole"])
X, y, grp = s[feats].to_numpy(float), np.log10(s[T].to_numpy(float)), s["Company_grp"].to_numpy()
pred, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
for tr, te in GroupKFold(n_splits=10).split(X, y, grp):
    m = GridSearchCV(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PLSRegression(scale=False)),
                     {"plsregression__n_components": [2, 4, 6, 8]}, cv=GroupKFold(3), scoring="neg_mean_squared_error", n_jobs=-1)
    m.fit(X[tr], y[tr], groups=grp[tr])
    pred[te] = np.ravel(m.predict(X[te])); base[te] = y[tr].mean()
s["MgO_predicted"] = 10 ** pred
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - base) ** 2)
corr = np.corrcoef(y, pred)[0, 1]
rel = np.abs(s["MgO_predicted"] - s[T]) / s[T]
ratio = np.maximum(s["MgO_predicted"] / s[T], s[T] / s["MgO_predicted"])
print(f"MgO, unseen company: {len(s):,} metres, {s.Hole.nunique()} holes | R2 {r2:.2f} | correlation (log) {corr:.2f} | "
      f"typical error {rel.median():.0%} of the lab value | within x1.5 of the lab value: {(ratio <= 1.5).mean():.0%}")
cnt = s["Hole"].value_counts()
spread = s[s["Hole"].isin(cnt[cnt >= 100].index)].groupby("Hole")[T].agg(lambda v: v.quantile(0.95) / v.quantile(0.05))
hole = spread.idxmax()
perhole = s.groupby("Hole").apply(lambda g: np.corrcoef(np.log10(g[T]), np.log10(g["MgO_predicted"]))[0, 1] if len(g) >= 30 else np.nan)
h = s[s["Hole"] == hole].sort_values("From_m")
hr = 1 - np.sum((np.log10(h[T]) - np.log10(h["MgO_predicted"])) ** 2) / np.sum((np.log10(h[T]) - np.log10(h[T]).mean()) ** 2)
print(f"Example hole (largest MgO spread, rank {int(perhole.rank(ascending=False)[hole])} of {perhole.notna().sum()} holes by correlation): {hole}, {len(h)} metres, correlation {np.corrcoef(np.log10(h[T]), np.log10(h['MgO_predicted']))[0, 1]:.2f}, "
      f"typical error {(np.abs(h['MgO_predicted'] - h[T]) / h[T]).median():.0%}")
s[["Hole", "From_m", "To_m", T, "MgO_predicted"]].to_csv(OUT / "16b_mgo_predictions.csv", index=False)

SURF = "#FDFCF9"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 7.2), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.3})
a1.scatter(s[T], s["MgO_predicted"], s=4, alpha=0.25, color="#2B6CB0", linewidths=0, rasterized=True)
lim = [0.05, 50]
a1.plot(lim, lim, color="#1F2A2E", lw=1.2, label="perfect prediction")
a1.fill_between(lim, [v / 1.5 for v in lim], [v * 1.5 for v in lim], color="#B5652B", alpha=0.12, label="within ×1.5 of the lab")
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel("MgO measured by the lab (%)"); a1.set_ylabel("MgO predicted from minerals (%)")
a1.set_title(f"All {len(s):,} assayed metres, {s.Hole.nunique()} holes", loc="left", fontsize=12)
a1.legend(frameon=False, fontsize=10, loc="upper left")
a2.plot(h[T], h["From_m"], color="#1F2A2E", lw=1.6, label="Lab assay")
a2.plot(h["MgO_predicted"], h["From_m"], color="#B5652B", lw=1.6, label="Predicted from minerals")
a2.invert_yaxis(); a2.set_xscale("log"); a2.set_xlabel("MgO (%)"); a2.set_ylabel("Depth (m)")
a2.set_title(f"Example hole {hole}: lab vs prediction down the hole", loc="left", fontsize=12)
a2.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.savefig(PLOTS / "16b_mgo_example.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("Saved plots/16b_mgo_example.png, 16b_mgo_predictions.csv")
