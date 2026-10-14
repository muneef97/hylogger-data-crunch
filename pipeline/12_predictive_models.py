"""12 - Predictive models: can HyLogger mineralogy predict geochemistry? (GSWA idea 5, 'PLS modelling')

A) Pathfinder check (answers: are As, S, Sb, Cu really linked to gold in OUR holes?)
   Spearman correlation of each element with Au, WITHIN each hole that has enough gold intervals.
B) Predict each element (log10 ppm) from the 23 HyLogger features used in script 05.
   Models: PLS regression (as on GSWA's slide; number of components chosen inside the training holes)
           and Random Forest.
   Validation: LEAVE-ONE-HOLE-OUT - train on 10 holes, predict the 11th, repeat. Scores are therefore
   on holes the model never saw. Baseline = always predicting the training average (R2 = 0).
C) Classify gold intervals (Au >= 0.5 ppm) - ROC AUC on held-out holes (0.5 = coin toss).
MRDH400 is excluded (no TIR data, different method); low-confidence intervals excluded.
Outputs: outputs/12_pathfinders_vs_gold.csv, outputs/12_model_scores.csv, outputs/12_feature_importance.csv,
         outputs/plots/12a_pred_vs_measured.png, outputs/plots/12b_downhole_prediction.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)
ELEMENTS = ["Au_PPM", "As_PPM", "S_PPM", "Sb_PPM", "Cu_PPM"]

d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
d = d[(d["Hole"] != "MRDH400") & ~d["QC_low_confidence"]].copy()
minerals = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].mean() > 0.005]
indices = [c for c in ["2200W_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "8625D_Quartz", "VIS-SWIR_Albedo"] if c in d]
feats = minerals + indices
d[indices] = d[indices].fillna(d[indices].median())

# ---------- A) pathfinders vs gold, within holes ----------
rows = []
g = d[d["Au_PPM"].notna()]
holes_ok = g.groupby("Hole").filter(lambda x: (x.Au_PPM >= 0.5).sum() >= 8)["Hole"].unique()
for el in ELEMENTS[1:]:
    rhos = []
    for h in holes_ok:
        x = g[(g.Hole == h)][[el, "Au_PPM"]].dropna()
        if len(x) >= 20 and x[el].nunique() > 3:
            rhos.append(spearmanr(x[el], x["Au_PPM"])[0])
    rhos = np.array(rhos)
    rows.append({"element": el.replace("_PPM", ""), "holes_tested": len(rhos), "median_rho_with_Au": round(np.median(rhos), 2) if len(rhos) else np.nan,
                 "holes_positive(>0.1)": int((rhos > 0.1).sum()), "holes_negative(<-0.1)": int((rhos < -0.1).sum())})
pf = pd.DataFrame(rows); pf.to_csv(OUT / "12_pathfinders_vs_gold.csv", index=False)
print("A) Pathfinder elements vs gold, WITHIN holes (holes with >= 8 gold intervals):")
print(pf.to_string(index=False))

# ---------- B) regression, leave-one-hole-out ----------
logo = LeaveOneGroupOut()
class PLSAuto:
    """PLS whose number of components is chosen by group CV INSIDE the training holes only."""
    def fit(self, X, y, groups):
        k = min(4, len(np.unique(groups)))
        base = make_pipeline(StandardScaler(), PLSRegression())
        if k >= 2:
            self.m = GridSearchCV(base, {"plsregression__n_components": [1, 2, 3, 5, 8]},
                                  cv=GroupKFold(k), scoring="r2").fit(X, y, groups=groups)
        else:
            self.m = base.set_params(plsregression__n_components=2).fit(X, y)
        return self
    def predict(self, X): return self.m.predict(X)
def pls_model(): return PLSAuto()
def rf_model():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)

scores, preds_store = [], {}
for el in ELEMENTS:
    s = d[d[el].notna() & (d[el] > 0)]
    if s["Hole"].nunique() < 4:
        print(f"  skipping {el}: assayed in only {s['Hole'].nunique()} holes"); continue
    X, y, grp = s[feats].to_numpy(), np.log10(s[el].to_numpy()), s["Hole"].to_numpy()
    for name, make in [("PLS", pls_model), ("RandomForest", rf_model)]:
        pred = np.full(len(y), np.nan)
        for tr, te in logo.split(X, y, grp):
            m = make()
            m.fit(X[tr], y[tr], **({"groups": grp[tr]} if name == "PLS" else {}))
            pred[te] = np.ravel(m.predict(X[te]))
        per_hole = [r2_score(y[grp == h], pred[grp == h]) for h in np.unique(grp) if (grp == h).sum() >= 30]
        scores.append({"element": el.replace("_PPM", ""), "model": name, "n": len(y), "holes": len(np.unique(grp)),
                       "R2_held_out_holes": round(r2_score(y, pred), 2),
                       "spearman_held_out": round(spearmanr(y, pred)[0], 2),
                       "median_R2_within_hole": round(np.median(per_hole), 2)})
        preds_store[(el, name)] = (s, y, pred)
sc = pd.DataFrame(scores); sc.to_csv(OUT / "12_model_scores.csv", index=False)
print("\nB) Predicting log10(element) from HyLogger features - scored on HELD-OUT holes (0 = no better than average):")
print(sc.to_string(index=False))

# ---------- B2) WITHIN-hole prediction (GSWA's PLS use case) ----------
# Calibrate on part of a hole's assays, predict the rest of the SAME hole. To avoid cheating with
# neighbouring metres, each hole is cut into 5 contiguous depth blocks; train on 4, predict the 5th.
within, store_w = [], {}
for el in ELEMENTS:
    for h, s_h in d[d[el].notna() & (d[el] > 0)].groupby("Hole"):
        if len(s_h) < 100 or s_h[el].nunique() < 10: continue
        s_h = s_h.sort_values("Mid_m")
        X, y = s_h[feats].to_numpy(), np.log10(s_h[el].to_numpy())
        block = np.minimum((np.arange(len(y)) * 5) // len(y), 4)
        for name in ["PLS", "RandomForest"]:
            pred = np.full(len(y), np.nan)
            for b in range(5):
                tr, te = block != b, block == b
                m = (make_pipeline(StandardScaler(), PLSRegression(n_components=3)) if name == "PLS" else rf_model())
                m.fit(X[tr], y[tr]); pred[te] = np.ravel(m.predict(X[te]))
            within.append({"element": el.replace("_PPM", ""), "hole": h, "model": name, "n": len(y),
                           "R2": round(r2_score(y, pred), 2), "spearman": round(spearmanr(y, pred)[0], 2)})
            store_w[(el, h, name)] = (s_h, y, pred)
w = pd.DataFrame(within); w.to_csv(OUT / "12_within_hole_scores.csv", index=False)
summ = w.groupby(["element", "model"]).agg(holes=("hole", "size"), median_R2=("R2", "median"),
                                             median_spearman=("spearman", "median"), best_R2=("R2", "max")).reset_index()
print("\nB2) WITHIN-hole prediction (5 depth blocks, train on 4, predict the 5th):")
print(summ.to_string(index=False))
print("\nBest individual cases:")
print(w.sort_values("R2", ascending=False).head(6).to_string(index=False))

# ---------- C) gold classification ----------
s = d[d["Au_PPM"].notna()]
X, yb, grp = s[feats].to_numpy(), (s["Au_PPM"] >= 0.5).to_numpy(), s["Hole"].to_numpy()
prob = np.full(len(yb), np.nan)
for tr, te in logo.split(X, yb, grp):
    if yb[tr].sum() == 0: continue
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)
    m.fit(X[tr], yb[tr]); prob[te] = m.predict_proba(X[te])[:, 1]
ok = ~np.isnan(prob)
auc_all = roc_auc_score(yb[ok], prob[ok])
auc_holes = {h: round(roc_auc_score(yb[(grp == h) & ok], prob[(grp == h) & ok]), 2)
             for h in np.unique(grp) if 8 <= yb[grp == h].sum() < (grp == h).sum()}
print(f"\nC) Gold interval (Au >= 0.5 ppm) classifier, held-out holes: ROC AUC = {auc_all:.2f} (0.5 = coin toss)")
print("   Within each held-out hole (holes with >= 8 gold intervals):", auc_holes)

# ---------- feature importance for the best-predicted element ----------
best = sc.sort_values("R2_held_out_holes", ascending=False).iloc[0]
el = best["element"] + "_PPM"
s = d[d[el].notna() & (d[el] > 0)]
rf = rf_model().fit(s[feats], np.log10(s[el]))
pi = permutation_importance(rf, s[feats], np.log10(s[el]), n_repeats=5, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
imp.round(4).to_csv(OUT / "12_feature_importance.csv", header=["importance"])
print(f"\nMost useful HyLogger features for predicting {best['element']} (permutation importance):")
print(imp.head(8).round(3).to_string())

# ---------- figures ----------
SURF, INK2, GREY = "#fcfcfb", "#52514e", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
s, y, pred = preds_store[(el, best["model"])]
fig, ax = plt.subplots(figsize=(6.2, 6))
ax.scatter(y, pred, s=6, alpha=0.35, color="#2a78d6", linewidths=0)
lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
ax.plot(lim, lim, color=INK2, lw=1, ls="--")
ax.set_xlabel(f"Measured log10 {best['element']} (ppm, lab assay)")
ax.set_ylabel(f"Predicted log10 {best['element']} from HyLogger (hole never seen in training)")
ax.set_title(f"{best['element']} from minerals: R² = {best['R2_held_out_holes']}, Spearman = {best['spearman_held_out']} ({best['model']})",
             loc="left", weight="bold", fontsize=10)
fig.savefig(PLOTS / "12a_pred_vs_measured.png", dpi=150, bbox_inches="tight"); plt.close(fig)

bw = w.sort_values("R2", ascending=False).iloc[0]
hh, yw, pw = store_w[(bw["element"] + "_PPM", bw["hole"], bw["model"])]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(hh["Mid_m"], pw, color="#2a78d6", lw=1.2, label=f"Predicted from HyLogger ({bw['model']}, depth block held out)")
ax.scatter(hh["Mid_m"], yw, s=7, color="#e34948", zorder=3, label="Lab assay")
ax.set_xlabel("Depth (m)"); ax.set_ylabel(f"log10 {bw['element']} (ppm)")
ax.set_title(f"{bw['hole']}: {bw['element']} predicted from minerals within the hole (R² = {bw['R2']}, Spearman = {bw['spearman']})",
             loc="left", weight="bold", fontsize=10)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.savefig(PLOTS / "12b_downhole_prediction.png", dpi=150, bbox_inches="tight"); plt.close(fig)
hole = bw["hole"]
print("\nFigures: 12a_pred_vs_measured.png, 12b_downhole_prediction.png (hole:", hole, ")")
