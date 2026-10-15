"""12 - Predictive models: can HyLogger mineralogy predict geochemistry? (GSWA idea 5, 'PLS modelling')

A) Pathfinder check (answers: are As, S, Sb, Cu really linked to gold in OUR holes?)
   Spearman correlation of each element with Au, WITHIN each hole that has enough gold intervals.
B) Predict each element (log10 ppm) from the 23 HyLogger features used in script 05.
   Models: PLS regression (as on GSWA's slide; number of components chosen inside the training holes)
           and Random Forest.
   Validation: LEAVE-ONE-HOLE-OUT - train on the other holes, predict the held-out one, repeat.
   Two scores: R2 (sklearn, vs the mean of the held-out values) and SKILL vs the real baseline
   = 1 - SSE(model) / SSE(predicting the TRAINING mean). Skill < 0 = worse than that baseline.
B2) Within-hole: 5 blocks of EQUAL DEPTH LENGTH; train on the other blocks minus a 5 m buffer next
   to the held-out block (neighbouring metres are similar), predict the held-out block.
C) Classify gold intervals (Au >= 0.5 ppm) - ROC AUC on held-out holes (0.5 = coin toss).
Feature importance: permutation importance measured on HELD-OUT depth blocks of the best within-hole
case (not on training rows). It shows what the model relied on, not a validated mineral-element law.
Features = exactly the 23 used in script 05; missing spectral indices are filled with the median of
the TRAINING rows inside every fold (no information from held-out data).
Low-confidence intervals are excluded (in 05); MRDH400 is removed in 02.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)
PATHFINDERS = ["As_PPM", "S_PPM", "Sb_PPM", "Cu_PPM"]
# MgO (%) = the major element in GSWA's own PLS example; it is assayed in the expanded hole set
ELEMENTS = ["Au_PPM"] + PATHFINDERS + ["MgO_PCT"]

d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
feats = pd.read_csv(OUT / "05_pca_loadings.csv", index_col=0).index.tolist()   # the exact script-05 features
BUFFER_M = 5
def skill(y, pred, base):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - base) ** 2)

# ---------- A) pathfinders vs gold, within holes ----------
rows = []
g = d[d["Au_PPM"].notna()]
holes_ok = g.groupby("Hole").filter(lambda x: (x.Au_PPM >= 0.5).sum() >= 8)["Hole"].unique()
for el in PATHFINDERS:
    rhos = []
    for h in holes_ok:
        x = g[(g.Hole == h)][[el, "Au_PPM"]].dropna()
        if len(x) >= 20 and x[el].nunique() > 3:
            rhos.append(spearmanr(x[el], x["Au_PPM"])[0])
    rhos = np.array(rhos)
    rows.append({"element": el.replace("_PPM", "").replace("_PCT", ""), "holes_tested": len(rhos), "median_rho_with_Au": round(np.median(rhos), 2) if len(rhos) else np.nan,
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
        base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PLSRegression())
        if k >= 2:
            self.m = GridSearchCV(base, {"plsregression__n_components": [1, 2, 3, 5, 8]},
                                  cv=GroupKFold(k), scoring="r2").fit(X, y, groups=groups)
        else:
            self.m = base.set_params(plsregression__n_components=2).fit(X, y)
        return self
    def predict(self, X): return self.m.predict(X)
def pls_model(): return PLSAuto()
def rf_model():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0))

scores, preds_store = [], {}
for el in ELEMENTS:
    s = d[d[el].notna() & (d[el] > 0)]
    if s["Hole"].nunique() < 4:
        print(f"  skipping {el}: assayed in only {s['Hole'].nunique()} holes"); continue
    X, y, grp = s[feats].to_numpy(), np.log10(s[el].to_numpy()), s["Hole"].to_numpy()
    for name, make in [("PLS", pls_model), ("RandomForest", rf_model)]:
        pred, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for tr, te in logo.split(X, y, grp):
            m = make()
            m.fit(X[tr], y[tr], **({"groups": grp[tr]} if name == "PLS" else {}))
            pred[te] = np.ravel(m.predict(X[te])); base[te] = y[tr].mean()
        per_hole = [r2_score(y[grp == h], pred[grp == h]) for h in np.unique(grp) if (grp == h).sum() >= 30]
        scores.append({"element": el.replace("_PPM", "").replace("_PCT", ""), "model": name, "n": len(y), "holes": len(np.unique(grp)),
                       "skill_vs_train_mean": round(skill(y, pred, base), 2),
                       "R2_held_out_holes": round(r2_score(y, pred), 2),
                       "spearman_held_out": round(spearmanr(y, pred)[0], 2),
                       "median_R2_within_hole": round(np.median(per_hole), 2)})
        preds_store[(el, name)] = (s, y, pred)
sc = pd.DataFrame(scores); sc.to_csv(OUT / "12_model_scores.csv", index=False)
print("\nB) Predicting log10(element) from HyLogger features - scored on HELD-OUT holes")
print("   skill_vs_train_mean: 0 = as good as predicting the training average, < 0 = worse")
print(sc.to_string(index=False))

# ---------- B2) WITHIN-hole prediction (GSWA's PLS use case) ----------
# Calibrate on part of a hole's assays, predict the rest of the SAME hole. Each hole's assayed depth
# range is cut into 5 blocks of equal LENGTH; training excludes the held-out block AND metres within
# BUFFER_M of it, so neighbouring (similar) metres cannot leak into the test.
within, store_w, blocks_w = [], {}, {}
for el in ELEMENTS:
    for h, s_h in d[d[el].notna() & (d[el] > 0)].groupby("Hole"):
        if len(s_h) < 100 or s_h[el].nunique() < 10: continue
        s_h = s_h.sort_values("Mid_m")
        X, y = s_h[feats].to_numpy(), np.log10(s_h[el].to_numpy())
        z = s_h["Mid_m"].to_numpy()
        edges = np.linspace(z.min(), z.max() + 1e-6, 6)
        block = np.digitize(z, edges[1:-1])
        for name in ["PLS", "RandomForest"]:
            pred, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for b in range(5):
                te = block == b
                if te.sum() == 0: continue
                lo, hi = edges[b] - BUFFER_M, edges[b + 1] + BUFFER_M
                tr = ~te & ((z < lo) | (z >= hi))
                m = (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PLSRegression(n_components=3))
                     if name == "PLS" else rf_model())
                m.fit(X[tr], y[tr]); pred[te] = np.ravel(m.predict(X[te])); base[te] = y[tr].mean()
            within.append({"element": el.replace("_PPM", "").replace("_PCT", ""), "hole": h, "model": name, "n": len(y),
                           "skill_vs_train_mean": round(skill(y, pred, base), 2),
                           "R2": round(r2_score(y, pred), 2), "spearman": round(spearmanr(y, pred)[0], 2)})
            store_w[(el, h, name)] = (s_h, y, pred); blocks_w[(el, h)] = (block, edges, z)
w = pd.DataFrame(within); w.to_csv(OUT / "12_within_hole_scores.csv", index=False)
summ = w.groupby(["element", "model"]).agg(holes=("hole", "size"), median_skill=("skill_vs_train_mean", "median"),
                                             median_R2=("R2", "median"),
                                             median_spearman=("spearman", "median"), best_R2=("R2", "max")).reset_index()
print(f"\nB2) WITHIN-hole prediction (5 equal-length depth blocks, {BUFFER_M} m buffer, predict each block):")
print(summ.to_string(index=False))
print("\nBest individual cases:")
print(w.sort_values("skill_vs_train_mean", ascending=False).head(6).to_string(index=False))

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
print("   Gold / non-gold assayed metres in those holes:",
      {h: f"{int(yb[grp == h].sum())}/{int((~yb[grp == h]).sum())}" for h in auc_holes})
print("   (AUC measures ranking only, not calibrated probabilities or a usable cut-off.)")

# ---------- feature importance: OUT-OF-BLOCK, for the best within-hole Random Forest case ----------
bi = w[w["model"] == "RandomForest"].sort_values("skill_vs_train_mean", ascending=False).iloc[0]
el_i, h_i = [e for e in ELEMENTS if e.startswith(bi["element"] + "_")][0], bi["hole"]
s_h, y_i, _ = store_w[(el_i, h_i, "RandomForest")]
block, edges, z = blocks_w[(el_i, h_i)]
X_i, imps = s_h[feats], []
for b in range(5):
    te = block == b
    if te.sum() < 10: continue
    tr = ~te & ((z < edges[b] - BUFFER_M) | (z >= edges[b + 1] + BUFFER_M))
    m = rf_model().fit(X_i[tr], y_i[tr])
    pi = permutation_importance(m, X_i[te], y_i[te], n_repeats=10, random_state=0, n_jobs=-1)
    imps.append(pi.importances_mean)
imp = pd.Series(np.mean(imps, axis=0), index=feats).sort_values(ascending=False)
imp.round(4).to_csv(OUT / "12_feature_importance.csv", header=["importance"])
print(f"\nFeature importance for {bi['element']} in {h_i} (Random Forest, measured on HELD-OUT depth blocks, "
      f"mean of {len(imps)} blocks; what the model relied on, not a validated law):")
print(imp.head(8).round(3).to_string())
best = sc.sort_values("skill_vs_train_mean", ascending=False).iloc[0]
el = [e for e in ELEMENTS if e.startswith(best["element"] + "_")][0]

# ---------- figures ----------
SURF, INK2, GREY = "#fcfcfb", "#52514e", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
s, y, pred = preds_store[(el, best["model"])]
fig, ax = plt.subplots(figsize=(6.2, 6))
ax.scatter(y, pred, s=6, alpha=0.35, color="#2a78d6", linewidths=0)
lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
ax.plot(lim, lim, color=INK2, lw=1, ls="--")
ax.set_xlabel(f"Measured log10 {best['element']} (lab assay; ppm, or % for MgO)")
ax.set_ylabel(f"Predicted log10 {best['element']} from HyLogger (hole never seen in training)")
ax.set_title(f"{best['element']} from minerals, unseen holes: skill = {best['skill_vs_train_mean']}, Spearman = {best['spearman_held_out']} ({best['model']})",
             loc="left", weight="bold", fontsize=10)
fig.savefig(PLOTS / "12a_pred_vs_measured.png", dpi=150, bbox_inches="tight"); plt.close(fig)

bw = w.sort_values("skill_vs_train_mean", ascending=False).iloc[0]
hh, yw, pw = store_w[([e for e in ELEMENTS if e.startswith(bw["element"] + "_")][0], bw["hole"], bw["model"])]
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(hh["Mid_m"], pw, color="#2a78d6", lw=1.2, label=f"Predicted from HyLogger ({bw['model']}, depth block + {BUFFER_M} m buffer held out)")
ax.scatter(hh["Mid_m"], yw, s=7, color="#e34948", zorder=3, label="Lab assay")
ax.set_xlabel("Depth (m)"); ax.set_ylabel(f"log10 {bw['element']}")
ax.set_title(f"{bw['hole']}: {bw['element']} predicted from minerals within the hole (skill = {bw['skill_vs_train_mean']}, Spearman = {bw['spearman']})",
             loc="left", weight="bold", fontsize=10)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
fig.savefig(PLOTS / "12b_downhole_prediction.png", dpi=150, bbox_inches="tight"); plt.close(fig)
hole = bw["hole"]
print("\nFigures: 12a_pred_vs_measured.png, 12b_downhole_prediction.png (hole:", hole, ")")
