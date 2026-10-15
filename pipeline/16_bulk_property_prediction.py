"""16 - "Can we extract other properties in bulk?" (GSWA Data Crunch task 5, beyond MgO)

For every assayed element with enough data, predict it from the same 23 HyLogger features as script 05
(chosen in 05 from mineral averages only - never from assays, so no target leakage) and score it ONLY
on data the model never saw.

Targets: log10 of the assay. Blank = not assayed (left out). Negative values were already converted to
half the detection limit in script 03. Exact zeros (rare; counted and printed) are treated as below
detection: half of that element's smallest positive value in the same hole. Where an element is
reported both as an oxide (e.g. MgO_PCT) and as ppm (Mg_PPM), only the oxide is used. "Major element"
is chemistry (Al, Ti, Mg, ...), whatever unit the lab reported; the unit is shown separately.
Elements need >= 10 holes with >= 30 assayed metres.

Validation (two levels, both 10-fold GroupKFold):
  * by HOLE     - every hole held out once (= "a new hole");
  * by COMPANY  - all holes of a company held out together (stricter: closer to "a new project area";
                  holes with no company recorded count as their own group).
Models, BOTH reported for every element (no picking the better one per element - that would use the
test results to choose and inflate the scores):
  * PLS (GSWA's method): number of components chosen INSIDE the training folds (inner grouped CV);
  * gradient boosting (non-linear check, fixed settings, handles missing values itself).
Scores on held-out data (log space):
  skill            = 1 - SSE(model) / SSE(training mean)        (> 0 = better than guessing the average)
  per_hole_skill   = median of the same score computed hole by hole (each hole counts once)
  within_hole      = DIAGNOSTIC only: skill after removing each held-out hole's own average from both
                     measurements and predictions (needs that average, so not usable on an unassayed
                     hole); shows whether the model follows the ups and downs inside a hole.
  median_rel_error = median |10^pred - measured| / measured (back in lab units; rough, as 10^pred is a
                     median-type, not mean, estimate)
The 0.3 skill line is a reporting choice, not a scientific boundary.
Usage: python 16_bulk_property_prediction.py            (all elements)
       python 16_bulk_property_prediction.py MgO K2O    (quick test on some)
Outputs: outputs/16_bulk_prediction_scores.csv, outputs/plots/16_bulk_prediction.png
"""
import sys
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from config import OUT

PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
feats = pd.read_csv(OUT / "05_pca_loadings.csv", index_col=0).index.tolist()
company = pd.read_csv(OUT / "02_holes.csv").set_index("Hole")["Company"]
d["Company_grp"] = d["Hole"].map(company).fillna("hole:" + d["Hole"])

OXIDE_OF = {"Mg": "MgO", "Ca": "CaO", "K": "K2O", "Na": "Na2O", "Fe": "FeO", "Mn": "MnO", "P": "P2O5",
            "Al": "Al2O3", "Ti": "TiO2", "Si": "SiO2"}
MAJOR = set(OXIDE_OF) | set(OXIDE_OF.values())
cols = [c for c in d if c.endswith(("_PCT", "_PPM"))]
targets = []
for c in cols:
    el = c.rsplit("_", 1)[0]
    if c.endswith("_PPM") and el in OXIDE_OF and f"{OXIDE_OF[el]}_PCT" in cols:
        continue
    s = d[d[c].notna()]
    if (s.groupby("Hole").size() >= 30).sum() >= 10:
        targets.append(c)
if len(sys.argv) > 1:
    targets = [t for t in targets if t.rsplit("_", 1)[0] in sys.argv[1:]]
print(f"Elements tested: {len(targets)} ->", [t.rsplit('_', 1)[0] for t in targets])

def skill(y, p, b): return 1 - np.sum((y - p) ** 2) / np.sum((y - b) ** 2)
def make(name, groups):
    if name == "PLS":
        base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PLSRegression(scale=False))
        return GridSearchCV(base, {"plsregression__n_components": [2, 4, 6, 8]}, cv=GroupKFold(3),
                            scoring="neg_mean_squared_error")
    return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, min_samples_leaf=40, random_state=0)

rows, store, zeros = [], {}, {}
for t in targets:
    s = d[d[t].notna()].copy()
    s = s[s.groupby("Hole")["Hole"].transform("size") >= 30]
    zeros[t.rsplit("_", 1)[0]] = int((s[t] <= 0).sum())
    half_min = s[t].where(s[t] > 0).groupby(s["Hole"]).transform("min") / 2
    s[t] = np.where(s[t] > 0, s[t], half_min.fillna(s[t][s[t] > 0].min() / 2))
    X, y = s[feats].to_numpy(float), np.log10(s[t].to_numpy(float))
    hole = s["Hole"].to_numpy()
    for name in ["PLS", "Boosting"]:
        for level, grp in [("hole", hole), ("company", s["Company_grp"].to_numpy())]:
            pred, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for tr, te in GroupKFold(n_splits=10).split(X, y, grp):
                m = make(name, grp[tr])
                m.fit(X[tr], y[tr], **({"groups": grp[tr]} if name == "PLS" else {}))
                pred[te] = np.ravel(m.predict(X[te])); base[te] = y[tr].mean()
            yc = y - pd.Series(y).groupby(hole).transform("mean").to_numpy()
            pc = pred - pd.Series(pred).groupby(hole).transform("mean").to_numpy()
            ph = [skill(y[hole == h], pred[hole == h], base[hole == h]) for h in np.unique(hole)
                  if (hole == h).sum() >= 30 and np.std(y[hole == h]) > 0]
            el = t.rsplit("_", 1)[0]
            rows.append({"element": el, "unit": "%" if t.endswith("_PCT") else "ppm",
                         "type": "major element" if el in MAJOR else "trace element", "model": name,
                         "held_out": level, "holes": len(np.unique(hole)), "groups": len(np.unique(grp)),
                         "n": len(y), "skill": round(skill(y, pred, base), 2),
                         "per_hole_skill": round(float(np.median(ph)), 2),
                         "within_hole": round(1 - np.sum((yc - pc) ** 2) / np.sum(yc ** 2), 2),
                         "spearman": round(spearmanr(y, pred)[0], 2),
                         "median_rel_error": round(float(np.median(np.abs(10 ** pred - 10 ** y) / 10 ** y)), 2)})
            store[(el, name, level)] = (y, pred)
res = pd.DataFrame(rows)
res.to_csv(OUT / "16_bulk_prediction_scores.csv", index=False)
print("Exact zeros treated as below detection:", {k: v for k, v in zeros.items() if v} or "none")

pd.set_option("display.width", 220)
wide = res.pivot_table(index="element", columns=["model", "held_out"], values="skill")
wide.columns = [f"{m}_{h}" for m, h in wide.columns]
extra = res[(res.model == "PLS") & (res.held_out == "hole")].set_index("element")[
    ["type", "unit", "holes", "per_hole_skill", "within_hole", "median_rel_error"]].add_prefix("")
tab = extra.join(wide).sort_values("PLS_hole", ascending=False)
print("\nSkill on held-out data (> 0 = better than the training average). PLS columns = GSWA's method;")
print("per_hole_skill / within_hole / median_rel_error are for PLS, held out by hole:")
print(tab.to_string())
for name in ["PLS", "Boosting"]:
    for level in ["hole", "company"]:
        x = res[(res.model == name) & (res.held_out == level)]
        print(f"  {name:8s} held out by {level:7s}: skill >= 0.3 for {(x.skill >= 0.3).sum():2d} of {len(x)} elements "
              f"(median skill {x.skill.median():.2f})")
pls = tab
print(f"\nConsistently predictable (PLS skill >= 0.3 held out by hole AND by company, per-hole median > 0): "
      f"{list(pls[(pls.PLS_hole >= 0.3) & (pls.PLS_company >= 0.3) & (pls.per_hole_skill > 0)].index)}")
print(f"Not predictable (PLS skill < 0.1 by hole): {list(pls[pls.PLS_hole < 0.1].index)}")

# ---------------- figure ----------------
SURF, INK2, GREY = "#fcfcfb", "#52514e", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
b = tab.sort_values("PLS_hole")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, max(5, 0.26 * len(b))), gridspec_kw={"width_ratios": [1.4, 1]})
ypos = np.arange(len(b))
major = b["type"].eq("major element").to_numpy()
a1.barh(ypos, b["PLS_hole"].clip(lower=-0.2), height=0.6, color=np.where(major, "#2a78d6", "#86b6ef"))
a1.scatter(b["Boosting_hole"].clip(lower=-0.2), ypos, marker="o", s=14, facecolors="none", edgecolors="#0b0b0b", zorder=3)
a1.scatter(b["PLS_company"].clip(lower=-0.2), ypos, marker="|", s=80, color="#e34948", zorder=3)
a1.set_yticks(ypos, [f"{e} ({u})" for e, u in zip(b.index, b["unit"])])
a1.axvline(0, color=INK2, lw=0.8); a1.axvline(0.3, color=GREY, lw=0.8, ls="--")
a1.set_xlim(-0.2, 1); a1.set_xlabel("Skill vs guessing the training average (held-out data; < -0.2 clipped)")
a1.legend(handles=[Patch(color="#2a78d6", label="PLS, new hole - major element"),
                   Patch(color="#86b6ef", label="PLS, new hole - trace element"),
                   Line2D([], [], marker="o", ls="", mfc="none", mec="#0b0b0b", label="boosting, new hole"),
                   Line2D([], [], marker="|", ls="", color="#e34948", markersize=10, label="PLS, new company's holes")],
          frameon=False, fontsize=7.5, loc="lower right")
y, p = store[("MgO", "PLS", "company")] if ("MgO", "PLS", "company") in store else next(iter(store.values()))
a2.scatter(y, p, s=2, alpha=0.25, color="#2a78d6", linewidths=0, rasterized=True)
lim = [min(y.min(), p.min()), max(y.max(), p.max())]
a2.plot(lim, lim, color=INK2, lw=1, ls="--")
a2.set_xlabel("Measured log10 MgO (%, lab assay)"); a2.set_ylabel("Predicted from HyLogger minerals")
r_ = res[(res.element == "MgO") & (res.model == "PLS") & (res.held_out == "company")]
if len(r_):
    a2.set_title(f"MgO, PLS, whole companies held out: skill {r_.skill.item()}", loc="left", fontsize=9)
fig.suptitle("Which properties can HyLogger minerals predict in holes the model has never seen?", x=0.01,
             ha="left", weight="bold")
fig.savefig(PLOTS / "16_bulk_prediction.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nFigure: 16_bulk_prediction.png")
