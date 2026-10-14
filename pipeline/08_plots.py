"""08 - Figures for the pitch / report (PNG, saved to outputs/plots/).

  08a_strip_<HOLE>.png      downhole strip log per hole: SWIR mineral mix | cluster | Au (log) | anomalies
  08b_clusters_all_holes.png cluster column for all 12 holes side by side
  08c_pca_gold.png           PC1 vs PC2 coloured by Au (log), cluster centres labelled
  08d_cluster_gold_lift.png  gold lift per cluster after controlling for hole (from script 06)
  08e_cluster_minerals.png   heatmap of mean mineral proportion per cluster
  08f_hole_map.png           hole locations (lat/lon), sized by gold intervals
Colours: colourblind-validated categorical palette in fixed order; one-hue blue ramp for magnitudes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#b9b8b3"; INK = "#0b0b0b"; INK2 = "#52514e"; SURF = "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GREY,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv(OUT / "07_intervals_with_anomaly.csv", low_memory=False)
holes = pd.read_csv(OUT / "02_holes.csv")
lift = pd.read_csv(OUT / "06_cluster_gold_lift.csv")

# ---- automatic cluster names: dominant mineral + most ENRICHED mineral vs dataset average ----
mins = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].mean() > 0.005]
prof = d.groupby("Cluster")[mins].mean()
def short(c): return c.replace("SWIR_", "").replace("TIR_", "")
enrich = (prof - d[mins].mean()) / d[mins].std()          # z-score of cluster mean
names = {}
for k in prof.index:
    dom = short(prof.loc[k].idxmax())
    enr = [short(c) for c in enrich.loc[k].sort_values(ascending=False).index if short(c) != dom][0]
    names[k] = f"C{k}: {dom}-rich, high {enr}"
clusters = sorted(names)
cl_cmap = ListedColormap(CAT[:len(clusters)])

# ---- SWIR mineral groups for the strip log (fixed order, max 8 incl. Other) ----
GROUPS = {"White mica": ["SWIR_White-Mica"], "Chlorite": ["SWIR_Chlorite"], "Amphibole": ["SWIR_Amphibole"],
          "Dark mica": ["SWIR_Dark-Mica"], "Carbonate": ["SWIR_Carbonate"], "Epidote": ["SWIR_Epidote"],
          "Clays (kaolin/smectite)": ["SWIR_Kaolin", "SWIR_Smectite"]}
for g in GROUPS: d[g] = d[[c for c in GROUPS[g] if c in d]].sum(axis=1)
grp = list(GROUPS)
d["Other / invalid"] = (1 - d[grp].sum(axis=1)).clip(lower=0)
grp_all = grp + ["Other / invalid"]
grp_cols = CAT[:7] + [GREY]

# ---- 08a strip logs ----
for hole, h in d.sort_values("From_m").groupby("Hole"):
    fig, ax = plt.subplots(1, 4, figsize=(8.5, 10), sharey=True,
                           gridspec_kw={"width_ratios": [4, 0.6, 2, 0.5], "wspace": 0.08})
    y0, y1 = h["From_m"].values, h["To_m"].values
    left = np.zeros(len(h))
    for g, col in zip(grp_all, grp_cols):
        ax[0].barh(y0, h[g].values, height=y1 - y0, left=left, color=col, align="edge", linewidth=0)
        left += h[g].values
    ax[0].set_xlim(0, 1); ax[0].set_xlabel("SWIR mineral proportion"); ax[0].set_ylabel("Depth (m)")
    # cluster runs: alternating grey bands, labelled C# where the run is >= 4 m
    run = (h["Cluster"] != h["Cluster"].shift()).cumsum()
    for j, (_, r) in enumerate(h.groupby(run)):
        top, bot = r["From_m"].min(), r["To_m"].max()
        ax[1].barh(top, 1, height=bot - top, color=["#e4e3df", "#cfcec9"][j % 2], align="edge", linewidth=0)
        if bot - top >= 4:
            ax[1].text(0.5, (top + bot) / 2, f"C{r['Cluster'].iloc[0]}", ha="center", va="center", fontsize=6.5, color=INK)
    ax[1].set_xlim(0, 1); ax[1].set_xticks([]); ax[1].set_xlabel("Cluster")
    au = h["Au_PPM"]
    ax[2].barh(y0, au.fillna(0).clip(lower=0.001), height=y1 - y0, color="#2a78d6", align="edge", linewidth=0)
    ax[2].set_xscale("log"); ax[2].set_xlim(0.001, max(1, au.max() * 1.5) if au.notna().any() else 1)
    ax[2].axvline(0.5, color=INK2, lw=1, ls="--"); ax[2].set_xlabel("Au ppm (log)\n-- 0.5 ppm")
    an = h[h["Is_anomaly"]]
    ax[3].scatter(np.full(len(an), 0.5), an["Mid_m"], marker="<", s=40, color=INK)
    ax[3].set_xlim(0, 1); ax[3].set_xticks([]); ax[3].set_xlabel("Anomaly")
    ax[0].invert_yaxis()
    for a in ax: a.grid(axis="x", color=GREY, lw=0.4, alpha=0.5)
    fig.suptitle(f"{hole} — HyLogger mineralogy, clusters and gold", x=0.06, ha="left", fontsize=12, weight="bold")
    h1 = [Patch(color=c, label=g) for g, c in zip(grp_all, grp_cols)]
    fig.legend(handles=h1, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.06, -0.005), fontsize=8)
    fig.text(0.06, -0.06, "Clusters:  " + "   ".join(names[k] for k in clusters[:4]) + "\n" + " " * 18 +
             "   ".join(names[k] for k in clusters[4:]), fontsize=7.5, color=INK2, va="top")
    fig.savefig(PLOTS / f"08a_strip_{hole}.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 08b all holes cluster columns ----
order = d.groupby("Hole")["To_m"].max().sort_values().index
fig, ax = plt.subplots(figsize=(11, 6.5))
for i, hole in enumerate(order):
    h = d[d["Hole"] == hole]
    ax.bar(np.full(len(h), i), h["To_m"] - h["From_m"], bottom=h["From_m"], width=0.7,
           color=[CAT[c] for c in h["Cluster"]], linewidth=0)
    g = h[h["Au_PPM"] >= 0.5]
    ax.scatter(np.full(len(g), i + 0.45), g["Mid_m"], s=6, color=INK, marker="_")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right")
ax.invert_yaxis(); ax.set_ylabel("Depth (m)")
ax.set_title("Mineral assemblage clusters down each hole  (black ticks = Au ≥ 0.5 ppm)", loc="left", weight="bold")
ax.legend(handles=[Patch(color=CAT[k], label=names[k]) for k in clusters], frameon=False,
          bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
fig.savefig(PLOTS / "08b_clusters_all_holes.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 08c PCA coloured by gold ----
fig, ax = plt.subplots(figsize=(8, 6.5))
no = d[d["Au_PPM"].isna()]
ax.scatter(no["PC1"], no["PC2"], s=6, color=GREY, alpha=0.5, label="no assay", linewidths=0)
ya = d[d["Au_PPM"].notna()].sort_values("Au_PPM")
sc = ax.scatter(ya["PC1"], ya["PC2"], c=ya["Au_PPM"].clip(lower=0.002), cmap=BLUE_RAMP,
                norm=LogNorm(0.002, 10), s=9, linewidths=0)
cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("Au ppm (log)"); cb.outline.set_visible(False)
for k in clusters:
    c = d[d["Cluster"] == k][["PC1", "PC2"]].mean()
    ax.annotate(f"C{k}", c, ha="center", va="center", fontsize=10, weight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec=GREY, lw=0.8))
ax.set_xlabel("PC1  (white mica + carbonate  ←→  amphibole + epidote)")
ax.set_ylabel("PC2  (dark mica + plagioclase  ←→  chlorite + Fe)")
ax.set_title("Intervals in PCA space, coloured by gold grade", loc="left", weight="bold")
ax.legend(frameon=False, loc="lower right")
fig.savefig(PLOTS / "08c_pca_gold.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 08d gold lift ----
L = lift.sort_values("lift"); L["name"] = L["Cluster"].map(names)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(L["name"], L["lift"], color="#2a78d6", height=0.6)
ax.axvline(1, color=INK2, lw=1, ls="--")
for y, (v, n) in enumerate(zip(L["lift"], L["n"])):
    ax.text(v + 0.03, y, f"{v:.2f}×  (n={n})", va="center", fontsize=8, color=INK2,
            bbox=dict(fc=SURF, ec="none", pad=1), zorder=3)
ax.set_xlabel("Gold lift (observed ÷ expected Au ≥ 0.5 ppm intervals, controlling for hole)")
ax.set_title("Which assemblages are richer in gold than their own hole's average?", loc="left", weight="bold")
ax.set_xlim(0, L["lift"].max() * 1.3)
fig.savefig(PLOTS / "08d_cluster_gold_lift.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 08e cluster mineral heatmap ----
keep = [c for c in mins if prof[c].max() > 0.05]
P = prof[keep]
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(P.values, cmap=BLUE_RAMP, aspect="auto", vmin=0, vmax=P.values.max())
ax.set_xticks(range(len(keep))); ax.set_xticklabels(keep, rotation=45, ha="right")
ax.set_yticks(range(len(P))); ax.set_yticklabels([names[k] for k in P.index])
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        v = P.values[i, j]
        if v >= 0.15: ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                               color="white" if v > P.values.max() * 0.55 else INK)
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Mean proportion"); cb.outline.set_visible(False)
ax.set_title("Average mineral make-up of each cluster", loc="left", weight="bold")
fig.savefig(PLOTS / "08e_cluster_minerals.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---- 08f map ----
gc = d.groupby("Hole").agg(gold=("Au_PPM", lambda s: (s >= 0.5).sum())).reset_index()
m = holes.merge(gc, on="Hole")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(m["Lon"], m["Lat"], s=30 + m["gold"] * 4, color="#2a78d6", alpha=0.85, edgecolor=SURF, lw=1.5)
OFFSET = {"ROAD003": (8, 8), "BWD072": (-12, -4), "BNRD005": (8, 4), "BSDD019": (-8, -12)}
for _, r in m.iterrows():
    dx, dy = OFFSET.get(r["Hole"], (6, 4))
    ax.annotate(f"{r['Hole']} ({r['gold']})", (r["Lon"], r["Lat"]), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, ha="right" if dx < 0 else "left")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
ax.set_title("Starter holes, Yilgarn Craton  (size / label = intervals with Au ≥ 0.5 ppm)", loc="left", weight="bold")
ax.grid(color=GREY, lw=0.4, alpha=0.5)
fig.savefig(PLOTS / "08f_hole_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("Cluster names:"); [print(" ", v) for v in names.values()]
print("\nSaved:", sorted(p.name for p in PLOTS.glob("*.png")))
