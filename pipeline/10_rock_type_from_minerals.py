"""10 - Determine rock type from HyLogger minerals and check it against the geology logs.

Follows GSWA's 'Determining rocktype' idea (ternary + pseudologs):
  felsic   = TIR Silica + TIR K-Feldspar
  mafic    = TIR Pyroxene + TIR Amphibole
  chlorite = TIR Chlorite
Each interval's three values are rescaled to sum to 1 (a ternary position), then classified:
  Felsic index FI = felsic / (felsic + mafic)
    FI >= 0.75 and chlorite < 0.25 -> Felsic
    FI >= 0.40                     -> Intermediate
    otherwise                      -> Mafic
  Chlorite modifier on Intermediate / Mafic: weak (< 0.25), moderate (0.25-0.5), strong (> 0.5)
  felsic + mafic + chlorite < 0.15 of the interval -> 'Unclassified (low signal)'
Thresholds are OUR choices (set below) - easy to change.
Low-confidence intervals are not classified (removed in 05; MRDH400 is removed in 02).

Calibration: the two felsic-index cut-offs are tuned with leave-one-hole-out, using the SAME rule as
above (chlorite modifier included) and a minimum gap of 0.10 between them so 'Intermediate' always
exists. Reported: interval-weighted AND mean per-hole accuracy, plus per-class recall. The logs are a
reference, not verified ground truth (terms and scale vary between geologists).

Check against logs: only EXPLICIT words are used as 'truth' (lithology name, GSWA rock name,
geologist comments: felsic / intermediate / mafic / granite / dolerite / basalt / gabbro ...).
Company codes (e.g. 'Mb', 'IFD') are company-specific and NOT interpreted.

Also: logged lithology vs mineral clusters (script 05) = patterns across rock classifications.
Outputs: outputs/10_rock_type_intervals.csv, outputs/10_rock_type_agreement.csv,
         outputs/plots/10a_ternary.png, outputs/plots/10b_pseudologs.png
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)

FELSIC_FI, INTER_FI = 0.75, 0.40          # felsic-index cut-offs
CHL_WEAK, CHL_STRONG = 0.25, 0.50         # chlorite modifier cut-offs
MIN_SIGNAL = 0.15                         # min share of the interval explained by the 3 end-members

d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
full = pd.read_csv(OUT / "03_merged_intervals.csv", low_memory=False)
d = d.merge(full[["Hole", "From_m", "Geology_comments"]], on=["Hole", "From_m"], how="left") \
    if "Geology_comments" not in d else d

d = d.copy()                                  # de-fragment the wide table before adding columns
d["felsic"] = d["TIR_Silica"] + d["TIR_K-Feldspar"]
d["mafic"] = d["TIR_Pyroxene"] + d["TIR_Amphibole"]
d["chlorite"] = d["TIR_Chlorite"]
s = d[["felsic", "mafic", "chlorite"]].sum(axis=1)
for c in ["felsic", "mafic", "chlorite"]:
    d[c + "_n"] = np.where(s > 0, d[c] / s, np.nan)
fi = d["felsic_n"] / (d["felsic_n"] + d["mafic_n"]).replace(0, np.nan)

def broad_class(fi_vals, chl, felsic_fi, inter_fi):
    """ONE rule used everywhere (fixed and calibrated cut-offs)."""
    fi_vals, chl = np.asarray(fi_vals, float), np.asarray(chl, float)
    return np.where((fi_vals >= felsic_fi) & (chl < CHL_WEAK), "Felsic",
                    np.where(fi_vals >= inter_fi, "Intermediate", "Mafic"))

def classify(i):
    if s[i] < MIN_SIGNAL or pd.isna(fi[i]): return "Unclassified (low signal)"
    c = d.at[i, "chlorite_n"]
    mod = "weak chl" if c < CHL_WEAK else ("mod chl" if c < CHL_STRONG else "strong chl")
    b = broad_class([fi[i]], [c], FELSIC_FI, INTER_FI)[0]
    return b if b == "Felsic" else f"{b}, {mod}"
d["HyLogger_rock"] = [classify(i) for i in d.index]
d["HyLogger_broad"] = d["HyLogger_rock"].str.split(",").str[0]

# ---- 'truth' from explicit words in the logs ----
RULES = [("Felsic", r"\bfelsic\b|granite|granitic|syenogranite|rhyolit|pegmatite|granodiorit|tonalit"),
         ("Intermediate", r"\bintermediate\b|intermedite|andesit|diorit"),
         ("Mafic", r"\bmafic\b|dolerite|basalt|gabbro|amphibolite")]
def logged(row):
    text = " ".join(str(row[c]) for c in ["Lith_name", "GSWA_rock", "Geology_comments"] if pd.notna(row.get(c))).lower()
    hits = [name for name, pat in RULES if re.search(pat, text)]
    return hits[0] if len(hits) == 1 else (None if not hits else "Mixed")
d["Logged_broad"] = d.apply(logged, axis=1)
d.to_csv(OUT / "10_rock_type_intervals.csv", index=False)

print("HyLogger rock type, all classified intervals:")
print(d["HyLogger_rock"].value_counts().to_string())

cmp_ = d[d["Logged_broad"].isin(["Felsic", "Intermediate", "Mafic"]) & d["HyLogger_broad"].isin(["Felsic", "Intermediate", "Mafic"])]
print(f"\nIntervals with an explicit logged rock type AND a HyLogger class: {len(cmp_)} "
      f"(holes: {sorted(cmp_['Hole'].unique())})")
cm = pd.crosstab(cmp_["Logged_broad"], cmp_["HyLogger_broad"], margins=True)
print("\nConfusion table (rows = geologist's log, columns = HyLogger minerals):\n", cm.to_string())
exact = (cmp_["Logged_broad"] == cmp_["HyLogger_broad"]).mean()
order = {"Felsic": 0, "Intermediate": 1, "Mafic": 2}
within1 = ((cmp_["Logged_broad"].map(order) - cmp_["HyLogger_broad"].map(order)).abs() <= 1).mean()
fm = cmp_[cmp_["Logged_broad"].isin(["Felsic", "Mafic"]) & cmp_["HyLogger_broad"].isin(["Felsic", "Mafic"])]
print(f"\nExact agreement: {exact:.0%} | within one class (no felsic<->mafic swap): {within1:.0%} | "
      f"felsic vs mafic only: {(fm['Logged_broad'] == fm['HyLogger_broad']).mean():.0%} (n={len(fm)})")
agree = cmp_.assign(ok=cmp_.Logged_broad == cmp_.HyLogger_broad).groupby("Hole").agg(n=("ok", "size"), exact=("ok", "mean")).round(2)
print("\nAgreement per hole:\n", agree.to_string())
agree.to_csv(OUT / "10_rock_type_agreement.csv")

# ---- calibrated cut-offs, tested fairly (leave-one-hole-out) ----
# The fixed cut-offs above are a first guess. Here the two felsic-index cut-offs are tuned on all
# labelled holes EXCEPT one, then scored on the hole left out (repeated for every hole), so the
# reported accuracy is on holes the tuning never saw.
lab = d[d["Logged_broad"].isin(["Felsic", "Intermediate", "Mafic"]) & d["HyLogger_broad"].isin(["Felsic", "Intermediate", "Mafic"])].copy()
lab["FI"] = fi[lab.index]
def cls(df, a, b):
    return broad_class(df["FI"], df["chlorite_n"], a, b)
steps = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)]
grid = [(a, b) for a in steps for b in steps if a >= 0.50 and a - b >= 0.10 - 1e-9]   # Intermediate band >= 0.10 wide
rows = []
for h in lab["Hole"].unique():
    tr, te = lab[lab["Hole"] != h], lab[lab["Hole"] == h]
    best = max(grid, key=lambda ab: (cls(tr, *ab) == tr["Logged_broad"]).mean())
    lab.loc[te.index, "Pred_calibrated"] = cls(te, *best)
    rows.append({"held_out_hole": h, "n": len(te), "best_cutoffs_on_others": best,
                 "accuracy_on_held_out": (cls(te, *best) == te["Logged_broad"]).mean().round(2),
                 "accuracy_fixed_cutoffs": (te["Logged_broad"] == te["HyLogger_broad"]).mean().round(2)})
loho = pd.DataFrame(rows).sort_values("n", ascending=False)
w = lambda col: (loho[col] * loho["n"]).sum() / loho["n"].sum()
print("\nCalibrated cut-offs, leave-one-hole-out:\n", loho.to_string(index=False))
print(f"Overall, weighted by intervals: fixed cut-offs {w('accuracy_fixed_cutoffs'):.0%} -> calibrated {w('accuracy_on_held_out'):.0%}")
print(f"Overall, mean of holes (each hole counts once): fixed {loho['accuracy_fixed_cutoffs'].mean():.0%} -> "
      f"calibrated {loho['accuracy_on_held_out'].mean():.0%}")
print("\nCalibrated (held-out) confusion table, rows = log:")
print(pd.crosstab(lab["Logged_broad"], lab["Pred_calibrated"], margins=True).to_string())
rec = lambda col: (lab[col] == lab["Logged_broad"]).groupby(lab["Logged_broad"]).mean()   # recall per logged class
print("Recall per logged class (fixed -> calibrated):",
      {k: f"{a:.0%} -> {b:.0%}" for k, a, b in zip(rec("HyLogger_broad").index, rec("HyLogger_broad"), rec("Pred_calibrated"))})
best_all = max(grid, key=lambda ab: (cls(lab, *ab) == lab["Logged_broad"]).mean())
print(f"Candidate cut-offs tuned on ALL labelled holes (to be tested on new holes, not validated): "
      f"Felsic FI >= {best_all[0]:.2f}, Intermediate FI >= {best_all[1]:.2f}")
loho.to_csv(OUT / "10_rock_type_calibration_loho.csv", index=False)

# ---- clusters vs logged lithology (patterns across rock classifications) ----
lc = d[d["Logged_broad"].isin(["Felsic", "Intermediate", "Mafic"])]
print("\nMineral clusters (script 05) within each logged rock type (% of intervals):")
print((pd.crosstab(lc["Logged_broad"], lc["Cluster"], normalize="index") * 100).round(0).astype(int).to_string())

# ---- figures ----
CAT = {"Felsic": "#e34948", "Intermediate": "#eda100", "Mafic": "#2a78d6"}
GREY = "#b9b8b3"; INK2 = "#52514e"; SURF = "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9})
def tern(f, m, c):  # felsic top, mafic bottom-left, chlorite bottom-right
    return c + f / 2, f * np.sqrt(3) / 2
fig, ax = plt.subplots(figsize=(7, 6.3))
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=INK2, lw=1)
for k in [0.25, 0.5, 0.75]:
    ax.plot(*zip(tern(k, 1 - k, 0), tern(k, 0, 1 - k)), color=GREY, lw=0.5)
p = d[d["HyLogger_broad"].isin(CAT)]
for name, col in CAT.items():
    q = p[p["HyLogger_broad"] == name]
    x, y = tern(q["felsic_n"], q["mafic_n"], q["chlorite_n"])
    ax.scatter(x, y, s=6, color=col, alpha=0.45, linewidths=0, label=f"{name} ({len(q)})")
ax.text(0.5, np.sqrt(3) / 2 + 0.03, "Felsic\n(silica + K-feldspar)", ha="center", fontsize=9)
ax.text(-0.02, -0.06, "Mafic\n(pyroxene + amphibole)", ha="center", fontsize=9, va="top")
ax.text(1.02, -0.06, "Chlorite", ha="center", fontsize=9, va="top")
ax.set_aspect("equal"); ax.axis("off"); ax.legend(frameon=False, loc="upper right")
ax.set_title("HyLogger TIR ternary: each dot is one metre of core", loc="left", weight="bold", pad=42)
fig.savefig(PLOTS / "10a_ternary.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# pseudologs for holes with explicit logged rock types
show = cmp_["Hole"].value_counts()
show = show[show >= 40].index.tolist()[:4]
if show:
    fig, axes = plt.subplots(1, len(show), figsize=(3.2 * len(show), 8), squeeze=False)
    for ax, hole in zip(axes[0], show):
        h = d[d["Hole"] == hole].sort_values("From_m")
        broad = h["HyLogger_broad"].where(h["HyLogger_broad"].isin(CAT))
        # 'large scale' = most common class in a 9 m DEPTH window: put metres on a full 1 m grid first,
        # so unscanned gaps stay empty and are not bridged by the window
        codes = broad.map({"Felsic": 0, "Intermediate": 1, "Mafic": 2})
        codes.index = np.floor(h["From_m"]).astype(int)
        codes = codes[~codes.index.duplicated()]
        grid_m = codes.reindex(range(codes.index.min(), codes.index.max() + 1))
        large = grid_m.rolling(9, center=True, min_periods=3).apply(
            lambda w: pd.Series(w).dropna().mode().iloc[0] if pd.Series(w).notna().any() else np.nan, raw=True)
        large = pd.Series(large.reindex(np.floor(h["From_m"]).astype(int)).values, index=h.index)
        inv = {0: "Felsic", 1: "Intermediate", 2: "Mafic"}
        for j, (series, label) in enumerate([(broad, "HyLogger\n(1 m)"), (large.map(inv), "HyLogger\n(9 m smoothed)"),
                                             (h["Logged_broad"].where(h["Logged_broad"].isin(CAT)), "Geology\nlog")]):
            cols = [CAT.get(v, "#efeeea") for v in series]
            ax.barh(h["From_m"], 0.9, left=j, height=h["To_m"] - h["From_m"], color=cols, align="edge", linewidth=0)
            ax.text(j + 0.45, h["From_m"].min() - 4, label, ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xlim(-0.1, 2.9); ax.set_xticks([]); ax.invert_yaxis(); ax.set_title(hole, weight="bold", pad=28)
        ax.set_ylabel("Depth (m)")
        for sp in ["top", "right", "bottom"]: ax.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=n) for n, c in CAT.items()] + [Patch(color="#efeeea", label="not classified / not logged")],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Rock type from HyLogger minerals vs the geologist's log", x=0.02, ha="left", weight="bold")
    fig.savefig(PLOTS / "10b_pseudologs.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nFigures: 10a_ternary.png, 10b_pseudologs.png  (pseudolog holes:", show, ")")
