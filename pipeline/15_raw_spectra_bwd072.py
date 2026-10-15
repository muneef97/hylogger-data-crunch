"""15 - Raw HyLogger spectra, analysed OUTSIDE TSG (GSWA Data Crunch task 2: "Feature analysis").

One hole (BWD072, our best gold hole). Input: the raw SWIR spectra (BWD072_tsg.tsg + .bip, ~76,000
spectra, one every ~6 mm) read with the open-source `pytsg` reader - no TSG software involved.
  1) Quality: spectra flagged by TSG's own 'Final Mask' are dropped (count printed).
  2) Our own feature extraction: hull quotient (spectrum / upper convex hull, 1300-2500 nm) for every
     spectrum; the white-mica Al-OH absorption is the deepest point between 2180 and 2235 nm.
     Position = sub-band minimum (parabola through 3 points); depth = 1 - hull quotient. The minimum
     must be a true local minimum inside the window (an edge minimum is the shoulder of chlorite at 2250).
     A position is only recorded when the absorption is >= 2 % deep (otherwise there is no real
     white-mica feature and the position would be noise).
  3) Down-hole feature map (wavelength x depth, colour = median absorption depth of ALL features, not
     white-mica abundance), like GSWA's slide; 0.5 m bins with < 10 good spectra are left grey.
  4) Check against TSG: (a) per spectrum, vs TSG's own extracted feature in the same window (its
     deepest feature there - not necessarily labelled white mica by TSG); rejection reasons counted;
     (b) per metre, vs 2200W_AlOH in the 1 m summary log. Plus: averaging features vs averaging spectra.
  5) Gold (exploratory): is the white-mica position different in gold metres (Au >= 0.5 ppm) than in
     background metres (Au < 0.05)? 90 % interval from a 10 m block bootstrap. Repeated using only
     spectra TSG calls white mica, so a change in the chlorite/mica mix is not mistaken for a change in
     mica chemistry (shorter position = more Al-rich mica, e.g. paragonite/muscovite; longer = phengite).
Note: ~6 mm is the SAMPLING spacing, not a guaranteed 6 mm geological resolution.
Needs: pip install pytsg ; files in <data>/TSG/BWD072/ (see tools/fetch_tsg_spectra.py)
Outputs: outputs/15_bwd072_metre_features.csv, outputs/plots/15a_example_spectra.png,
         outputs/plots/15b_feature_map.png, outputs/plots/15c_tsg_check_and_gold.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA_DIR, OUT

HOLE = "BWD072"
TSG_DIR = DATA_DIR / "TSG" / HOLE
WIN = (2180, 2235)            # white-mica Al-OH window (nm)
MIN_DEPTH = 0.02              # minimum absorption depth for a real feature
PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
try:
    from pytsg.parse_tsg import read_tsg_bip_pair
except ImportError:
    sys.exit("pytsg is not installed: pip install pytsg")
if not (TSG_DIR / f"{HOLE}_tsg.bip").exists():
    sys.exit(f"Raw spectra not found in {TSG_DIR} - run tools/fetch_tsg_spectra.py first")

# ---------------- 1) read + quality ----------------
s = read_tsg_bip_pair(TSG_DIR / f"{HOLE}_tsg.tsg", TSG_DIR / f"{HOLE}_tsg.bip", "nir")
wl = s.wavelength.astype(float)
depth_m = pd.to_numeric(s.sampleheaders["D"]).to_numpy()
ok = s.scalars["Final Mask"].to_numpy() == 1
print(f"{HOLE}: {len(depth_m):,} SWIR spectra, {wl.min():.0f}-{wl.max():.0f} nm in {len(wl)} bands, "
      f"{depth_m.min():.1f}-{depth_m.max():.1f} m (~{len(depth_m) / (depth_m.max() - depth_m.min()):.0f} spectra per metre)")
print(f"Quality: {ok.sum():,} spectra kept, {(~ok).sum():,} dropped by TSG's Final Mask ({(~ok).mean():.1%})")

# ---------------- 2) hull quotient + white-mica feature ----------------
rng_ = (wl >= 1300) & (wl <= 2500)
x = wl[rng_]
def hull_quotient(y):
    """Spectrum divided by its upper convex hull (monotone chain)."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (x[h[-1]] - x[h[-2]]) * (y[i] - y[h[-2]]) - (y[h[-1]] - y[h[-2]]) * (x[i] - x[h[-2]]) >= 0:
            h.pop()
        h.append(i)
    return y / np.interp(x, x[h], y[h])

w_idx = np.where((x >= WIN[0]) & (x <= WIN[1]))[0]
def mica_feature(hq):
    """Deepest point in the window - only counted if it is a true local minimum INSIDE the window.
    A minimum on the window edge is the shoulder of a neighbouring feature (e.g. chlorite at 2250 nm)."""
    j = w_idx[np.argmin(hq[w_idx])]
    d = 1 - hq[j]
    if j <= w_idx[0] + 1 or j >= w_idx[-1] - 1:
        return np.nan, np.nan, "edge minimum"
    if not (hq[j] < hq[j - 1] and hq[j] < hq[j + 1]):
        return np.nan, np.nan, "no local minimum"
    if 0 < j < len(x) - 1:                                   # parabola through 3 points -> sub-band centre
        y0, y1, y2 = hq[j - 1], hq[j], hq[j + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        pos = x[j] + np.clip(off, -1, 1) * (x[1] - x[0])
    else:
        pos = x[j]
    if d < MIN_DEPTH:
        return np.nan, np.nan, f"shallower than {MIN_DEPTH:.0%}"
    return pos, d, "valid"

X = s.spectra[:, rng_].astype(float)
pos = np.full(len(X), np.nan); dep = np.full(len(X), np.nan)       # NaN = no valid feature (never 0)
reason = np.full(len(X), "masked by TSG", dtype=object)
img_rows = {}                                                 # absorption depth per 0.5 m bin (map)
for k in np.where(ok)[0]:
    hq = hull_quotient(X[k])
    pos[k], dep[k], reason[k] = mica_feature(hq)
    img_rows.setdefault(np.floor(depth_m[k] * 2) / 2, []).append(1 - hq)
has_mica = reason == "valid"                                         # validity flag
print(f"White-mica feature ({WIN[0]}-{WIN[1]} nm, >= {MIN_DEPTH:.0%} deep) found in {has_mica.sum():,} of {ok.sum():,} "
      f"good spectra ({has_mica.sum() / ok.sum():.0%}); median position {np.nanmedian(pos):.1f} nm")

# ---------------- 4a) per-spectrum check vs TSG's own feature ----------------
sc = s.scalars
cent = sc[[c for c in sc if c.startswith("Centre") and c[6:].isdigit()]].to_numpy(float)
dpth = sc[[c for c in sc if c.startswith("Depth") and c[5:].isdigit()]].to_numpy(float)
inwin = (cent >= WIN[0]) & (cent <= WIN[1]) & (dpth > 0) & (dpth < 1)
dmask = np.where(inwin, dpth, -1)
best = dmask.argmax(axis=1)
tsg_pos = np.where(inwin.any(axis=1), cent[np.arange(len(cent)), best], np.nan)
tsg_dep = np.where(inwin.any(axis=1), dpth[np.arange(len(dpth)), best], np.nan)
both = has_mica & ~np.isnan(tsg_pos)
r_pos = np.corrcoef(pos[both], tsg_pos[both])[0, 1]
r_dep = np.corrcoef(dep[both], tsg_dep[both])[0, 1]
print(f"\nCheck vs TSG's extracted feature in the same {WIN[0]}-{WIN[1]} nm window (TSG's deepest feature there; "
      f"not necessarily labelled white mica by TSG), per spectrum (n = {both.sum():,} where both found one):")
print(f"  position: r = {r_pos:.2f}, median |ours - TSG| = {np.median(np.abs(pos[both] - tsg_pos[both])):.1f} nm")
print(f"  depth:    r = {r_dep:.2f}, median ours {np.median(dep[both]):.3f} vs TSG {np.median(tsg_dep[both]):.3f}")
tsg_min = sc["Min1 sTSAS"].astype(str).to_numpy()
WHITE_MICA = ["Paragonite", "Muscovite", "Phengite"]
print("  our position by TSG's own mineral call (known order: paragonite < muscovite < phengite):")
for mn in WHITE_MICA + ["Chlorite-Fe", "Chlorite-FeMg"]:
    k_ = has_mica & (tsg_min == mn)
    if k_.sum(): print(f"    {mn:14s} n = {k_.sum():6,}  median position {np.nanmedian(pos[k_]):.1f} nm")
why = pd.Series(reason[ok & ~has_mica & ~np.isnan(tsg_pos)]).value_counts()
print("  where TSG has a feature but we reject it, our reason:", why.to_dict())
only_ours, only_tsg = (has_mica & np.isnan(tsg_pos)).sum(), (ok & ~has_mica & ~np.isnan(tsg_pos)).sum()
print(f"  feature found by us only: {only_ours:,} | by TSG only: {only_tsg:,}")

# ---------------- per metre + 4b) vs 1 m summary ----------------
is_wm = np.isin(tsg_min, WHITE_MICA)
spec = pd.DataFrame({"From_m": np.floor(depth_m), "pos": pos, "dep": np.where(ok, dep, np.nan),
                     "mica": has_mica.astype(float), "ok": ok, "wm": is_wm.astype(float),
                     "pos_wm": np.where(is_wm, pos, np.nan)})
spec = spec[spec["ok"]]
metre = spec.groupby("From_m").agg(n_good=("ok", "size"), mica_share=("mica", "mean"), wm_share=("wm", "mean"),
                                    pos_median=("pos", "median"), dep_median=("dep", "median"),
                                    pos_wm=("pos_wm", "median"), n_wm=("wm", "sum")).reset_index()
metre.loc[metre["n_good"] < 20, ["pos_median", "dep_median", "mica_share", "wm_share"]] = np.nan   # too few good spectra
metre.loc[metre["n_wm"] < 10, "pos_wm"] = np.nan          # white-mica-only position needs >= 10 such spectra
summ = pd.read_csv(OUT / "03_merged_intervals.csv", low_memory=False)
summ = summ[summ["Hole"] == HOLE][["From_m", "2200W_AlOH", "SWIR_White-Mica", "Au_PPM"]]
metre = metre.merge(summ, on="From_m", how="left")
mm = metre.dropna(subset=["pos_median", "2200W_AlOH"])
r_sum = mm["pos_median"].corr(mm["2200W_AlOH"], method="spearman")
r_wm = metre.dropna(subset=["mica_share", "SWIR_White-Mica"])
r_wm = r_wm["mica_share"].corr(r_wm["SWIR_White-Mica"], method="spearman")
print(f"\nCheck vs the 1 m summary log (n = {len(mm)} metres):")
print(f"  our median position vs 2200W_AlOH: Spearman {r_sum:.2f}; median difference "
      f"{(mm['pos_median'] - mm['2200W_AlOH']).median():+.1f} nm")
print(f"  share of spectra with a white-mica feature vs summary SWIR_White-Mica proportion: Spearman {r_wm:.2f}")

# averaging features vs averaging spectra (20 random metres)
rng = np.random.default_rng(0)
cmp_rows = []
cand = metre.dropna(subset=["pos_median"])["From_m"]
for m_ in rng.choice(cand, min(20, len(cand)), replace=False):
    idx = np.where(ok & (np.floor(depth_m) == m_))[0]
    p_avg_spec, _, _ = mica_feature(hull_quotient(X[idx].mean(axis=0)))
    cmp_rows.append((np.nanmedian(pos[idx]), p_avg_spec))
c = np.array(cmp_rows)
okc = ~np.isnan(c).any(axis=1)
print(f"  averaging features vs averaging spectra ({okc.sum()} of {len(c)} random metres with a feature both ways): "
      f"median |difference| {np.median(np.abs(c[okc, 0] - c[okc, 1])):.1f} nm")
metre.to_csv(OUT / "15_bwd072_metre_features.csv", index=False)

# ---------------- 5) gold (exploratory) ----------------
ga = metre.dropna(subset=["wm_share", "Au_PPM"]).copy()           # all assayed metres with >= 20 good spectra
ga["gold"], ga["bg"] = ga["Au_PPM"] >= 0.5, ga["Au_PPM"] < 0.05
g = ga.dropna(subset=["pos_median"]).copy()                       # ... that also have a measurable feature
def gap(df, col="pos_median"):
    df = df.dropna(subset=[col])
    return df.loc[df["gold"], col].median() - df.loc[df["bg"], col].median()
def boot_ci(df, col):
    blocks = (df["From_m"] // 10).to_numpy(); ub = np.unique(blocks); out = []
    for _ in range(2000):
        b = pd.concat([df[blocks == u] for u in rng.choice(ub, len(ub))])
        if b["gold"].any() and b["bg"].any(): out.append(gap(b, col))
    return np.nanpercentile(out, [5, 95])
lo, hi = boot_ci(g, "pos_median")
lo2, hi2 = boot_ci(g, "pos_wm")
rho = g["pos_median"].corr(g["Au_PPM"], method="spearman")
print("\nGold (exploratory, ONE hole; intervals are within-hole block resampling, not hole-to-hole uncertainty)")
print(f"  (a) white-mica PREVALENCE, all assayed metres with >= 20 good spectra "
      f"({ga['gold'].sum()} gold / {ga['bg'].sum()} background): share of spectra TSG calls white mica "
      f"{ga.loc[ga.gold, 'wm_share'].median():.0%} vs {ga.loc[ga.bg, 'wm_share'].median():.0%}")
print(f"  (b) feature POSITION, those metres that also have a measurable feature "
      f"({g['gold'].sum()} gold / {g['bg'].sum()} background):")
print(f"      median position: gold {g.loc[g.gold, 'pos_median'].median():.1f} nm vs background "
      f"{g.loc[g.bg, 'pos_median'].median():.1f} nm -> difference {gap(g):+.1f} nm "
      f"(90% interval {lo:+.1f} to {hi:+.1f}, 10 m blocks)")
print(f"      same, ONLY spectra TSG calls white mica (removes the effect of chlorite mixing): "
      f"{gap(g, 'pos_wm'):+.1f} nm (90% interval {lo2:+.1f} to {hi2:+.1f}; "
      f"{g.dropna(subset=['pos_wm']).gold.sum()} gold / {g.dropna(subset=['pos_wm']).bg.sum()} background metres)")
print(f"      Spearman (position vs Au): {rho:.2f}")
print("  -> the mix of minerals differs clearly; a change in the mica's own chemistry is NOT established "
      "(its interval includes 0). Sub-band positions come from a parabola fit: not 1 nm instrument resolution.")

# ---------------- figures ----------------
SURF, INK2, GREY = "#fcfcfb", "#52514e", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
# 15a: example spectra (deep mica / shallow mica / no mica)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
picks = [np.nanargmax(np.where(has_mica & (tsg_min == "Muscovite"), dep, np.nan)),
         np.where(has_mica)[0][np.argsort(np.abs(dep[has_mica] - np.nanmedian(dep[has_mica])))[0]],
         np.where(ok & ~has_mica)[0][len(np.where(ok & ~has_mica)[0]) // 2]]
for ax, k, t in zip(axes, picks, ["Strong white-mica feature (TSG: muscovite)", "Typical white-mica feature", "No white-mica feature"]):
    hq = hull_quotient(X[k])
    ax.plot(x, hq, color="#2a78d6", lw=1.2)
    ax.axvspan(*WIN, color="#eda100", alpha=0.15, lw=0)
    if has_mica[k]: ax.axvline(pos[k], color="#e34948", lw=1, ls="--")
    ax.set_title(f"{t}\n{depth_m[k]:.2f} m" + (f" | {pos[k]:.1f} nm, {dep[k]:.0%} deep" if has_mica[k] else ""),
                 loc="left", fontsize=9)
    ax.set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Hull quotient")
fig.suptitle(f"{HOLE}: raw spectra, background removed in Python (shaded = white-mica window)",
             x=0.01, y=1.08, ha="left", weight="bold")
fig.savefig(PLOTS / "15a_example_spectra.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 15b: down-hole feature map + our white-mica position
keys = sorted(img_rows)
img = np.array([np.median(img_rows[k_], axis=0) for k_ in keys])
n_bin = np.array([len(img_rows[k_]) for k_ in keys])
img[n_bin < 10] = np.nan                                          # sparse bins (< 10 good spectra) left blank
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 9), gridspec_kw={"width_ratios": [2.2, 1]}, sharey=True)
im = a1.imshow(np.clip(img, 0, 0.4), aspect="auto", cmap="viridis_r",
               extent=[x[0], x[-1], keys[-1] + 0.5, keys[0]], interpolation="nearest")
a1.set_xlabel("Wavelength (nm)"); a1.set_ylabel("Depth (m)")
a1.set_facecolor("#e6e5e0")                                       # grey = masked / sparse coverage

cb = fig.colorbar(im, ax=a1, shrink=0.6, location="bottom", pad=0.07); cb.set_label("absorption depth (1 - hull quotient), median per 0.5 m - all minerals, not white-mica abundance\n"
             "grey = no or < 10 good spectra in that 0.5 m bin", fontsize=8)
for lab, wv, ha in [("water 1910", 1910, "center"), ("Al-OH 2200 ", 2200, "right"), (" Fe/Mg-OH 2250", 2250, "left"),
                   ("carbonate 2340", 2340, "left")]:
    a1.annotate(lab, (wv, keys[0]), xytext=(0, 4 + 9 * (wv == 2340)), textcoords="offset points",
                ha=ha, va="bottom", fontsize=7, color=INK2, annotation_clip=False)
    a1.axvline(wv, color="white", lw=0.4, alpha=0.6)
a2.scatter(pos[has_mica], depth_m[has_mica], s=0.3, color="#2a78d6", alpha=0.15, linewidths=0, rasterized=True)
mp = metre.dropna(subset=["pos_median"])
a2.plot(mp["pos_median"], mp["From_m"] + 0.5, color="#0b0b0b", lw=0.8, label="our median per metre")
s2 = metre.dropna(subset=["2200W_AlOH"])
a2.plot(s2["2200W_AlOH"], s2["From_m"] + 0.5, color="#eb6834", lw=0.8, alpha=0.8, label="1 m summary (2200W_AlOH)")
gm = metre[metre["Au_PPM"] >= 0.5]
a2.scatter(np.full(len(gm), 2228), gm["From_m"] + 0.5, marker="s", s=6, color="#eda100", label="Au >= 0.5 ppm")
a2.set_xlim(2190, 2230); a2.set_xlabel("White-mica feature position (nm)")
a2.set_title("Every spectrum (dots) and per metre", loc="left", fontsize=9)
a2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1)
a1.invert_yaxis() if a1.get_ylim()[0] < a1.get_ylim()[1] else None
fig.suptitle(f"{HOLE}: spectral features mapped down the hole, outside TSG", x=0.01, ha="left", weight="bold")
fig.savefig(PLOTS / "15b_feature_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# 15c: TSG check + gold
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
sub = np.where(both)[0][::5]
b1.scatter(tsg_pos[sub], pos[sub], s=1, alpha=0.2, color="#2a78d6", linewidths=0, rasterized=True)
b1.plot(WIN, WIN, color=INK2, lw=1, ls="--")
b1.set_xlabel("TSG's feature position (nm)"); b1.set_ylabel("Our feature position (nm)")
b1.set_title(f"Per spectrum: ours vs TSG (r = {r_pos:.2f}, n = {both.sum():,})", loc="left", fontsize=9)
data = [g.loc[g.bg, "pos_median"], g.loc[g.gold, "pos_median"]]
b2.boxplot(data, widths=0.5, showfliers=False)
b2.set_xticks([1, 2], [f"Background\nAu < 0.05 (n={g.bg.sum()})", f"Gold\nAu >= 0.5 (n={g.gold.sum()})"])
b2.set_ylabel("White-mica position, median per metre (nm)")
b2.set_title(f"Gold vs background: {gap(g):+.1f} nm (90% interval {lo:+.1f} to {hi:+.1f})", loc="left", fontsize=9)
fig.savefig(PLOTS / "15c_tsg_check_and_gold.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nFigures: 15a_example_spectra.png, 15b_feature_map.png, 15c_tsg_check_and_gold.png")
