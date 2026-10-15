"""14 - "Can we detect anomalies or ENDMEMBERS in bulk?" (GSWA Data Crunch task 1; anomalies are in 07)

A) Mineral endmembers across all holes (bulk unmixing of the summary mineral logs)
   Every metre is a MIXTURE of minerals. Non-negative matrix factorisation (NMF) finds a small set of
   "endmember" assemblages so that each metre = a weighted mix of them (weights >= 0). Unlike the
   clusters in 05 (one label per metre), this keeps the mixing.
   * Input: SWIR + TIR mineral proportions (Invalid columns excluded), QC-passed metres (from 05).
   * Stability: refit on two random halves of the HOLES and match each endmember to the full fit (cosine
     similarity; 1 = the same endmember appears from independent holes).
   * Number of endmembers k (3-10): the LARGEST k for which every endmember is still stable (median
     cosine >= 0.9 over the half-hole refits). Reconstruction error keeps falling with k, so it cannot
     choose k on its own; stability says how many endmembers the data really support.
   * Weights are rescaled so they match the normalised endmember compositions (W x Hn = the original fit);
     "share" = relative contribution of each endmember to the model of a metre, NOT a % of the rock.
   * Gold (exploratory, as in 06): within-hole Spearman of each endmember's share with Au, in holes
     with >= 8 gold and >= 20 background metres; sign test across holes (holes with |rho| > 0.1), with a
     Benjamini-Hochberg adjustment for testing all endmembers. ~11 holes: exploratory only.
   Reproducibility: fixed random seeds; NMF results can differ slightly between scikit-learn versions.
B) Spectral chemistry in bulk (GSWA's carbonate example used absorption WAVELENGTHS)
   * White-mica Al-OH position (2200W) vs chlorite Fe-OH position (2250W), only where both absorptions
     are real (depth >= 0.02), coloured by region: longer 2200W = more phengitic (Fe/Mg) mica, longer
     2250W = more Fe-rich chlorite.
   * TIR carbonate peak position (11300W) where carbonate is present (depth >= 0.02), by region.
   * Metres from one hole are not independent: hole-level medians are also reported; regions with few
     holes show differences between holes, not proven regional signatures.
   Wavelengths are TSG-derived scalars from the summary logs (raw spectra: script 15).
Outputs: outputs/14_endmembers.csv, outputs/14_endmember_weights.csv,
         outputs/plots/14a_endmembers.png, outputs/plots/14b_spectral_chemistry.png
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import OUT

warnings.filterwarnings("ignore", message=".*constant.*")      # holes where an endmember is absent
PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
MIN_FEATURE_DEPTH = 0.02
d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)
holes = pd.read_csv(OUT / "02_holes.csv").set_index("Hole")
d["Region"] = d["Hole"].map(holes["TectonicRegion"]).fillna("Not recorded")
mins = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].max() > 0]
# Blank = not detected was already established in script 02 (proportions sum to 1 per sensor, checked
# there) and filled with 0 there; here we only confirm nothing is missing, rather than filling silently.
n_missing = int(d[mins].isna().sum().sum())
print(f"  missing mineral cells: {n_missing} of {d[mins].size:,} (blank = not detected, handled in script 02)")
assert n_missing == 0, "unexpected missing mineral values - check script 02"
X = d[mins].clip(lower=0).to_numpy()
short = lambda c: c.replace("SWIR_", "S:").replace("TIR_", "T:")
print(f"A) Endmembers: {len(X):,} metres from {d['Hole'].nunique()} holes, {len(mins)} mineral columns")

def fit(k, data, seed=0):
    m = NMF(n_components=k, init="nndsvda", max_iter=600, random_state=seed)
    w = m.fit_transform(data)
    return m, w

unit = lambda H: H / np.linalg.norm(H, axis=1, keepdims=True)
uh = d["Hole"].unique()
def stability(k, reps, seed=0):
    """Cosine similarity of each full-fit endmember to its best match in fits on half the holes."""
    rng_ = np.random.default_rng(seed)
    Hf = unit(fit(k, X)[0].components_); out = []
    for rep in range(reps):
        half = d["Hole"].isin(set(rng_.choice(uh, len(uh) // 2, replace=False))).to_numpy()
        for part in [half, ~half]:
            c = Hf @ unit(fit(k, X[part], seed=rep)[0].components_).T
            r, cc = linear_sum_assignment(-c)
            assert (r == np.arange(k)).all()               # rows = full-fit endmembers, in full-fit order
            out.append(c[r, cc])
    return np.array(out)
errs, worst = {}, {}
for k in range(3, 11):
    errs[k] = float(fit(k, X)[0].reconstruction_err_ / np.linalg.norm(X))
    worst[k] = float(np.median(stability(k, 5), axis=0).min())
print("  relative reconstruction error by k:", {k: round(v, 3) for k, v in errs.items()})
print("  least stable endmember (median cosine) by k:", {k: round(v, 2) for k, v in worst.items()})
K = max(k for k in worst if worst[k] >= 0.9)
print(f"  chosen k = {K} (largest k where every endmember is stable, median cosine >= 0.9)")

model, W = fit(K, X)
H = model.components_                                           # endmember x mineral
Hsum = H.sum(axis=1)
Hn = H / Hsum[:, None]                                          # each endmember's composition sums to 1
W = W * Hsum[None, :]                                           # rescale weights so W @ Hn == original W @ H
share = W / W.sum(axis=1, keepdims=True).clip(min=1e-12)       # relative contribution of each endmember
order = np.argsort(-share.mean(axis=0))                          # E1 = largest average share
Hn, W, share = Hn[order], W[:, order], share[:, order]
names = []
for i in range(K):
    top = np.argsort(-Hn[i])[:3]
    names.append(f"E{i + 1}: " + " + ".join(f"{short(mins[j])}" for j in top if Hn[i, j] > 0.05))

sims = stability(K, 10, seed=1)[:, order]                        # (20, K), in the same order as the names

# gold, within hole (as in 06)
g = d[d["Au_PPM"].notna()].copy()
for i in range(K): g[f"E{i + 1}"] = share[d["Au_PPM"].notna().to_numpy(), i]
cnt = g.groupby("Hole")["Au_PPM"].agg(gold=lambda s: (s >= 0.5).sum(), bg=lambda s: (s < 0.05).sum())
usable = cnt[(cnt.gold >= 8) & (cnt.bg >= 20)].index
rows = []
for i in range(K):
    rh = [g[g.Hole == h][f"E{i + 1}"].corr(g[g.Hole == h]["Au_PPM"], method="spearman") for h in usable]
    rh = np.array([r for r in rh if pd.notna(r)])
    dom = share[:, i] >= 0.5
    npos, nneg = int((rh > 0.1).sum()), int((rh < -0.1).sum())
    sign_p = binomtest(max(npos, nneg), npos + nneg, 0.5).pvalue if npos + nneg else np.nan
    rows.append({"endmember": names[i], "mean_share": round(share[:, i].mean(), 3),
                 "metres_dominant(>=50%)": int(dom.sum()), "holes_with_dominant": d.loc[dom, "Hole"].nunique(),
                 "stability_cosine_median": round(np.median(sims[:, i]), 2), "stability_cosine_min": round(sims[:, i].min(), 2),
                 "gold_holes_tested": len(rh), "gold_median_rho": round(np.median(rh), 2) if len(rh) else np.nan,
                 "gold_holes_pos": npos, "gold_holes_neg": nneg, "gold_sign_test_p": round(sign_p, 3),
                 "refits_matched(cos>=0.9)": f"{int((sims[:, i] >= 0.9).sum())}/{len(sims)}"})
em = pd.DataFrame(rows)
# Benjamini-Hochberg adjustment over the endmembers tested (7 tests -> small p-values happen by chance)
p_ = em["gold_sign_test_p"].to_numpy(float); ok_ = np.isfinite(p_); q_ = np.full(len(p_), np.nan)
if ok_.any():
    pv = p_[ok_]; o = np.argsort(pv); n_ = len(pv)
    adj = np.minimum.accumulate((pv[o] * n_ / np.arange(1, n_ + 1))[::-1])[::-1]
    q_[np.where(ok_)[0][o]] = np.minimum(adj, 1)
em.insert(em.columns.get_loc("gold_sign_test_p") + 1, "gold_sign_test_q_BH", np.round(q_, 3))
em.to_csv(OUT / "14_endmembers.csv", index=False)
pd.DataFrame(share, columns=[f"E{i + 1}" for i in range(K)]).assign(Hole=d["Hole"], From_m=d["From_m"]) \
  .to_csv(OUT / "14_endmember_weights.csv", index=False)
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 60)
print("\n  Endmembers (share = average weight; stability = same endmember from independent halves of the holes):")
print(em.to_string(index=False))
print("  (stability_cosine_min near 0: that endmember was not recovered in some half-sample fits - e.g. rare rock"
      " absent from that half, or a similar assemblage taking its place)")
print("  share = relative contribution of an endmember to the model of that metre - NOT a % of the rock.")
print("  gold_sign_test_p: among holes whose correlation is outside +-0.1, do positive (or negative) ones dominate")
print("  more than chance? q_BH = the same after Benjamini-Hochberg adjustment for testing every endmember.")
print("  Endmember names are descriptive labels (top minerals), not geological identities. Endmembers matched in")
print("  fewer than 20/20 refits are stable in most refits but less consistently recovered.")
print(f"  Gold tests use {len(usable)} holes with >= 8 gold and >= 20 background metres (within-hole Spearman).")

# ---------------- B) spectral chemistry ----------------
print("\nB) Spectral chemistry (absorption wavelengths, only where the absorption is real):")
both = d[(d["2200D_AlOH"] >= MIN_FEATURE_DEPTH) & (d["2250D_FeOH"] >= MIN_FEATURE_DEPTH)
          & d["2200W_AlOH"].notna() & d["2250W_FeOH"].notna()].copy()
carb = d[(d["11300D_Carbonate"] >= MIN_FEATURE_DEPTH) & d["11300W_Carbonate"].notna()]
print(f"  white mica + chlorite both present: {len(both):,} metres in {both['Hole'].nunique()} holes; "
      f"carbonate present: {len(carb):,} metres in {carb['Hole'].nunique()} holes")
reg = pd.DataFrame({
    "metres_mica_chl": both.groupby("Region").size(),
    "2200W_median": both.groupby("Region")["2200W_AlOH"].median().round(1),
    "2250W_median": both.groupby("Region")["2250W_FeOH"].median().round(1),
    "metres_carbonate": carb.groupby("Region").size(),
    "11300W_median": carb.groupby("Region")["11300W_Carbonate"].median().round(0)})
reg["metres_mica_chl"] = reg["metres_mica_chl"].fillna(0).astype(int)    # counts: 0; wavelengths stay NaN
reg["metres_carbonate"] = reg["metres_carbonate"].fillna(0).astype(int)
reg = reg.sort_values("metres_mica_chl", ascending=False)
print(reg.to_string())
print("  (several regions have only 2-6 holes: these are differences between holes, not proven regional signatures)")
rho = both["2200W_AlOH"].corr(both["2250W_FeOH"], method="spearman")
print(f"  2200W vs 2250W across all metres: Spearman {rho:.2f} "
      f"(positive = Fe/Mg-richer mica goes with Fe-richer chlorite; metres are not independent)")
hm = both.groupby("Hole").agg(Region=("Region", "first"), n=("Hole", "size"),
                              w2200=("2200W_AlOH", "median"), w2250=("2250W_FeOH", "median"))
hm = hm[hm["n"] >= 20]
print(f"  HOLE level (median per hole, holes with >= 20 such metres: {len(hm)}): "
      f"Spearman 2200W vs 2250W {hm['w2200'].corr(hm['w2250'], method='spearman'):.2f}")
byreg = hm.groupby("Region").agg(holes=("n", "size"), w2200_median_of_holes=("w2200", "median"),
                                 w2200_min=("w2200", "min"), w2200_max=("w2200", "max")).round(1)
print("  white-mica position by region, from HOLE medians (spread = range across holes):")
print(byreg.sort_values("holes", ascending=False).to_string())
gw = both[both["Au_PPM"].notna()]
gold_m, bg_m = gw[gw.Au_PPM >= 0.5], gw[gw.Au_PPM < 0.05]
print(f"  pooled, gold vs background metres: 2200W {gold_m['2200W_AlOH'].median():.1f} vs {bg_m['2200W_AlOH'].median():.1f} nm, "
      f"2250W {gold_m['2250W_FeOH'].median():.1f} vs {bg_m['2250W_FeOH'].median():.1f} nm "
      f"({len(gold_m)} / {len(bg_m)} metres; pooled across holes - see 06 for within-hole tests)")

# ---------------- figures ----------------
SURF, INK2, GREY = "#fcfcfb", "#52514e", "#b9b8b3"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#8a5a2b", "#0097a7"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
keep = np.where(Hn.max(axis=0) >= 0.05)[0]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.5 * K + 3), gridspec_kw={"width_ratios": [2.2, 1]})
im = a1.imshow(Hn[:, keep], cmap="Blues", aspect="auto", vmin=0)
a1.set_xticks(range(len(keep)), [short(mins[j]) for j in keep], rotation=60, ha="right", fontsize=8)
a1.set_yticks(range(K), names, fontsize=8)
for i in range(K):
    for j, jj in enumerate(keep):
        if Hn[i, jj] >= 0.1: a1.text(j, i, f"{Hn[i, jj]:.2f}", ha="center", va="center", fontsize=6.5,
                                     color="white" if Hn[i, jj] > 0.4 else "#0b0b0b")
a1.set_title("Mineral make-up of each endmember (S: = SWIR, T: = TIR)", loc="left", fontsize=9)
for s_ in a1.spines.values(): s_.set_visible(False)
a2.barh(range(K), em["stability_cosine_median"], color="#2a78d6", height=0.6)
a2.errorbar(em["stability_cosine_median"], range(K), xerr=[em["stability_cosine_median"] - em["stability_cosine_min"],
            np.zeros(K)], fmt="none", ecolor="#0b0b0b", capsize=2)
a2.set_yticks(range(K), [f"E{i + 1}  (avg. model share {em['mean_share'][i]:.0%})" for i in range(K)], fontsize=8)
a2.invert_yaxis()
a2.set_xlim(0, 1); a2.set_xlabel("Stability: cosine similarity when refit on half the holes\n(bar = median, line = worst of 20 fits)")
a2.set_title("Do the same endmembers appear from independent holes?", loc="left", fontsize=9)
fig.suptitle(f"{K} mineral endmembers unmixed from {len(X):,} metres in {d['Hole'].nunique()} holes", x=0.01, ha="left", weight="bold")
fig.savefig(PLOTS / "14a_endmembers.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.55})
regions = both["Region"].value_counts().index.tolist()
for i, r_ in enumerate(regions):
    q = both[both.Region == r_]
    b1.scatter(q["2250W_FeOH"], q["2200W_AlOH"], s=3, alpha=0.35, color=CAT[i % len(CAT)], linewidths=0,
               label=f"{r_} ({q['Hole'].nunique()} holes)", rasterized=True)
b1.set_xlim(*np.nanpercentile(both["2250W_FeOH"], [0.5, 99.5])); b1.set_ylim(*np.nanpercentile(both["2200W_AlOH"], [0.5, 99.5]))
b1.set_xlabel("Chlorite Fe-OH position, 2250W (nm)  ->  more Fe-rich chlorite")
b1.set_ylabel("White-mica Al-OH position, 2200W (nm)  ->  more phengitic mica")
b1.set_title(f"Mica vs chlorite chemistry, {len(both):,} metres where both are present (Spearman {rho:.2f})", loc="left", fontsize=9)
b1.legend(frameon=False, fontsize=7, markerscale=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
cr = carb["Region"].value_counts()
cr = cr[cr >= 100].index.tolist()
_horiz = {"orientation": "horizontal"} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10) else {"vert": False}
b2.boxplot([carb.loc[carb.Region == r_, "11300W_Carbonate"] for r_ in cr], showfliers=False, widths=0.5, **_horiz)
b2.set_yticks(range(1, len(cr) + 1), [f"{r_} (n={int((carb.Region == r_).sum())})" for r_ in cr], fontsize=7.5)
b2.set_xlabel("TIR carbonate peak position, 11300W (nm)")
b2.set_title("Carbonate chemistry by region (metres with carbonate)", loc="left", fontsize=9)
fig.suptitle("Spectral chemistry in bulk: absorption wavelengths across all holes", x=0.01, ha="left", weight="bold")
fig.savefig(PLOTS / "14b_spectral_chemistry.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nFigures: 14a_endmembers.png, 14b_spectral_chemistry.png")
