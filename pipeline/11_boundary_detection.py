"""11 - Automatic boundary detection: split each hole into depth zones where the mineralogy changes.

Follows GSWA's 'Boundary detection' idea ("Can we automatically map divisions in stratigraphy?").
Method (per hole):
  * Features = SWIR + TIR mineral proportions (standardised across all holes).
  * Low-confidence intervals are skipped; ANY depth gap (unscanned or skipped metres) starts a new
    run, so PELT never treats metres either side of a gap as neighbours.
  * Change-point detection with PELT (ruptures library, 'l2' cost = changes in the average mix),
    minimum zone length 5 m, penalty = PEN_FACTOR x log(n) x number of features (a BIC-style penalty).
Validation (independent of the method): a 'logged contact' is where two DEPTH-CONTIGUOUS metres both
have a company lithology code and the codes differ (from script 03, all metres). We measure the share
of our boundaries within +-3 m of a logged contact. Fair baseline: the same number of random boundaries
drawn only from ELIGIBLE positions (positions where PELT could have put a boundary AND the log has a
code), at least 5 m apart - 5,000 draws per hole, each draw keeping that hole's boundary count; the
pooled statistic sums the holes draw by draw. p = (1 + draws doing as well) / (1 + 5,000).
Only our boundaries at eligible positions are scored.
Codes are only used to find WHERE the log changes; their meaning is not interpreted.
Outputs: outputs/11_zones.csv, outputs/11_boundaries.csv, outputs/plots/11_zones_<HOLE>.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs"
PLOTS = OUT / "plots"; PLOTS.mkdir(exist_ok=True)
MIN_ZONE_M, PEN_FACTOR, TOL_M, N_RAND = 5, 1.0, 3, 5000
rng = np.random.default_rng(0)
for old in PLOTS.glob("11_zones_*.png"):        # no stale figures from earlier runs
    try: old.unlink()
    except OSError: print("Could not delete old figure:", old.name)

# logged contacts from ALL metres (script 03): contiguous metres, both coded, codes differ
logs = pd.read_csv(OUT / "03_merged_intervals.csv", low_memory=False).sort_values(["Hole", "From_m"])
def logged_contacts(hole):
    a = logs[logs["Hole"] == hole]
    prev_code, prev_to = a["Lith_code"].shift(), a["To_m"].shift()
    ok = a["Lith_code"].notna() & prev_code.notna() & (a["From_m"] == prev_to) & (a["Lith_code"] != prev_code)
    return a.loc[ok, "From_m"].to_numpy()
coded = set(zip(logs.loc[logs["Lith_code"].notna(), "Hole"], logs.loc[logs["Lith_code"].notna(), "From_m"]))

def random_spaced(pos, k):
    """k distinct eligible positions, all at least MIN_ZONE_M apart (rejection sampling)."""
    for _ in range(1000):
        x = np.sort(rng.choice(pos, k, replace=False))
        if k < 2 or np.diff(x).min() >= MIN_ZONE_M: return x
    raise RuntimeError("could not place random boundaries with the spacing rule")

d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False).sort_values(["Hole", "From_m"])
lith = pd.read_csv(OUT / "03_merged_intervals.csv", low_memory=False)[["Hole", "From_m", "Lith_code"]]
d = d.merge(lith, on=["Hole", "From_m"], how="left", suffixes=("", "_l"))
if "Lith_code_l" in d: d["Lith_code"] = d["Lith_code"].fillna(d.pop("Lith_code_l"))

feats = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].mean() > 0.005]
Z = (d[feats] - d[feats].mean()) / d[feats].std()

zones, bounds, checks = [], [], []
pooled_obs, pooled_rand = [], []
for hole, h in d.groupby("Hole"):
    h = h[~h["QC_low_confidence"]]
    use = feats
    # split into continuous runs (any gap starts a new run)
    run_id = (h["From_m"] - h["To_m"].shift() > 0).cumsum()
    hole_b, eligible = [], []
    for _, r in h.groupby(run_id):
        X = Z.loc[r.index, use].to_numpy()
        if len(r) >= 2 * MIN_ZONE_M:
            algo = rpt.Pelt(model="l2", min_size=MIN_ZONE_M).fit(X)
            cps = algo.predict(pen=PEN_FACTOR * np.log(len(r)) * len(use))[:-1]
        else:
            cps = []
        edges = [0] + cps + [len(r)]
        for a, b in zip(edges[:-1], edges[1:]):
            seg = r.iloc[a:b]
            zones.append({"Hole": hole, "Zone_from_m": seg["From_m"].min(), "Zone_to_m": seg["To_m"].max(),
                          "Length_m": round(seg["To_m"].max() - seg["From_m"].min(), 1),
                          "Main_cluster": int(seg["Cluster"].mode().iloc[0]),
                          "Top_minerals": ", ".join(f"{c} {v:.2f}" for c, v in seg[use].mean().sort_values(ascending=False).head(3).items()),
                          "Mean_Au_ppm": round(seg["Au_PPM"].mean(), 3) if seg["Au_PPM"].notna().any() else np.nan})
        hole_b += [r["From_m"].iloc[c] for c in cps]
        if len(r) >= 2 * MIN_ZONE_M:                 # positions PELT could choose in this run
            eligible += [x for x in r["From_m"].iloc[MIN_ZONE_M:len(r) - MIN_ZONE_M + 1] if (hole, x) in coded]
    bounds += [{"Hole": hole, "Boundary_m": b} for b in hole_b]

    # ---- validation against logged contacts ----
    contacts = logged_contacts(hole)
    eligible = np.array(sorted(set(eligible)))
    inside = [b for b in hole_b if b in set(eligible)]
    if len(contacts) >= 3 and len(inside) >= 3:
        hit = lambda bs: int(sum(np.min(np.abs(contacts - b)) <= TOL_M for b in bs))
        rand = np.array([hit(random_spaced(eligible, len(inside))) for _ in range(N_RAND)])
        obs = hit(inside)
        pooled_obs.append(obs); pooled_rand.append(rand)
        checks.append({"Hole": hole, "our_boundaries": len(inside), "eligible_positions": len(eligible),
                       "logged_contacts": len(contacts), "ours_near_contact": round(obs / len(inside), 2),
                       "random_near_contact": round(rand.mean() / len(inside), 2),
                       "p_value": round((1 + (rand >= obs).sum()) / (1 + N_RAND), 4)})

zones = pd.DataFrame(zones); bounds = pd.DataFrame(bounds); checks = pd.DataFrame(checks)
zones.to_csv(OUT / "11_zones.csv", index=False); bounds.to_csv(OUT / "11_boundaries.csv", index=False)
checks.to_csv(OUT / "11_boundary_validation.csv", index=False)

print(f"Boundaries found by PELT (mineral change points): {len(bounds)} across {bounds['Hole'].nunique()} holes")
print(f"Zones: {len(zones)} (includes short pieces between scanning gaps, where no boundary can be tested; "
      f"median zone length {zones['Length_m'].median():.0f} m)")
print(zones.groupby("Hole").agg(zones=("Length_m", "size"), median_len_m=("Length_m", "median")).to_string())
print(f"\nValidation vs logged lithology contacts (boundary within +-{TOL_M} m of a contact):")
print(checks.to_string(index=False))
if len(checks):
    w = checks["our_boundaries"]
    print(f"Validated subset: {w.sum()} of the {len(bounds)} PELT boundaries, in {len(checks)} holes "
          f"(only boundaries at positions with a lithology code, in holes with >= 3 contacts and >= 3 boundaries)")
    print(f"Weighted: ours {np.average(checks['ours_near_contact'], weights=w):.0%} vs random "
          f"{np.average(checks['random_near_contact'], weights=w):.0%}")
    tot_obs, tot_rand = sum(pooled_obs), np.sum(pooled_rand, axis=0)
    print(f"Pooled test across holes: {tot_obs:.0f} of {w.sum()} boundaries near a contact; random average "
          f"{tot_rand.mean():.1f}; p = {(1 + (tot_rand >= tot_obs).sum()) / (1 + N_RAND):.4f} "
          f"({N_RAND:,} random draws from eligible, logged positions, >= {MIN_ZONE_M} m apart)")

# ---- figures: strip of mineral mix with zone boundaries + logged contacts ----
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
GROUPS = {"White mica": "SWIR_White-Mica", "Chlorite": "SWIR_Chlorite", "Amphibole": "SWIR_Amphibole",
          "Dark mica": "SWIR_Dark-Mica", "Carbonate": "SWIR_Carbonate", "Epidote": "SWIR_Epidote"}
show = checks.sort_values("our_boundaries", ascending=False)["Hole"].head(4).tolist() or zones["Hole"].unique()[:4].tolist()
for hole in show:
    h = d[(d["Hole"] == hole) & ~d["QC_low_confidence"]]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    bottom = np.zeros(len(h))
    for (name, col), c in zip(GROUPS.items(), CAT):
        ax.bar(h["From_m"], h[col], width=h["To_m"] - h["From_m"], bottom=bottom, color=c, align="edge", linewidth=0, label=name)
        bottom += h[col].to_numpy()
    for b in bounds.loc[bounds["Hole"] == hole, "Boundary_m"]:
        ax.axvline(b, color="#0b0b0b", lw=1.6)
    for c in logged_contacts(hole):
        ax.plot([c, c], [1.02, 1.1], color="#e34948", lw=1.4, clip_on=False)
    ax.set_ylim(0, 1); ax.set_xlabel("Depth (m)"); ax.set_ylabel("SWIR mineral share")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{hole}: automatic zone boundaries (black lines) vs logged lithology contacts (red ticks above)",
                 loc="left", weight="bold", fontsize=10, pad=16)
    ax.legend(ncol=6, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(PLOTS / f"11_zones_{hole}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nFigures:", [f"11_zones_{h}.png" for h in show])
