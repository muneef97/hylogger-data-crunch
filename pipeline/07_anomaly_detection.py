"""07 - Flag mineralogically unusual intervals with Isolation Forest.

Same 23 features and scaling as script 05 (input has already passed the QC filter).
  * Anomaly score: higher = more unusual. The top 3% are flagged: 3% is a setting we chose
    (contamination=0.03), not something the model discovered.
  * For each anomaly we list its largest standardised feature deviations (|z-score|). These describe
    what is unusual about the metre; they are not the model's internal reasons.
  * Gold check is done pooled AND controlling for each hole's own gold rate (as in script 06),
    with a 90% interval from a 10 m block bootstrap, and repeated at other cut-offs (1-10%).
  * Consecutive anomalous intervals (gap <= 2 m) are grouped into 'anomalous zones'.
Outputs: outputs/07_intervals_with_anomaly.csv, outputs/07_anomalous_zones.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

OUT = Path(__file__).resolve().parents[1] / "outputs"
d = pd.read_csv(OUT / "05_intervals_with_clusters.csv", low_memory=False)

minerals = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].mean() > 0.005]
indices = [c for c in ["2200W_AlOH", "2250W_FeOH", "900D_Fe", "1900D_Water", "8625D_Quartz", "VIS-SWIR_Albedo"] if c in d]
X = d[minerals + indices].copy()
X[indices] = X[indices].fillna(X[indices].median())

Z = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=d.index)

iso = IsolationForest(n_estimators=500, contamination=0.03, random_state=0).fit(Z)
d["Anomaly_score"] = -iso.score_samples(Z)
d["Is_anomaly"] = iso.predict(Z) == -1

def why(i, n=3):
    z = Z.loc[i].abs().sort_values(ascending=False).head(n)
    return ", ".join(f"{c} {'high' if Z.loc[i, c] > 0 else 'low'} (z={Z.loc[i, c]:+.1f})" for c in z.index)
d["Anomaly_reason"] = ""
d.loc[d["Is_anomaly"], "Anomaly_reason"] = [why(i) for i in d.index[d["Is_anomaly"]]]
d.to_csv(OUT / "07_intervals_with_anomaly.csv", index=False)

# group into zones
a = d[d["Is_anomaly"]].sort_values(["Hole", "From_m"])
zone_id = (a["Hole"] != a["Hole"].shift()) | (a["From_m"] - a["To_m"].shift() > 2)
a = a.assign(Zone=zone_id.cumsum())
zones = a.groupby("Zone").agg(Hole=("Hole", "first"), From_m=("From_m", "min"), To_m=("To_m", "max"),
                              n_intervals=("Hole", "size"), max_score=("Anomaly_score", "max"),
                              max_Au_ppm=("Au_PPM", "max"),
                              reason=("Anomaly_reason", lambda s: s.iloc[np.argmax(a.loc[s.index, 'Anomaly_score'])]))
zones = zones.sort_values("max_score", ascending=False).round(3)
zones.to_csv(OUT / "07_anomalous_zones.csv", index=False)

print(f"Anomalous intervals: {d['Is_anomaly'].sum()} of {len(d)} ({d['Is_anomaly'].mean():.1%})")
print("\nAnomalies per hole:\n", d.groupby("Hole")["Is_anomaly"].sum().sort_values(ascending=False).to_string())
g = d[d["Au_PPM"].notna()].copy()                          # assayed metres only
g["gold"] = g["Au_PPM"] >= 0.5
print(f"\nAssayed anomalies: {g.Is_anomaly.sum()} of {d.Is_anomaly.sum()}")
print(f"Share with Au >= 0.5 ppm (pooled): anomalies {g[g.Is_anomaly].gold.mean():.1%} "
      f"vs normal {g[~g.Is_anomaly].gold.mean():.1%}")

# Controlling for hole: expected gold among anomalies if each hole's own gold rate applied
def anomaly_lift(s):
    exp = s["Hole"].map(s.groupby("Hole")["gold"].mean())
    a = s["Is_anomaly"]
    return s.loc[a, "gold"].sum() / exp[a].sum() if exp[a].sum() > 0 else np.nan
g["block"] = g["Hole"] + "_" + (g["Mid_m"] // 10).astype(int).astype(str)
rng = np.random.default_rng(0)
blocks = {h: x["block"].unique() for h, x in g.groupby("Hole")}
by_block = {b: x for b, x in g.groupby("block")}
boot = [anomaly_lift(pd.concat([by_block[b] for bl in blocks.values() for b in rng.choice(bl, len(bl))]))
        for _ in range(500)]
exp_all = g["Hole"].map(g.groupby("Hole")["gold"].mean())
print(f"Controlling for hole: {g[g.Is_anomaly].gold.sum()} gold anomalies observed vs "
      f"{exp_all[g.Is_anomaly].sum():.1f} expected -> lift {anomaly_lift(g):.2f} "
      f"(90% interval {np.nanquantile(boot, 0.05):.2f}-{np.nanquantile(boot, 0.95):.2f})")
# Sensitivity: the 3% cut-off is our choice -> show the hole-adjusted lift at other cut-offs
g["score"] = d.loc[g.index, "Anomaly_score"]
exp_g = g["Hole"].map(g.groupby("Hole")["gold"].mean())
print("Lift at other cut-offs (top % most unusual; assayed / gold / expected / lift):")
for q in [0.01, 0.02, 0.03, 0.05, 0.10]:
    a = g["score"] >= d["Anomaly_score"].quantile(1 - q)
    print(f"  top {q:4.0%}: {a.sum():4d} {g[a].gold.sum():4d} {exp_g[a].sum():6.1f}  {g[a].gold.sum() / exp_g[a].sum():.2f}")
print("Per hole (assayed anomalies / gold among them / hole gold rate):")
print(g.drop(columns="Hole").groupby(g["Hole"]).apply(lambda x: pd.Series({
    "anomalies": int(x.Is_anomaly.sum()), "gold_anom": int(x[x.Is_anomaly].gold.sum()),
    "hole_gold_rate": round(x.gold.mean(), 3)})).query("anomalies > 0").to_string())
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(f"\nAnomalous zones: {len(zones)}. Top 10:\n", zones.head(10).to_string(index=False))
