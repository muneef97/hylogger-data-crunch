"""07 - Flag mineralogically unusual intervals with Isolation Forest.

Same 23 features and scaling as script 05. MRDH400 is NOT scored (no TIR data).
  * Anomaly score: higher = more unusual. Top 3% of intervals flagged as anomalies.
  * For each anomaly, the features that deviate most (largest |z-score|) explain WHY it is unusual.
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

fit_mask = (d["Hole"] != "MRDH400").values
scaler = StandardScaler().fit(X[fit_mask])
Z = pd.DataFrame(scaler.transform(X), columns=X.columns, index=d.index)

iso = IsolationForest(n_estimators=500, contamination=0.03, random_state=0).fit(Z[fit_mask])
d["Anomaly_score"] = -iso.score_samples(Z)
d["Is_anomaly"] = iso.predict(Z) == -1
# MRDH400 has no TIR data -> its TIR zeros would look 'unusual' for the wrong reason; not scored.
d.loc[~fit_mask, ["Anomaly_score", "Is_anomaly"]] = [np.nan, False]

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
g = d[d["Au_PPM"].notna() & d["Anomaly_score"].notna()]   # scored intervals only
print(f"\nShare with Au >= 0.5 ppm: anomalies {(g[g.Is_anomaly].Au_PPM >= 0.5).mean():.1%} "
      f"vs normal {(g[~g.Is_anomaly].Au_PPM >= 0.5).mean():.1%}")
print(f"Share flagged low-confidence (QC): anomalies {d[d.Is_anomaly].QC_low_confidence.mean():.1%} "
      f"vs normal {d[~d.Is_anomaly].QC_low_confidence.mean():.1%}")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
print(f"\nAnomalous zones: {len(zones)}. Top 10:\n", zones.head(10).to_string(index=False))
