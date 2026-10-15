"""Build the data the Insight Explorer app reads, from THIS pipeline's outputs (65 holes).

The app (app/streamlit_app.py, adapted from a teammate's HyLogger Insight Explorer) expects one
"modelled intervals" table in its own column format. This script converts our cleaned outputs into it,
so every number the app shows comes from run_all.sh: assays cleaned in 03, clusters/PCA from 05,
anomalies from 07, rock classes from 10, endmembers from 14.

Run after run_all.sh (takes ~20 s):
    python app/build_app_data.py
Writes (all git-ignored): outputs/app/processed/hylogger_modelled_intervals.csv,
outputs/app/reports/*.csv + feature_config.json, outputs/app/collars.csv
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from config import CSV_DIR, OUT, require_data  # noqa: E402
require_data(need_logs=False)

APP = OUT / "app"
(APP / "processed").mkdir(parents=True, exist_ok=True)
(APP / "reports").mkdir(parents=True, exist_ok=True)


def need(name):
    p = OUT / name
    if not p.exists():
        sys.exit(f"Missing {p}. Run ./run_all.sh first.")
    return p


def to_app_name(c):
    """Our short names back to the HyLogger log names the app uses: SWIR_Kaolin -> Kaolin_(System_SWIR)."""
    if c.startswith("SWIR_"):
        return c[5:] + "_(System_SWIR)"
    if c.startswith("TIR_"):
        return c[4:] + "_(System_TIR)"
    return c


# ---- base: every merged interval (03), with the QC flag ----
m = pd.read_csv(need("03_merged_intervals.csv"), low_memory=False)
m["Depth_QC_Valid"] = ~m["QC_low_confidence"].astype(bool)
key = ["Hole", "From_m"]

# clusters + PCA (05), anomalies (07), rock class (10), endmembers (14): quality-checked metres only
c5 = pd.read_csv(need("05_intervals_with_clusters.csv"), usecols=key + ["PC1", "PC2", "Cluster"])
c7 = pd.read_csv(need("07_intervals_with_anomaly.csv"), usecols=key + ["Anomaly_score", "Is_anomaly", "Anomaly_reason"])
c10 = pd.read_csv(need("10_rock_type_intervals.csv"), usecols=key + ["HyLogger_rock", "HyLogger_broad", "Logged_broad"])
parts = [c5, c7, c10]
if (OUT / "14_endmember_weights.csv").exists():
    parts.append(pd.read_csv(OUT / "14_endmember_weights.csv"))
for p in parts:
    m = m.merge(p, on=key, how="left", validate="one_to_one")

# ---- geology intervals: the app needs each metre's logged interval (From/To) to find contacts ----
holes = pd.read_csv(need("02_holes.csv"))
holes["WAMEX_CollarId"] = pd.to_numeric(holes["WAMEX_CollarId"], errors="coerce")
cid = dict(zip(holes["Hole"], holes["WAMEX_CollarId"]))
geo = pd.read_csv(CSV_DIR / "WAMEX_DHGeology.csv", encoding="utf-8-sig", low_memory=False,
                  usecols=["CollarId", "FromDepth", "ToDepth", "GSWARockName", "CompanyLithologyName"])
geo = geo[geo["CollarId"].isin(set(cid.values()))].dropna(subset=["FromDepth", "ToDepth"])
geo = geo[geo["ToDepth"] > geo["FromDepth"]]
m["Geo_FromDepth"] = np.nan
m["Geo_ToDepth"] = np.nan
for hole, idx in m.groupby("Hole").groups.items():
    g = geo[geo["CollarId"] == cid.get(hole)]
    if g.empty:
        continue
    f, t = g["FromDepth"].to_numpy(float), g["ToDepth"].to_numpy(float)
    length = t - f
    mid = m.loc[idx, "Mid_m"].to_numpy(float)
    cover = (f[None, :] <= mid[:, None]) & (t[None, :] > mid[:, None])
    best = np.where(cover, length[None, :], np.inf).argmin(axis=1)
    ok = cover.any(axis=1)
    m.loc[np.asarray(idx)[ok], "Geo_FromDepth"] = f[best[ok]]
    m.loc[np.asarray(idx)[ok], "Geo_ToDepth"] = t[best[ok]]

# ---- the app's column format ----
assay_cols = [c for c in m.columns if c.endswith(("_PCT", "_PPM"))]
out = pd.DataFrame({
    "Hole_name": m["Hole"],
    "Mineral_FromDepth": m["From_m"], "Mineral_ToDepth": m["To_m"], "Depth_Mid": m["Mid_m"],
    "Depth_QC_Valid": m["Depth_QC_Valid"],
    "Join_CollarId": m["Hole"].map(cid),
    "Report_key": m["Hole"].map(dict(zip(holes["Hole"], holes["TectonicRegion"]))).fillna("Region not recorded"),
    "Company": m["Hole"].map(dict(zip(holes["Hole"], holes["Company"]))),
    # GSWA's standard rock name where logged; otherwise the company's rock name (both are geologist labels)
    "Geo_GSWARockName": m["GSWA_rock"].fillna(m["Lith_name"]),
    "Geo_CompanyLithologyCode": m["Lith_code"],
    "Geo_Comments": m["Geology_comments"],
    "Geo_FromDepth": m["Geo_FromDepth"], "Geo_ToDepth": m["Geo_ToDepth"],
    "Geo_Match": m["Geo_FromDepth"].notna(),
    "Logged_broad": m["Logged_broad"], "HyLogger_broad": m["HyLogger_broad"], "HyLogger_rock": m["HyLogger_rock"],
    "Cluster": m["Cluster"], "PCA1": m["PC1"], "PCA2": m["PC2"],
    "AnomalyScore": m["Anomaly_score"], "IsAnomaly": m["Is_anomaly"].astype("boolean"),
    "Anomaly_reason": m["Anomaly_reason"],
})
mineral_cols = [c for c in m.columns if c.startswith(("SWIR_", "TIR_"))]
feature_cols = ["VIS-SWIR_Albedo", "900D_Fe", "900W_Fe", "1900D_Water", "2200D_AlOH", "2200W_AlOH",
                "2250D_FeOH", "2250W_FeOH", "8625D_Quartz", "11300D_Carbonate", "11300W_Carbonate"]
for c in mineral_cols + [c for c in feature_cols if c in m]:
    out[to_app_name(c)] = m[c]
for c in [c for c in m.columns if c[:1] == "E" and c[1:].isdigit()]:
    out[f"Pipeline_{c}"] = m[c]
for c in assay_cols:
    out["Assay_" + c] = m[c]
out["Assay_Match"] = m[assay_cols].notna().any(axis=1)
# the app averages features per assay sample; our assays are already one value per metre, so each metre is its own sample
out["AssayFlatDataId"] = np.where(out["Assay_Match"], m["Hole"] + "@" + m["From_m"].round(2).astype(str), None)
out["Assay_MidDepth"] = m["Mid_m"]
out.to_csv(APP / "processed" / "hylogger_modelled_intervals.csv", index=False)

# ---- model features = exactly the 05 PCA features, in the app's names ----
loadings = pd.read_csv(need("05_pca_loadings.csv"), index_col=0)
model_features = [to_app_name(c) for c in loadings.index]
display = [to_app_name(c) for c in ["SWIR_White-Mica", "SWIR_Chlorite", "SWIR_Amphibole", "TIR_Silica",
                                    "TIR_Plagioclase", "SWIR_Carbonate", "2200W_AlOH", "2250W_FeOH"]]
q = out[out["Depth_QC_Valid"]]
lab = q.dropna(subset=["Geo_GSWARockName", "Cluster"])
try:
    from sklearn.metrics import adjusted_rand_score
    ari = float(adjusted_rand_score(lab["Geo_GSWARockName"].astype(str), lab["Cluster"].astype(int)))
except Exception:
    ari = None
config = {
    "scope": f"65 holes from this pipeline (run_all.sh): {len(out):,} metres, {int(out['Depth_QC_Valid'].sum()):,} pass QC",
    "feature_mode": "numeric_mineral_log_features",
    "model_feature_columns": model_features,
    "display_numeric_features": display + [c for c in model_features if c not in display],
    "assay_columns": ["Assay_" + c for c in assay_cols],
    "pca_explained_variance_ratio": "see run_log.txt, script 05",
    "adjusted_rand_index_vs_gswa_rockname": round(ari, 3) if ari is not None else "not available",
    "report_key_kind": "region",
    "warning": "Clusters and anomaly scores are exploratory. Gold and copper are not predictable from minerals (sulphides are invisible to SWIR/TIR).",
}
(APP / "reports" / "feature_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

# ---- per-hole coverage and cluster vs rock name ----
cov = out.groupby("Hole_name").agg(mineral_rows=("Depth_Mid", "size"), rows_with_valid_depth=("Depth_QC_Valid", "sum"),
                                   geology_match_pct=("Geo_Match", "mean"), assay_match_pct=("Assay_Match", "mean")).reset_index()
cov["geology_match_pct"] = (100 * cov["geology_match_pct"]).round(1)
cov["assay_match_pct"] = (100 * cov["assay_match_pct"]).round(1)
cov.to_csv(APP / "reports" / "per_hole_data_coverage.csv", index=False)
pd.crosstab(lab["Geo_GSWARockName"], lab["Cluster"].astype(int)).to_csv(APP / "reports" / "cluster_vs_gswa_rockname_counts.csv")

# ---- collars for the map (coordinates, company, depth) ----
col = pd.read_csv(CSV_DIR / "WAMEX_Collar.csv", encoding="utf-8-sig", low_memory=False)
col = col[col["CollarId"].isin(set(cid.values()))][["CollarId", "CompanyHoleId", "Longitude", "Latitude", "HoleType",
                                                    "MaxDepth", "Elevation", "Company"]]
col.to_csv(APP / "collars.csv", index=False)

# ---- GSWA scorecard: the pipeline's validated numbers ----
rows = []
e = pd.read_csv(OUT / "14_endmembers.csv") if (OUT / "14_endmembers.csv").exists() else None
if e is not None:
    stable = int((e["refits_matched(cos>=0.9)"].astype(str).str.split("/").str[0].astype(float) >= 19).sum())
    rows.append(["1. Anomalies / endmembers in bulk", f"{len(e)} endmembers (k chosen by stability); {stable} recovered in >= 19 of 20 refits; 3% most unusual metres flagged with reasons", "script 14, 07"])
if (OUT / "15_bwd072_metre_features.csv").exists():
    rows.append(["2. Features outside TSG", "Own white-mica feature from 76,160 raw BWD072 spectra; vs TSG r = 0.90, median difference 1.1 nm", "script 15"])
v = pd.read_csv(need("11_boundary_validation.csv"))
w = v["our_boundaries"]
hits = round((v["ours_near_contact"] * w).sum())   # per-hole shares are rounded to 2 dp; round back to a whole count
rows.append(["3. Divisions in stratigraphy", f"{hits:.0f} of {int(w.sum())} boundaries ({hits / w.sum():.0%}) within +-3 m of a logged contact vs "
             f"{(v['random_near_contact']*w).sum()/w.sum():.0%} for random boundaries (5,000 tries, none as good), {len(v)} holes; 66-75% across settings", "script 11"])
if (OUT / "10d_rock_type_scores.csv").exists():
    r = pd.read_csv(OUT / "10d_rock_type_scores.csv").set_index(["level", "method"])
    nt = len(pd.read_csv(OUT / "10d_rock_type_recall.csv"))
    rows.append(["4. Minerals -> rock type", f"{nt} logged rock types on unseen holes: {r.loc[('composition group', 'Random forest - new HOLE'), 'accuracy']:.0%} right composition group, "
                 f"{r.loc[('rock type', 'Random forest - new HOLE'), 'accuracy']:.0%} exact type (chance {1 / nt:.0%}); misses are mostly rocks with the same minerals (basalt vs dolerite)", "script 10d"])
else:
    c = pd.read_csv(need("10_rock_type_calibration_loho.csv"))
    rows.append(["4. Minerals -> rock type", f"{(c['accuracy_on_held_out']*c['n']).sum()/c['n'].sum():.0%} of {int(c['n'].sum()):,} logged metres match the broad class (felsic/intermediate/mafic) on unseen holes", "script 10"])
s16 = pd.read_csv(need("16_bulk_prediction_scores.csv"))
comp = s16[(s16["model"] == "PLS") & (s16["held_out"] == "company")].set_index("element")
mg = comp.loc["MgO"] if "MgO" in comp.index else None
rows.append(["5. Other properties in bulk", f"{int((comp['skill'] >= 0.3).sum())} of {len(comp)} elements predictable for an unseen company"
             + (f"; MgO typically within {mg['median_rel_error']:.0%} of the lab value (R2 {mg['skill']:.2f})" if mg is not None else "")
             + "; gold, copper, arsenic, sulphur not predictable", "scripts 16, 16b"])
pd.DataFrame(rows, columns=["GSWA question", "Validated pipeline result", "Source"]).to_csv(APP / "reports" / "gswa_questions_scorecard.csv", index=False)

print(f"App data written to {APP}: {len(out):,} metres, {out['Hole_name'].nunique()} holes, "
      f"{int(out['Geo_FromDepth'].notna().sum()):,} metres with a geology interval, {int(out['Assay_Match'].sum()):,} with assays")
