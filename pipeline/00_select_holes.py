"""00 - Select the starter holes from the full HyLogger index (run BEFORE downloading mineral logs).

Steps (each count is printed):
  0. Clean the index: valid http(s) mineral-log link; chip/cuttings scans removed (not drill core);
     one scan per hole when a hole was scanned more than once (prefer the scan whose file name
     matches the hole name, e.g. GBD011 over GBD011RE).
  1. Has a WAMEX collar (index -> collar join checked to be many-to-one), > 100 assay rows and
     > 50 geology-log rows.
  2. Yilgarn Craton + commodity exactly 'gold'. Mixed-commodity holes ('gold,nickel', 'copper,gold', ...)
     are deliberately excluded to keep a clean gold-only set; their number is printed.
  3. Hole depth 100-700 m.
  4. Ranked by DATA VOLUME (assay rows + geology rows; NOT a quality score - spectral quality is checked
     after download, in script 02). Ties broken by hole name. Top hole per project until N_HOLES.
     Project key = PROJECT. Holes with a BLANK project (e.g. BWD072) form one explicit group,
     'UNKNOWN_PROJECT', so at most one of them is picked. Not used as a key: the company name (one
     company runs many projects) or the WAMEX report number (one project files a new report every year,
     which would make one project look like many).
  5. Output checks (unique project per hole, names, coordinates) + a sampling-bias table comparing the
     selected holes with the other eligible candidates.
Output: STARTER_HOLES_download_list.csv (repo root). Holes are chosen for having LOTS of data - not random.
"""
from pathlib import Path
import pandas as pd
from config import CSV_DIR, require_data
require_data(need_logs=False)

ROOT = Path(__file__).resolve().parents[1]
N_HOLES, REGION, COMMODITY, DEPTH = 12, "Yilgarn Craton", "gold", (100, 700)

idx = pd.read_csv(CSV_DIR / "HyLogger_Dataset_Index.csv", encoding="utf-8-sig", dtype=str)
col = pd.read_csv(CSV_DIR / "WAMEX_Collar.csv", encoding="utf-8-sig", dtype={"CollarId": str})
assay = pd.read_csv(CSV_DIR / "WAMEX_AssayFlatData.csv", encoding="utf-8-sig", usecols=["CollarId", "Au_PPM"],
                    dtype={"CollarId": str}, na_values=["NULL"])
n_assay = assay["CollarId"].value_counts()
n_geo = pd.read_csv(CSV_DIR / "WAMEX_DHGeology.csv", encoding="utf-8-sig", usecols=["CollarId"], dtype=str)["CollarId"].value_counts()
gold_rate = (pd.to_numeric(assay["Au_PPM"], errors="coerce") >= 0.5).groupby(assay["CollarId"]).mean()

print(f"Holes in the HyLogger index: {len(idx)} rows")
# ---- 0. clean the index ----
url = idx["MINERALLOG"].fillna("").str.strip()
idx = idx[url.str.match(r"(?i)^https?://")].copy()
idx["File"] = idx["MINERALLOG"].str.extract(r"(SM_[^/&?]+\.csv)", expand=False)
idx = idx[~idx["File"].str.contains("_chips", case=False, na=False)]
stem = idx["File"].str.replace("SM_", "", regex=False).str.replace(".csv", "", regex=False)
idx["name_match"] = stem == idx["Hole_name"].str.replace(" ", "", regex=False)
idx = idx.sort_values("name_match", ascending=False).drop_duplicates(["Hole_name", "WAMEX_CollarId"])
print(f"0. Valid mineral-log link, core only, one scan per hole: {len(idx)}")

# ---- 1. join collars + data volume ----
d = idx.merge(col, left_on="WAMEX_CollarId", right_on="CollarId", how="inner", validate="many_to_one")
d["n_assay"] = d["CollarId"].map(n_assay).fillna(0).astype(int)
d["n_geo"] = d["CollarId"].map(n_geo).fillna(0).astype(int)
d["gold_rate"] = d["CollarId"].map(gold_rate)
d = d[(d["n_assay"] > 100) & (d["n_geo"] > 50)]
print(f"1. With collar + > 100 assay rows + > 50 geology rows: {len(d)}")

# ---- 2. region + commodity ----
reg = d[d["TectonicRegion"] == REGION]
mixed = reg["Commodity_vocab"].fillna("").str.contains(COMMODITY) & (reg["Commodity_vocab"] != COMMODITY)
d = reg[reg["Commodity_vocab"] == COMMODITY]
print(f"2. {REGION}, commodity exactly '{COMMODITY}': {len(d)}  (excluded mixed-commodity gold holes: {mixed.sum()})")

# ---- 3. depth ----
d = d.assign(MaxDepth=pd.to_numeric(d["MaxDepth"], errors="coerce"))
d = d[d["MaxDepth"].between(*DEPTH)]
print(f"3. Depth {DEPTH[0]}-{DEPTH[1]} m: {len(d)}")

# ---- 4. rank + one per project ----
d["Project_key"] = d["PROJECT"].fillna("UNKNOWN_PROJECT")
print(f"   Candidates with a blank project (one shared group, at most 1 picked): {d['PROJECT'].isna().sum()}")
d = d.dropna(subset=["Project_key", "Hole_name"])
d["score"] = d["n_assay"] + d["n_geo"]
d = d.sort_values(["score", "Hole_name"], ascending=[False, True])
pick = d.drop_duplicates("Project_key").head(N_HOLES)
print(f"4. Top {len(pick)}, one per project (ranked by data volume):\n",
      pick[["Hole_name", "Project_key", "Company", "MaxDepth", "n_assay", "n_geo"]].to_string(index=False))

# ---- 5. output checks + sampling bias ----
pick = pick.assign(**{c: pd.to_numeric(pick[c], errors="coerce") for c in ["Shapefile_latitude", "Shapefile_longitude"]})
assert pick["Project_key"].is_unique, "two holes from the same project"
assert pick["Hole_name"].notna().all() and pick["File"].notna().all(), "missing hole name or file"
assert pick[["Shapefile_latitude", "Shapefile_longitude"]].notna().all().all(), "missing coordinates"
assert pick["Shapefile_latitude"].between(-36, -13).all() and pick["Shapefile_longitude"].between(112, 130).all(), "coordinates outside WA"
others = d[~d.index.isin(pick.index)]
bias = pd.DataFrame({"selected": pick[["MaxDepth", "n_assay", "n_geo", "gold_rate"]].median(),
                     "other_candidates": others[["MaxDepth", "n_assay", "n_geo", "gold_rate"]].median()}).round(3)
bias.index = ["Median depth (m)", "Median assay rows", "Median geology rows", "Median share of assays with Au >= 0.5 ppm"]
print(f"\n5. Checks passed. Sampling bias - selected ({len(pick)}) vs other eligible candidates ({len(others)}):\n", bias.to_string())
bias.to_csv(ROOT / "outputs" / "00_selection_bias.csv") if (ROOT / "outputs").exists() else None

out = pick.rename(columns={"PROJECT": "Project", "MaxDepth": "MaxDepth_m", "n_assay": "Assay_rows", "n_geo": "Geology_log_rows",
                           "Shapefile_latitude": "Lat", "Shapefile_longitude": "Lon", "MINERALLOG": "MineralLog_CSV_URL",
                           "SUM_HISTO": "Summary_Histogram_PDF", "MOSAIC": "Core_Mosaic_viewer"})
out[["Hole_name", "NVCL_HoleName", "Project", "Project_key", "Company", "MaxDepth_m", "Assay_rows", "Geology_log_rows", "Lat", "Lon",
     "MineralLog_CSV_URL", "Summary_Histogram_PDF", "Core_Mosaic_viewer"]].to_csv(ROOT / "STARTER_HOLES_download_list.csv", index=False)
print("\nSaved STARTER_HOLES_download_list.csv")
