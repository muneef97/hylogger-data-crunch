"""00 - Select the starter holes from the full HyLogger index (run BEFORE downloading mineral logs).

Filters the ~2,000-hole DASC HyLogger index, step by step:
  1. Has a mineral log link, a WAMEX collar, > 100 assay rows and > 50 geology-log rows
  2. Yilgarn Craton + commodity 'gold' (the largest consistent group)
  3. Hole depth 100-700 m
  4. Ranked by (assay rows + geology rows); the top hole from each project, until N_HOLES are chosen
Output: STARTER_HOLES_download_list.csv (in the repo root) with download links for each hole.
Note: holes chosen for having LOTS of data -> not a random sample.
"""
from pathlib import Path
import pandas as pd
from config import CSV_DIR, LOGS_DIR, require_data
require_data(need_logs=False)

ROOT = Path(__file__).resolve().parents[1]
CSV = CSV_DIR
N_HOLES, REGION, COMMODITY, DEPTH = 12, "Yilgarn Craton", "gold", (100, 700)

idx = pd.read_csv(CSV / "HyLogger_Dataset_Index.csv", encoding="utf-8-sig", dtype=str)
col = pd.read_csv(CSV / "WAMEX_Collar.csv", encoding="utf-8-sig", dtype={"CollarId": str})
n_assay = pd.read_csv(CSV / "WAMEX_AssayFlatData.csv", encoding="utf-8-sig", usecols=["CollarId"], dtype=str)["CollarId"].value_counts()
n_geo = pd.read_csv(CSV / "WAMEX_DHGeology.csv", encoding="utf-8-sig", usecols=["CollarId"], dtype=str)["CollarId"].value_counts()

print(f"Holes in the HyLogger index: {len(idx)}")
d = idx.merge(col, left_on="WAMEX_CollarId", right_on="CollarId", how="inner")
d["n_assay"] = d["CollarId"].map(n_assay).fillna(0).astype(int)
d["n_geo"] = d["CollarId"].map(n_geo).fillna(0).astype(int)
d = d[d["MINERALLOG"].fillna("").str.startswith("http") & (d["n_assay"] > 100) & (d["n_geo"] > 50)]
print(f"1. With mineral log + > 100 assay rows + > 50 geology rows: {len(d)}")
d = d[(d["TectonicRegion"] == REGION) & (d["Commodity_vocab"] == COMMODITY)]
print(f"2. {REGION}, {COMMODITY}: {len(d)}")
d["MaxDepth"] = pd.to_numeric(d["MaxDepth"], errors="coerce")
d = d[d["MaxDepth"].between(*DEPTH)]
print(f"3. Depth {DEPTH[0]}-{DEPTH[1]} m: {len(d)}")
d = d.assign(score=d["n_assay"] + d["n_geo"]).sort_values("score", ascending=False)
pick = d.drop_duplicates("PROJECT").head(N_HOLES)
print(f"4. Top {N_HOLES}, one per project:\n", pick[["Hole_name", "PROJECT", "Company", "MaxDepth", "n_assay", "n_geo"]].to_string(index=False))

out = pick.rename(columns={"Hole_name": "Hole_name", "PROJECT": "Project", "MaxDepth": "MaxDepth_m", "n_assay": "Assay_rows",
                           "n_geo": "Geology_log_rows", "Shapefile_latitude": "Lat", "Shapefile_longitude": "Lon",
                           "MINERALLOG": "MineralLog_CSV_URL", "SUM_HISTO": "Summary_Histogram_PDF", "MOSAIC": "Core_Mosaic_viewer"})
out[["Hole_name", "NVCL_HoleName", "Project", "Company", "MaxDepth_m", "Assay_rows", "Geology_log_rows", "Lat", "Lon",
     "MineralLog_CSV_URL", "Summary_Histogram_PDF", "Core_Mosaic_viewer"]].to_csv(ROOT / "STARTER_HOLES_download_list.csv", index=False)
print("\nSaved STARTER_HOLES_download_list.csv")
