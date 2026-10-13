"""00b - Choose EXTRA holes to download (expansion beyond the 12 starter holes).

Pool: holes with a mineral log + > 100 assay rows + > 50 geology rows (522 holes), diamond core only.
  Tier A - more Yilgarn gold (strengthens gold statistics): depth 100-700 m, ranked by data volume,
           at most 4 holes per project, not already downloaded.                        -> N_A holes
  Tier B - other regions (tests whether patterns hold across regions):
           gold: top 3 per region outside the Yilgarn; copper: top 3 in the 4 biggest copper regions;
           nickel: top 3 in the Yilgarn.                                               -> ~30 holes
Outputs (repo root): EXPANSION_download_list.csv  and  download_mineral_logs.sh
"""
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import pandas as pd
from config import CSV_DIR, LOGS_DIR, require_data
require_data(need_logs=False)

ROOT = Path(__file__).resolve().parents[1]
CSV = CSV_DIR
LOGS = LOGS_DIR
N_A, PER_PROJECT = 30, 4

idx = pd.read_csv(CSV / "HyLogger_Dataset_Index.csv", encoding="utf-8-sig", dtype=str)
col = pd.read_csv(CSV / "WAMEX_Collar.csv", encoding="utf-8-sig", dtype={"CollarId": str})
na = pd.read_csv(CSV / "WAMEX_AssayFlatData.csv", encoding="utf-8-sig", usecols=["CollarId"], dtype=str)["CollarId"].value_counts()
ng = pd.read_csv(CSV / "WAMEX_DHGeology.csv", encoding="utf-8-sig", usecols=["CollarId"], dtype=str)["CollarId"].value_counts()
d = idx.merge(col, left_on="WAMEX_CollarId", right_on="CollarId")
d["n_assay"] = d["CollarId"].map(na).fillna(0).astype(int)
d["n_geo"] = d["CollarId"].map(ng).fillna(0).astype(int)
d["MaxDepth"] = pd.to_numeric(d["MaxDepth"], errors="coerce")
d = d[d["MINERALLOG"].fillna("").str.startswith("http") & (d["n_assay"] > 100) & (d["n_geo"] > 50) & (d["HoleType"] == "DD")]
d["file"] = d["MINERALLOG"].map(lambda u: Path(parse_qs(urlparse(u).query)["fileName"][0]).name)
have = {p.name for p in LOGS.glob("SM_*.csv")} if LOGS.exists() else set()
d = d[~d["file"].isin(have)].assign(score=lambda x: x["n_assay"] + x["n_geo"]).sort_values("score", ascending=False)
d["commodity"] = d["Commodity_vocab"].fillna("").str.split(",").str[0]

# Tier A
ya = d[(d["TectonicRegion"] == "Yilgarn Craton") & (d["Commodity_vocab"] == "gold") & d["MaxDepth"].between(100, 700)]
tier_a = ya.groupby(ya["PROJECT"].fillna("(none)"), sort=False).head(PER_PROJECT).head(N_A).assign(Tier="A: Yilgarn gold")
# Tier B
rest = d[~d.index.isin(tier_a.index)]
gold_other = rest[(rest["commodity"] == "gold") & (rest["TectonicRegion"] != "Yilgarn Craton")]
gold_other = gold_other[gold_other["TectonicRegion"].map(gold_other["TectonicRegion"].value_counts()) >= 3]
b1 = gold_other.groupby("TectonicRegion").head(3).assign(Tier="B: gold, other region")
cu = rest[rest["commodity"] == "copper"]
top_cu = cu["TectonicRegion"].value_counts().head(4).index
b2 = cu[cu["TectonicRegion"].isin(top_cu)].groupby("TectonicRegion").head(3).assign(Tier="B: copper")
b3 = rest[(rest["commodity"] == "nickel") & (rest["TectonicRegion"] == "Yilgarn Craton")].head(3).assign(Tier="B: nickel")
sel = pd.concat([tier_a, b1, b2, b3]).drop_duplicates("file")

out = sel[["Tier", "Hole_name", "file", "TectonicRegion", "Commodity_vocab", "PROJECT", "Company", "MaxDepth", "n_assay", "n_geo", "MINERALLOG"]]
out.columns = ["Tier", "Hole_name", "File", "Region", "Commodity", "Project", "Company", "MaxDepth_m", "Assay_rows", "Geology_rows", "MineralLog_URL"]
out.to_csv(ROOT / "EXPANSION_download_list.csv", index=False)
print(f"Already downloaded: {len(have)} files. New holes selected: {len(out)}")
print(out.groupby(["Tier", "Region"]).size().to_string())

sh = ["#!/bin/bash", "# Downloads every mineral log in EXPANSION_download_list.csv into MINERAL_LOGS/.",
      "# Run from the HyLogger folder:  bash download_mineral_logs.sh", "set -u",
      'DEST="${HYLOGGER_DATA:-INTERIM_HyLogger_Dataset_Package_GDA2020_DB}/MINERAL_LOGS"', 'mkdir -p "$DEST"', "ok=0; fail=0"]
for f, u in zip(out["File"], out["MineralLog_URL"]):
    sh.append(f'if [ -s "$DEST/{f}" ]; then echo "skip {f}"; '
              f'elif curl -L -sS -f --retry 2 -o "$DEST/{f}" "{u}"; then echo "ok   {f}"; ok=$((ok+1)); '
              f'else echo "FAIL {f}"; rm -f "$DEST/{f}"; fail=$((fail+1)); fi; sleep 0.5')
sh.append('echo "Done: $ok downloaded, $fail failed."')
(ROOT / "download_mineral_logs.sh").write_text("\n".join(sh) + "\n")
print("\nWrote EXPANSION_download_list.csv and download_mineral_logs.sh")
