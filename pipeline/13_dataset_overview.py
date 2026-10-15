"""13 - Overview of the WHOLE HyLogger dataset (the DASC index: 2,019 SCANS of ~1,900 named holes).

One index row = one scan. Some holes have several scans (wedges, cuttings, chips), so hole counts use
unique hole names. Data availability = the flags recorded IN THE INDEX (not checked record by record).
Commodity groups are mutually exclusive, in priority order: Not recorded > Petroleum > Gold only >
Gold + other commodities > Base metals (no gold) > Other. Many holes list several commodities.

Uses only the index + collar tables, so it covers every hole, not just the ones analysed in depth:
  * holes by source database, region and commodity
  * what extra data exists per hole (mineral log, collar, survey, assays, geology log)
  * scanning over time
  * a map of every hole, with the analysed holes highlighted
Outputs: outputs/13_dataset_overview.csv, outputs/plots/13a_all_holes_map.png, outputs/plots/13b_holes_by_region.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CSV_DIR, LOGS_DIR, require_data
require_data(need_logs=False)

ROOT = Path(__file__).resolve().parents[1]
CSV = CSV_DIR
OUT = ROOT / "outputs"; PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)

idx = pd.read_csv(CSV / "HyLogger_Dataset_Index.csv", encoding="utf-8-sig", dtype=str)
wamex = pd.read_csv(CSV / "WAMEX_Collar.csv", encoding="utf-8-sig", dtype=str)[["CollarId", "TectonicRegion"]]
wapims = pd.read_csv(CSV / "WAPIMS_Collar.csv", encoding="utf-8-sig", dtype=str)[["WellId", "Basin"]]
n_idx = len(idx)
d = idx.merge(wamex, left_on="WAMEX_CollarId", right_on="CollarId", how="left", validate="many_to_one") \
       .merge(wapims, left_on="WAPIMS_report_Wnumber", right_on="WellId", how="left", validate="many_to_one")
assert len(d) == n_idx, "a join duplicated index rows"
d["Region"] = d["TectonicRegion"].fillna(d["Basin"].map(lambda b: f"{b} Basin" if isinstance(b, str) else b)).fillna("Region not recorded")
d["Lat"] = pd.to_numeric(d["Shapefile_latitude"], errors="coerce")
d["Lon"] = pd.to_numeric(d["Shapefile_longitude"], errors="coerce")
c = d["Commodity_vocab"].fillna("").str.lower().str.strip()
d["Commodity_group"] = np.select(
    [c.eq(""), c.str.contains("petroleum"), c.eq("gold"), c.str.contains("gold"),
     c.str.contains("copper|nickel|zinc|lead|base metal")],
    ["Not recorded", "Petroleum", "Gold only", "Gold + other commodities", "Base metals (no gold)"], "Other")
scan_date = pd.to_datetime(d["SCAN_DATE"], dayfirst=True, errors="coerce")
tir_date = pd.to_datetime(d["SCDATE_TIR"], dayfirst=True, errors="coerce")
d["Scan_year"] = scan_date.fillna(tir_date).dt.year
d["Scan_type"] = np.select([d["MINERALLOG"].str.contains("wedge", case=False, na=False),
                            d["MINERALLOG"].str.contains("cutting", case=False, na=False),
                            d["MINERALLOG"].str.contains("chip", case=False, na=False)],
                           ["Wedge", "Cuttings", "Chips"], "Core / main scan")
analysed = (set(pd.read_csv(OUT / "02_holes.csv")["Index_hole_name"]) if (OUT / "02_holes.csv").exists() else set())
d["Analysed_here"] = d["Hole_name"].isin(analysed)

flags = {"Mineral log": d["MINERALLOG"].fillna("").str.startswith("http"),   # values checked: URL or empty; others 0/1
         "Collar": d["Collar_data"] == "1", "Survey": d["Survey_data"] == "1",
         "Assays": d["Assay_data"] == "1", "Geology log": d["Logging_data"] == "1"}
for k, v in flags.items(): d[f"has_{k}"] = v
all3 = flags["Mineral log"] & flags["Assays"] & flags["Geology log"]

holes = d.groupby("Hole_name")
print(f"Scans in the HyLogger index: {len(d)} | unique hole names: {d['Hole_name'].nunique()}")
print("Scan types:", d["Scan_type"].value_counts().to_dict())
print("\nBy source database (holes):", holes["Database"].first().value_counts().to_dict())
print("\nBy commodity group (holes; mutually exclusive, priority order):\n",
      holes["Commodity_group"].first().value_counts().to_string())
multi = c.str.contains(",")
print(f"Scans listing more than one commodity: {multi.sum()}; most common:",
      c[multi].value_counts().head(5).to_dict())
print("\nTop regions (holes):\n", holes["Region"].first().value_counts().head(12).to_string())
print("\nData flagged in the index, per hole (any of its scans):")
hf = pd.DataFrame(flags).groupby(d["Hole_name"]).any()
for k in flags: print(f"  {k:12s} {hf[k].sum():5d} ({hf[k].mean():.0%})")
h3 = hf["Mineral log"] & hf["Assays"] & hf["Geology log"]
print(f"  Mineral log + assays + geology log: {h3.sum()} ({h3.mean():.0%})  "
      f"(index flags: data is recorded, not checked for completeness)")
yrs = d["Scan_year"].dropna()
print(f"\nScan years: {int(yrs.min())}-{int(yrs.max())}; scans per year (last 6):",
      d["Scan_year"].value_counts().sort_index().tail(6).astype(int).to_dict())
n_an = d.loc[d["Analysed_here"], "Hole_name"].nunique()
print(f"\nAnalysed in depth here: {n_an} holes = {n_an / d['Hole_name'].nunique():.1%} of the holes in the index "
      f"(chosen for rich data, not at random: see outputs/00_selection_bias.csv)")
map_ok = d["Lat"].between(-45, -8) & d["Lon"].between(110, 130)        # broad WA box incl. offshore Timor Sea wells
print(f"\nScans with usable WA coordinates: {map_ok.sum()} | missing: {(d['Lat'].isna() | d['Lon'].isna()).sum()} | "
      f"outside the WA box (not mapped):", d.loc[~map_ok & d["Lat"].notna() & d["Lon"].notna(), ["Hole_name", "Lat", "Lon"]].values.tolist())

d[["Hole_name", "Database", "Region", "Commodity_group", "Lat", "Lon", "Scan_year", "Analysed_here"] +
  [f"has_{k}" for k in flags]].to_csv(OUT / "13_dataset_overview.csv", index=False)

# ---- figures ----
SURF, INK, INK2, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b3"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COLS = {"Gold only": "#eda100", "Gold + other commodities": "#e87ba4", "Base metals (no gold)": "#2a78d6",
        "Petroleum": "#4a3aa7", "Other": "#1baf7a", "Not recorded": GREY}
m = d[map_ok].drop_duplicates("Hole_name")                              # one dot per hole
fig, ax = plt.subplots(figsize=(7.5, 8.5))
for g, col in COLS.items():
    q = m[m["Commodity_group"] == g]
    ax.scatter(q["Lon"], q["Lat"], s=9, color=col, alpha=0.75, linewidths=0, label=f"{g} ({len(q)})")
a = m[m["Analysed_here"]]
ax.scatter(a["Lon"], a["Lat"], s=70, facecolors="none", edgecolors=INK, linewidths=1.4, label=f"Analysed in this project ({len(a)})")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect("equal")
ax.set_title(f"All {len(m):,} mapped HyLogger holes in the GSWA index", loc="left", weight="bold")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8); ax.grid(color=GREY, lw=0.3, alpha=0.5)
fig.savefig(PLOTS / "13a_all_holes_map.png", dpi=150, bbox_inches="tight"); plt.close(fig)

reg = holes["Region"].first().value_counts().head(12).sort_values()
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(reg.index, reg.values, color="#2a78d6", height=0.65)
for y, v in enumerate(reg.values): ax.text(v + 8, y, str(v), va="center", fontsize=8, color=INK2)
ax.set_xlabel("Number of scanned holes"); ax.set_xlim(0, reg.max() * 1.12)
ax.set_title("Where the HyLogger holes are (top 12 regions)", loc="left", weight="bold")
fig.savefig(PLOTS / "13b_holes_by_region.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nFigures: 13a_all_holes_map.png, 13b_holes_by_region.png")
