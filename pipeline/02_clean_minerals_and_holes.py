"""02 - Clean the mineral table and build a hole (collar) table.

Cleaning rules:
  * Mineral proportions sum to 1 per interval, so blank = not detected -> 0.
  * Holes are EXCLUDED by rule (not by name), with the reason printed:
      - chip/cuttings scans ('_chips' files): not drill core
      - no automated ('System') results for most intervals: only geologist-refined 'User' results,
        a different method from the other holes (e.g. the MacPhersons MRDH holes)
      - no TIR data at all: TIR minerals (quartz, feldspar, ...) are needed for every model
  * 'User' columns are then dropped; all remaining holes use the same automated method.
  * Columns renamed to short names, e.g. 'White-Mica_(System_SWIR)' -> 'SWIR_White-Mica'.
  * Blank -> 0 is checked, not assumed: each interval's SWIR and TIR proportions (including the 'Invalid'
    category, which records bad readings) must sum to 1; the result is printed.
  * Only minerals NEVER detected in any interval are dropped; rare/trace minerals are kept and their
    frequency is printed.
  * QC flag: SWIR invalid fraction > 0.5 -> low-confidence interval.
Outputs: outputs/02_minerals_clean.csv, outputs/02_holes.csv
"""
from pathlib import Path
import pandas as pd
from config import CSV_DIR, LOGS_DIR, require_data
require_data(need_logs=False)

ROOT = Path(__file__).resolve().parents[1]
CSV = CSV_DIR
OUT = ROOT / "outputs"

df = pd.read_csv(OUT / "01_mineral_intervals.csv")

# ---- Exclusion rules (applied per hole) ----
sys_cols = [c for c in df if "(System_" in c]
tir_cols = [c for c in df if "(System_TIR)" in c]
per_hole = df.groupby("Hole").agg(n=("From_m", "size"))
per_hole["share_no_system"] = df[sys_cols].isna().all(axis=1).groupby(df["Hole"]).mean()
per_hole["has_tir"] = df[tir_cols].notna().any(axis=1).groupby(df["Hole"]).any()
reasons = {}
for h, r in per_hole.iterrows():
    why = []
    if h.lower().endswith("_chips"): why.append("chip scan, not drill core")
    if r["share_no_system"] > 0.5:   why.append("no automated (System) results - geologist-refined only")
    if not r["has_tir"]:             why.append("no TIR data")
    if why: reasons[h] = "; ".join(why)
print(f"Holes loaded: {len(per_hole)} | excluded: {len(reasons)} | kept: {len(per_hole) - len(reasons)}")
for h, why in reasons.items():
    print(f"  excluded {h:14s} ({int(per_hole.at[h, 'n'])} intervals): {why}")
pd.Series(reasons, name="reason").rename_axis("Hole").to_csv(OUT / "02_excluded_holes.csv")
df = df[~df["Hole"].isin(reasons)].reset_index(drop=True)
df = df.drop(columns=[c for c in df if "(User_" in c])

# Merge the two TIR invalid spellings
if "INVALID_(System_TIR)" in df and "Invalid_(System_TIR)" in df:
    df["Invalid_(System_TIR)"] = df["Invalid_(System_TIR)"].fillna(df.pop("INVALID_(System_TIR)"))

# Short names + blanks -> 0 for mineral proportions
rename = {c: f"{c.split('_(System_')[1].rstrip(')')}_{c.split('_(')[0]}" for c in df if "_(System_" in c}
df = df.rename(columns=rename)
mineral_cols = list(rename.values())
df[mineral_cols] = df[mineral_cols].fillna(0)

# Check that blank really means "not detected": proportions (incl. Invalid) must sum to 1 per interval
for sensor in ["SWIR", "TIR"]:
    cols = [c for c in mineral_cols if c.startswith(sensor + "_")]
    total = df[cols].sum(axis=1)
    print(f"{sensor}: intervals whose proportions sum to 1 (+-0.01): {(total.sub(1).abs() < 0.01).mean():.1%}")

# Drop only minerals never detected anywhere; keep rare ones and report how often each occurs
absent = [c for c in mineral_cols if df[c].max() == 0 and "Invalid" not in c]
df = df.drop(columns=absent)
freq = (df[[c for c in mineral_cols if c not in absent]] > 0).mean().sort_values()
print("Rare minerals kept (present in < 5% of intervals):",
      {c: f"{v:.1%}" for c, v in freq[freq < 0.05].items()})

df["QC_low_confidence"] = df["SWIR_Invalid"] > 0.5
df.to_csv(OUT / "02_minerals_clean.csv", index=False)

# ---- hole table ----
# Matched by the mineral-log FILE name (SM_<file>.csv), because index hole names can differ
# from file names (e.g. index 'KPDDH 003' vs file 'SM_KPDDH003.csv').
idx = pd.read_csv(CSV / "HyLogger_Dataset_Index.csv", encoding="utf-8-sig")
col = pd.read_csv(CSV / "WAMEX_Collar.csv", encoding="utf-8-sig")
idx["Hole"] = idx["MINERALLOG"].fillna("").str.extract(r"SM_(.+)\.csv", expand=False)
holes = idx[idx["Hole"].isin(df["Hole"].unique())].drop_duplicates("Hole")[
    ["Hole", "Hole_name", "WAMEX_CollarId", "PROJECT", "Commodity_vocab", "Shapefile_latitude", "Shapefile_longitude"]]
holes = holes.merge(col[["CollarId", "Company", "MaxDepth", "HoleType", "TectonicRegion"]],
                    left_on="WAMEX_CollarId", right_on="CollarId", how="left").drop(columns="CollarId")
holes = holes.rename(columns={"Hole_name": "Index_hole_name", "Shapefile_latitude": "Lat", "Shapefile_longitude": "Lon"})
missing = sorted(set(df["Hole"]) - set(holes["Hole"]))
if missing: print("WARNING - holes not found in the index:", missing)
holes.to_csv(OUT / "02_holes.csv", index=False)

print(f"Clean table: {df.shape[0]} intervals x {df.shape[1]} columns")
print("Dropped (never detected in any interval):", absent)
print("Mineral columns kept:", [c for c in mineral_cols if c not in absent])
print(f"\nLow-confidence intervals (SWIR invalid > 50%): {df['QC_low_confidence'].sum()} "
      f"({df['QC_low_confidence'].mean():.1%})")
print(df.groupby("Hole")["QC_low_confidence"].mean().round(3).sort_values(ascending=False).to_string())
print("\nHole table:\n", holes.to_string(index=False))
