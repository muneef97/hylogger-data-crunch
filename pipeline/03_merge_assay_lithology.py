"""03 - Attach assays (geochemistry) and logged lithology to each mineral interval.

Join rule: for each 1 m HyLogger interval, look at every assay / geology record (same hole) whose
From-To range contains the interval midpoint, and take each value from the shortest (most specific)
record that has it; equal-length assay duplicates (lab check samples) are averaged.
Assay rows longer than 10 m (whole-hole / summary rows) are removed.
Handheld XRF (pXRF) rows are removed: field screening, not lab assays.
Gold checks: values > 100x another record at the same depth are dropped (unit clash, e.g. ppb stored
as ppm), and holes whose median gold is > 1 ppm (implausible) have gold set to missing.
Negative assay values are below-detection flags (for Au they equal the documented lower limits)
-> replaced with half the absolute value. Missing assays stay missing (never filled with 0).
Only assay elements with data in >= 30% of the matched intervals are kept.
Output: outputs/03_merged_intervals.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from config import CSV_DIR, LOGS_DIR, require_data
require_data(need_logs=False)

ROOT = Path(__file__).resolve().parents[1]
CSV = CSV_DIR
OUT = ROOT / "outputs"

minerals = pd.read_csv(OUT / "02_minerals_clean.csv")
holes = pd.read_csv(OUT / "02_holes.csv")
collar_to_hole = dict(zip(holes["WAMEX_CollarId"].astype(int), holes["Hole"]))

def load_for_holes(name, usecols=None):
    df = pd.read_csv(CSV / name, encoding="utf-8-sig", na_values=["NULL"], usecols=usecols, low_memory=False)
    df = df[df["CollarId"].isin(collar_to_hole)].copy()
    df["Hole"] = df["CollarId"].map(collar_to_hole)
    return df

def attach(intervals, table, cols, numeric):
    """Attach every source record whose [FromDepth, ToDepth) contains the interval midpoint.

    Source tables contain OVERLAPPING and DUPLICATE intervals (e.g. gold and multi-element assays stored
    as separate rows for the same depth), so all covering records are considered, not just one:
      numeric (assays)  -> per element, mean of the SHORTEST covering records that have a value
      text (geology)    -> value from the shortest (most specific) covering record that has one
    Also returns how many source records covered each interval.
    """
    out, n_cover = [], []
    for hole, m in intervals.groupby("Hole"):
        t = table[table["Hole"] == hole]
        if t.empty:
            continue
        # pair each record with the intervals whose midpoint lies in [FromDepth, ToDepth) without a full
        # cross join (that needs too much memory on long holes): binary search on sorted midpoints
        t = t.dropna(subset=["FromDepth", "ToDepth"])
        m = m.sort_values("Mid_m"); mids = m["Mid_m"].to_numpy()
        i0 = np.searchsorted(mids, t["FromDepth"].to_numpy(), side="left")
        i1 = np.searchsorted(mids, t["ToDepth"].to_numpy(), side="left")
        n = np.clip(i1 - i0, 0, None)
        if n.sum() == 0:
            continue
        rec = np.repeat(np.arange(len(t)), n)
        pos = np.concatenate([np.arange(a, b) for a, b in zip(i0, i1) if b > a])
        x = t[["FromDepth", "ToDepth"] + cols].iloc[rec].reset_index(drop=True)
        x["index"] = m.index.to_numpy()[pos]
        n_cover.append(x.groupby("index").size())
        x = x.assign(_len=x["ToDepth"] - x["FromDepth"]).sort_values("_len", kind="stable")
        if numeric:
            # per element: keep only the shortest records that have a value, then average them
            # (equal-length duplicates are lab check samples, e.g. BWD072 "_CD")
            vals = {}
            for c in cols:
                shortest = x["_len"].where(x[c].notna()).groupby(x["index"]).transform("min")
                vals[c] = x[c].where(x["_len"] == shortest)
            out.append(pd.DataFrame(vals).groupby(x["index"]).mean())
        else:
            out.append(x.groupby("index")[cols].first())   # first non-null per column, shortest first
    res = pd.concat(out).reindex(intervals.index) if out else pd.DataFrame(index=intervals.index, columns=cols)
    cover = pd.concat(n_cover).reindex(intervals.index).fillna(0).astype(int) if n_cover else pd.Series(0, index=intervals.index)
    return res, cover

# ---- assays ----
assay = load_for_holes("WAMEX_AssayFlatData.csv")

# Drop handheld (portable XRF) readings. They are field screening, not lab assays: in 12DHDD0002 the
# Niton pXRF rows report Au up to 257 ppm where the fire assay of the same metre gives < 0.07 ppm.
# The metadata table lists methods per hole, not per row, so in a hole that has a portable method we
# drop the rows without a CompanySampleId (lab samples always carry one; here the pXRF rows never do).
meta = pd.read_csv(CSV / "WAMEX_AssayFlatMetadata.csv", encoding="utf-8-sig", low_memory=False)
PORTABLE = r"niton|pxrf|portable|handheld|olympus|vanta"
meta_txt = meta["LabMethod"].fillna("") + " " + meta["Laboratory"].fillna("")
pxrf_collars = set(meta.loc[meta_txt.str.contains(PORTABLE, case=False), "CollarId"])
is_pxrf = assay["CollarId"].isin(pxrf_collars) & assay["CompanySampleId"].isna()
print("Handheld XRF rows removed per hole:", assay.loc[is_pxrf, "Hole"].value_counts().to_dict() or "none")
assay = assay[~is_pxrf]

# Drop assay rows longer than 10 m: whole-hole or summary rows, not samples
# (e.g. ABND198 has one 0-308.8 m row with Au 23.22 ppm). 99% of rows in the source are <= 4 m.
too_long = (assay["ToDepth"] - assay["FromDepth"]) > 10
print("Assay rows longer than 10 m removed per hole:", assay.loc[too_long, "Hole"].value_counts().to_dict() or "none")
assay = assay[~too_long]

# ---- gold sanity checks (the flat table can mix units / measurement types in the Au_PPM column) ----
# (1) Same sample depth reported twice with values > 100x apart (e.g. KBD089: 0.002 and 11 -> a ppb
#     value stored as ppm). Keep the lower value, drop the higher one.
au = assay["Au_PPM"].abs().clip(lower=0.0005)
lo = au.groupby([assay["Hole"], assay["FromDepth"], assay["ToDepth"]]).transform("min")
unit_clash = assay["Au_PPM"].notna() & (au > 100 * lo)
print("Au values dropped as unit clashes (>100x the other record at the same depth):",
      assay.loc[unit_clash, "Hole"].value_counts().to_dict() or "none")
assay.loc[unit_clash, "Au_PPM"] = np.nan
# (2) A whole hole whose MEDIAN gold assay is above 1 ppm is not plausible for drill core (it would make
#     the entire hole ore grade); e.g. 18KLDD008 reports 50-100 below 330 m, which look like leach
#     recovery percentages. Gold is set to missing for such holes (other elements are kept).
med = assay.groupby("Hole")["Au_PPM"].median()
bad_au = med[med > 1].index.tolist()
print("Holes whose gold is set to missing (median Au > 1 ppm, implausible):",
      {h: round(med[h], 2) for h in bad_au} or "none")
assay.loc[assay["Hole"].isin(bad_au), "Au_PPM"] = np.nan

elem_cols = [c for c in assay if c.endswith(("_PCT", "_PPM", "_PPB"))]
assay[elem_cols] = assay[elem_cols].apply(pd.to_numeric, errors="coerce")
assay[elem_cols] = assay[elem_cols].where(assay[elem_cols] >= 0, assay[elem_cols].abs() / 2)
A, a_cover = attach(minerals, assay, elem_cols, numeric=True)
keep = A.columns[A.notna().mean() >= 0.30]
A = A[keep]

# ---- lithology ----
geo = load_for_holes("WAMEX_DHGeology.csv")
# GSWARegolithName is NOT used: 0 % filled for these holes (0.3 % across the whole source file).
# Comments IS used: ~47 % filled and often holds the geologist's written rock description.
geo_cols = ["CompanyLithologyCodeMerged", "CompanyLithologyName", "GSWARockName", "Weathering_Code", "Comments"]
G, g_cover = attach(minerals, geo, geo_cols, numeric=False)
G.columns = ["Lith_code", "Lith_name", "GSWA_rock", "Weathering", "Geology_comments"]

merged = pd.concat([minerals, A, G], axis=1)
merged["n_assay_records"], merged["n_geology_records"] = a_cover.values, g_cover.values
print(f"Intervals covered by more than one assay record: {(a_cover > 1).mean():.1%} | geology: {(g_cover > 1).mean():.1%} (overlaps merged)")
merged.to_csv(OUT / "03_merged_intervals.csv", index=False)

print(f"Merged: {merged.shape[0]} intervals x {merged.shape[1]} columns")
print(f"Assay elements kept ({len(keep)}):", list(keep))
cov = pd.DataFrame({
    "assay_matched": A.notna().any(axis=1).groupby(merged["Hole"]).mean(),
    "lith_matched": G["Lith_code"].notna().groupby(merged["Hole"]).mean(),
}).round(2)
print("\nShare of HyLogger intervals with a matching assay / lithology record:\n", cov.to_string())
print("\nShare of intervals with each geology field filled:")
print(G.notna().mean().round(3).to_string())
any_lith = G[["Lith_code", "Lith_name", "GSWA_rock", "Geology_comments"]].notna().any(axis=1)
print("\nIntervals with ANY rock information, per hole (code, name, GSWA rock or comment):")
print(any_lith.groupby(merged["Hole"]).mean().round(2).to_string())
print("\nExample comments:")
print(G["Geology_comments"].dropna().drop_duplicates().head(8).to_string())
