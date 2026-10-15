"""03 - Attach assays (geochemistry) and logged lithology to each mineral interval.

Join rule: for each 1 m HyLogger interval, look at every assay / geology record (same hole) whose
From-To range contains the interval midpoint, and take each value from the shortest (most specific)
record that has it; equal-length assay duplicates (lab check samples) are averaged.
Assay rows longer than 10 m (whole-hole / summary rows) are removed.
Handheld XRF (pXRF) rows are removed: field screening, not lab assays.
Gold checks (the flat table merges several methods into one Au column, with no method per row):
  (a) where a hole has a dedicated gold assay (rows with Au and at most 3 other elements, e.g. fire
      assay Au/Pd/Pt) AND multi-element rows that also report Au at the same depths, and the two
      disagree (Spearman < 0.3), Au from the multi-element rows is ignored (e.g. KBD089: whole-number
      'Au' 1-35, Spearman -0.03 vs fire assay; method not recorded in the metadata);
  (b) values > 100x another record at the same depth are dropped (keep the lower);
  (c) holes whose median gold is > 1 ppm (implausible) have gold set to missing.
Missing-data sentinels (-999, -5000, -9999) -> missing. Other negative assay values are below-detection flags (for Au they equal the documented lower limits)
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

# ---- gold sanity checks (the flat table merges several methods into one Au_PPM column) ----
elem_raw = [c for c in assay if c.endswith(("_PCT", "_PPM", "_PPB"))]
n_elem = assay[elem_raw].notna().sum(axis=1)
has_au = assay["Au_PPM"].notna()
gold_row = has_au & (n_elem <= 4)        # dedicated gold assay: Au + at most 3 others (e.g. Pd, Pt, Ag)
multi_row = has_au & (n_elem > 4)        # multi-element package that also reports Au
# (a) Multi-element Au that does not agree with the dedicated gold assay at the same depths is ignored.
key = ["Hole", "FromDepth", "ToDepth"]
au_num = pd.to_numeric(assay["Au_PPM"], errors="coerce")
g = au_num[gold_row].groupby([assay.loc[gold_row, k] for k in key]).mean().rename("gold")
m = au_num[multi_row].groupby([assay.loc[multi_row, k] for k in key]).mean().rename("multi")
pair = pd.concat([g, m], axis=1, join="inner").reset_index()
rows = []
for h, d in pair.groupby("Hole"):
    if len(d) >= 10:
        rows.append((h, len(d), round(d["gold"].corr(d["multi"], method="spearman"), 2)))
check = pd.DataFrame(rows, columns=["Hole", "shared_depths", "spearman"])
print("Multi-element Au vs dedicated gold assay at the same depths (holes with >= 10 shared depths):")
print(check.to_string(index=False) if len(check) else "  none")
bad_multi = set(check.loc[check["spearman"] < 0.3, "Hole"])
drop_multi = multi_row & assay["Hole"].isin(bad_multi)
print("Multi-element Au values ignored (disagree with the gold assay):",
      assay.loc[drop_multi, "Hole"].value_counts().to_dict() or "none")
assay.loc[drop_multi, "Au_PPM"] = np.nan
# (b) Same sample depth reported twice with values > 100x apart: keep the lower value.
au = assay["Au_PPM"].abs().clip(lower=0.0005)
lo = au.groupby([assay["Hole"], assay["FromDepth"], assay["ToDepth"]]).transform("min")
unit_clash = assay["Au_PPM"].notna() & (au > 100 * lo)
print("Au values dropped (>100x another record at the same depth):",
      assay.loc[unit_clash, "Hole"].value_counts().to_dict() or "none")
assay.loc[unit_clash, "Au_PPM"] = np.nan
# (c) A whole hole whose MEDIAN gold assay is above 1 ppm is not plausible for drill core (it would make
#     the entire hole ore grade); e.g. 18KLDD008 reports 50-100 below 330 m, which look like leach
#     recovery percentages. Gold is set to missing for such holes (other elements are kept).
med = assay.groupby("Hole")["Au_PPM"].median()
bad_au = med[med > 1].index.tolist()
print("Holes whose gold is set to missing (median Au > 1 ppm, implausible):",
      {h: round(med[h], 2) for h in bad_au} or "none")
assay.loc[assay["Hole"].isin(bad_au), "Au_PPM"] = np.nan

elem_cols = [c for c in assay if c.endswith(("_PCT", "_PPM", "_PPB"))]
assay[elem_cols] = assay[elem_cols].apply(pd.to_numeric, errors="coerce")
# Missing-data sentinels are NOT below-detection flags: -5000 fills almost every element in 1,579 rows
# of ETG0202/0204/0226 (incl. Au), and -999 is GSWA's missing marker (meta_data_quality checks 34-36).
SENTINELS = [-999, -5000, -9999]
is_sent = assay[elem_cols].isin(SENTINELS)
print("Missing-data sentinels (-999/-5000/-9999) set to missing, per hole:",
      assay.loc[is_sent.any(axis=1), "Hole"].value_counts().to_dict() or "none")
assay[elem_cols] = assay[elem_cols].mask(is_sent)
assay[elem_cols] = assay[elem_cols].where(assay[elem_cols].isna() | (assay[elem_cols] >= 0), assay[elem_cols].abs() / 2)
assay = assay.copy()  # de-fragment after the column-wise cleaning (avoids pandas PerformanceWarning spam)
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
