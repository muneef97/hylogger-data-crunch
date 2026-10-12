"""01 - Load all HyLogger mineral log CSVs (SM_*.csv) into one tidy table.

Each SM file has ~16 metadata lines before the real header (starts with 'Depth_from').
Missing values are written as 'N.A.'.
Interval checks are printed (missing/invalid depths, duplicates, overlaps, gaps); nothing is deleted here.
Output: outputs/01_mineral_intervals.csv
"""
from pathlib import Path
import pandas as pd
from config import CSV_DIR, LOGS_DIR, require_data
require_data(need_logs=True)

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = LOGS_DIR
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def read_mineral_log(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_row = next(i for i, l in enumerate(lines) if l.startswith("Depth_from"))
    df = pd.read_csv(path, skiprows=header_row, na_values=["N.A."])
    df.insert(0, "Hole", path.stem.replace("SM_", ""))
    return df

frames = [read_mineral_log(p) for p in sorted(LOG_DIR.glob("SM_*.csv"))]
df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={"Depth_from_(m)": "From_m", "Depth_to_(m)": "To_m"})
df["Mid_m"] = (df["From_m"] + df["To_m"]) / 2

# ---- interval checks (report only, nothing is deleted) ----
print("Interval checks:")
print(f"  missing depths: {int(df[['From_m', 'To_m']].isna().sum().sum())} | To <= From: {int((df['To_m'] <= df['From_m']).sum())}"
      f" | duplicate rows: {int(df.duplicated().sum())} | same hole+depth twice: {int(df.duplicated(['Hole', 'From_m']).sum())}")
print(f"  interval lengths (m): {(df['To_m'] - df['From_m']).round(2).value_counts().head(3).to_dict()}")
dup_cols = df.columns[df.columns.duplicated()].tolist()
spellings = [c for c in df if c.lower() == "invalid_(system_tir)"]
print(f"  duplicate column names: {dup_cols or 'none'}" + (f" | note: {spellings} are two spellings of one column, merged in script 02" if len(spellings) > 1 else ""))
for hole, g in df.sort_values(["Hole", "From_m"]).groupby("Hole"):
    prev_to = g["To_m"].shift()
    overlaps, gaps = int((g["From_m"] < prev_to).sum()), int((g["From_m"] > prev_to).sum())
    if overlaps or gaps:
        print(f"  {hole}: overlaps={overlaps}, gaps={gaps} ({(g['From_m'] - prev_to)[g['From_m'] > prev_to].sum():.0f} m not scanned)")

df.to_csv(OUT / "01_mineral_intervals.csv", index=False)

print(f"Holes loaded: {df['Hole'].nunique()}  |  Intervals: {len(df)}  |  Columns: {df.shape[1]}")
print("\nPer hole (intervals, depth range m):")
print(df.groupby("Hole").agg(n=("From_m", "size"), top=("From_m", "min"), bottom=("To_m", "max")).to_string())
miss = df.isna().mean().sort_values()
print("\nColumns always present (<5% missing):", list(miss[miss < 0.05].index))
print("\nColumns mostly empty (>95% missing):", list(miss[miss > 0.95].index))
