"""Shared paths for every pipeline script.

Where is the HyLogger data?  (checked in this order)
  1. The environment variable HYLOGGER_DATA, if set, e.g.
       export HYLOGGER_DATA=~/Downloads/INTERIM_HyLogger_Dataset_Package_GDA2020_DB
  2. Otherwise: <repo root>/INTERIM_HyLogger_Dataset_Package_GDA2020_DB
Scripts find the repo root from their own location, so they can be run from any folder.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HYLOGGER_DATA", ROOT / "INTERIM_HyLogger_Dataset_Package_GDA2020_DB")).expanduser()
CSV_DIR = DATA_DIR / "DATABASES" / "CSV"
LOGS_DIR = DATA_DIR / "MINERAL_LOGS"
OUT = ROOT / "outputs"


def require_data(need_logs=True):
    """Stop with a clear message (instead of a long traceback) if the data is not where we expect."""
    problems = []
    if not (CSV_DIR / "HyLogger_Dataset_Index.csv").exists():
        problems.append(f"DASC package not found: expected {CSV_DIR / 'HyLogger_Dataset_Index.csv'}")
    if need_logs and not any(LOGS_DIR.glob("SM_*.csv")):
        problems.append(f"No mineral logs (SM_*.csv) found in {LOGS_DIR}")
    if problems:
        sys.exit("\n".join(["DATA NOT FOUND", *problems, "",
                            "Fix: download the data as described in README.md (section 1), then either",
                            "  - put the INTERIM_HyLogger_Dataset_Package_GDA2020_DB folder in the repo root, or",
                            "  - export HYLOGGER_DATA=/path/to/INTERIM_HyLogger_Dataset_Package_GDA2020_DB"]))
