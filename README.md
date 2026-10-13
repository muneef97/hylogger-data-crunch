# HyLogger Data Crunch — minerals to rock type, boundaries, chemistry and gold vectors

GSWA HyLogger Hackathon 2026 · **Theme 1: Data crunch**

A reproducible, open Python pipeline that takes GSWA HyLogger mineral logs (and, optionally, raw spectra) and answers
GSWA's five Data Crunch questions:

| GSWA question | Scripts | Main output |
|---|---|---|
| 1. Finding features of interest (anomalies, endmembers in bulk) | 05, 07, 14 | assemblage clusters, anomaly zones, 7 stable mineral endmembers |
| 2. Feature analysis outside TSG | 15 | white-mica feature from raw `.tsg` spectra, checked against TSG |
| 3. Boundary detection | 11 | mineral zones; boundaries tested against logged contacts |
| 4. Determining rock type | 10 | felsic / intermediate / mafic per metre, ternary plot, pseudologs vs geology log |
| 5. Deriving other parameters in bulk | 12, 16 | 34 assay elements predicted from mineralogy (PLS, boosting), tested on unseen holes and companies |

Demonstrated on **65 holes** (29,003 one-metre intervals) from gold, copper and nickel projects across WA; runs
unchanged on any set of HyLogger summary mineral logs (`SM_<hole>.csv`).
**Results: see [`FINDINGS.md`](FINDINGS.md).**

---

## 1. Get the data (not included in this repo)

HyLogger data — and anything derived from it per metre (tables, the HTML viewer, run logs) — must not be committed.
`.gitignore` already excludes it. Download it yourself:

1. **DASC package** – https://dasc.dmirs.wa.gov.au/ → search **"HyLogger Drillhole Package"** → download
   *INTERIM HyLogger Dataset Package (GDA2020, database)* and unzip into the repo root, giving
   `INTERIM_HyLogger_Dataset_Package_GDA2020_DB/DATABASES/CSV/...`
2. **Mineral logs** (`SM_<hole>.csv`) into `INTERIM_HyLogger_Dataset_Package_GDA2020_DB/MINERAL_LOGS/`:
   * 12 starter holes: `STARTER_HOLES_download_list.csv` (from script 00)
   * 59 expansion holes: `bash download_mineral_logs.sh` (list in `EXPANSION_download_list.csv`, from script 00b)
   * 71 downloaded → 65 analysed (6 excluded in script 02, see Design choices).
3. **Optional – raw spectra for step 15** (~550 MB; downloads only the spectra, not the core photos):
   ```bash
   python tools/fetch_tsg_spectra.py https://thredds.nci.org.au/thredds/fileServer/rs07/WA/BWD072.zip
   pip install pytsg
   ```
   Files go to `INTERIM_HyLogger_Dataset_Package_GDA2020_DB/TSG/BWD072/`.

Data © Geological Survey of Western Australia, licensed CC BY 4.0.

```
HyLogger/
├── INTERIM_HyLogger_Dataset_Package_GDA2020_DB/   (downloaded, git-ignored)
│   ├── DATABASES/CSV/*.csv
│   ├── MINERAL_LOGS/SM_*.csv
│   └── TSG/BWD072/                                 (optional, step 15)
├── pipeline/       00_ … 16_ scripts, config.py, cluster_names.py, viewer template
├── tools/          fetch_tsg_spectra.py
├── outputs/        tables, run_log.txt, viewer (git-ignored) + plots/
├── run_all.sh
├── requirements.txt
├── README.md
└── FINDINGS.md
```

**Data somewhere else?** `export HYLOGGER_DATA=/path/to/INTERIM_HyLogger_Dataset_Package_GDA2020_DB`
(all paths come from `pipeline/config.py`; scripts stop with a clear message if the data is missing).

## 2. Setup

Python 3.10–3.12 recommended (`ruptures` may have no wheel yet for the newest Python).

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Git Bash)**
```bash
py -3.12 -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
```
A Git "LF will be replaced by CRLF" warning is harmless (`.gitattributes` keeps `run_all.sh` in LF).

## 3. Run

```bash
./run_all.sh 2>&1 | tee outputs/run_log.txt          # macOS / Linux
PYTHON=python ./run_all.sh 2>&1 | tee outputs/run_log.txt   # Windows (Git Bash)
python pipeline/15_raw_spectra_bwd072.py             # optional, needs step 1.3
```

`run_all.sh` runs 01–14 and 16 in order and stops at the first error, so no step uses stale outputs.
Scripts 00 and 00b (hole selection) are run once, before downloading. Step 16 alone takes ~5 minutes on a laptop.

| Step | Script | What it does | Main output |
|---|---|---|---|
| 00 | `select_holes` | From the index: mineral log + > 100 assay rows + > 50 geology rows (522 holes) → Yilgarn gold, 100–700 m → top 12 by data volume, one per project | `STARTER_HOLES_download_list.csv`, `00_selection_bias.csv` |
| 00b | `select_expansion_holes` | 59 more holes: Tier A more Yilgarn gold (≤ 4 per project); Tier B other gold regions, copper, nickel | `EXPANSION_download_list.csv`, `download_mineral_logs.sh` |
| 01 | `load_mineral_logs` | Reads every `SM_*.csv`, skips the metadata header, `N.A.` → missing | `01_mineral_intervals.csv` |
| 02 | `clean_minerals_and_holes` | Blank proportion = not detected (0); excludes holes by rule (see below); drops only minerals never detected; QC flag where SWIR invalid > 50% | `02_minerals_clean.csv`, `02_holes.csv` |
| 03 | `merge_assay_lithology` | Joins WAMEX assays + logged geology to each metre (shortest covering record); removes handheld XRF, > 10 m rows, ppb/ppm gold clashes, holes with implausible gold | `03_merged_intervals.csv` |
| 04 | `exploratory_summary` | QC counts, dominant minerals, per-hole profiles, gold/background counts | `04_*.csv` |
| 05 | `pca_clustering` | PCA + k-means (k 4–10 by silhouette) → mineral assemblages | `05_*.csv` |
| 06 | `within_hole_gold` | Mineral–gold correlation *within* each hole; cluster gold lift with 10 m block-bootstrap CI | `06_*.csv` |
| 07 | `anomaly_detection` | Isolation Forest (top 3%) with the minerals that make each metre unusual; lift vs gold at several cut-offs | `07_*.csv` |
| 08 | `plots` | Strip logs, clusters, PCA, lift chart, cluster heatmap, map | `plots/08*.png` |
| 09 | `interactive_viewer` | Self-contained offline HTML viewer of all holes | `09_hylogger_viewer.html` |
| 10 | `rock_type_from_minerals` | Felsic index from TIR minerals → felsic / intermediate / mafic; cut-offs calibrated leave-one-hole-out; ternary + pseudologs | `10_*.csv`, `plots/10a_ternary.png`, `plots/10b_pseudologs.png` |
| 11 | `boundary_detection` | PELT change-points (split at scanning gaps) → mineral zones; tested against logged contacts vs 5,000 random boundary sets | `11_*.csv`, `plots/11_zones_*.png` |
| 12 | `predictive_models` | Au, As, S, Sb, Cu, MgO from minerals (PLS, Random Forest) on unseen holes and depth blocks; pathfinders vs gold | `12_*.csv`, `plots/12*.png` |
| 13 | `dataset_overview` | Whole index: 2,019 scans of 1,907 holes by region, commodity and data available; map | `13_dataset_overview.csv`, `plots/13*.png` |
| 14 | `endmembers_and_features` | NMF endmembers, number chosen by refit stability; within-hole gold sign test with BH correction; wavelength features | `14_*.csv`, `plots/14*.png` |
| 15 | `raw_spectra_bwd072` | *Optional.* Raw BWD072 spectra via `pytsg`: own hull-quotient white-mica feature, checked against TSG and the 1 m log | `15_bwd072_metre_features.csv`, `plots/15*.png` |
| 16 | `bulk_property_prediction` | 34 elements from minerals, PLS (inner CV) and gradient boosting, GroupKFold by hole and by company | `16_bulk_prediction_scores.csv`, `plots/16_bulk_prediction.png` |

## 4. Design choices & limitations

* **Validation on unseen holes / companies.** Models are always scored on holes (or companies) they never saw, and
  against the simple "predict the training average" baseline. Within-hole scores are reported only as a diagnostic.
* **Gold is tested within holes**, because pooled correlations mostly reflect differences between holes. Only 11 holes
  have enough gold and background metres for this, so gold results are exploratory; multiple tests are BH-corrected.
* **Assay cleaning (03):** handheld XRF rows (Niton/pXRF metadata, or no company sample ID), assay rows > 10 m, gold
  values > 100× another record at the same depth (ppb/ppm clash), and holes with median Au > 1 ppm (recovery %
  stored as grade) are removed. Below detection → half the detection limit.
* **Excluded holes (02), by rule, not by name:** MRDH400/401/402/403/407 (geologist-refined results only — a different
  method — and no TIR) and GBD011_chips (chip scan, not drill core; no TIR). Listed in `outputs/02_excluded_holes.csv`.
* **Scans ≠ holes:** the index lists 2,019 scans of 1,907 holes (re-scans, wedges, chips, cuttings).
* **Clusters overlap** (silhouette ≈ 0.21) — assemblage tendencies, not sharp classes. k is capped at 10 for interpretability.
* **HyLogger minerals are TSA algorithm interpretations**, not measured mineral percentages; 13.7% of metres fail QC.
* **Depths are down-hole.** WAMEX_Survey, WAPIMS_Survey and WAPIMS_Lithology are not used yet (true depth / 3-D, petroleum wells).
* **Selection bias:** the 65 holes have richer data and more gold than typical holes (`00_selection_bias.csv`).

## 5. Extending

Drop more `SM_*.csv` files into `MINERAL_LOGS/` and re-run `run_all.sh`. Hole metadata, assays and lithology are
looked up automatically from the DASC package. For raw spectra of another hole, run `tools/fetch_tsg_spectra.py` with
its NCI zip URL and change `HOLE` in script 15.

## 6. Team & AI use

| Role | Scripts |
|---|---|
| Data engineering | 00, 00b, 01–03, 13, 15, `run_all.sh`, `tools/` |
| Machine learning | 04–07, 16 |
| Geoscience | 10–12, 14 |
| Visualisation | 08, 09, `cluster_names.py` |

Code was developed with AI assistance (Claude) and reviewed step by step by the team and a second AI reviewer; every
number in `FINDINGS.md` comes from running this repository.
