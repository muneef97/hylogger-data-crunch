# Findings — GSWA HyLogger Data Crunch (Theme 1)

**Scope.** 65 drill holes (gold, copper and nickel projects across WA), 29,003 one-metre
intervals, of which 25,044 pass spectral quality control. HyLogger mineralogy is joined to company assays (WAMEX)
and logged geology. One hole (BWD072) was also analysed from its **raw spectra** (76,160 spectra, read without TSG).
The 65 holes are 3.4% of the 1,907 holes in the HyLogger index and were chosen *for rich data, not at random*
(`outputs/00_selection_bias.csv`: e.g. median share of gold-grade assays 9.3% vs 1.4% for other holes).

All numbers below are produced by `bash run_all.sh` (plus optional step 15) and can be checked in `outputs/run_log.txt`.

**How to read the gold results.** Gold tests are *within-hole* (each hole compared with itself), because holes
differ in grade, company, lab and logging style. Only 10 of the 65 holes have enough gold and background intervals
for that test, so every gold result here is **exploratory** and needs more holes to confirm.

---

## Headline results

| GSWA question | What we did | Result (65 holes) | Confidence |
|---|---|---|---|
| **1. Features of interest** (anomalies, endmembers in bulk) | NMF endmembers (14), Isolation Forest anomalies (07), PCA + k-means assemblages (05) | 7 mineral endmembers; 4 are very stable (recovered in 19–20 of 20 refits). Anomalies flag unusual mineralogy but are **not** linked to gold. | Endmembers: good. Anomaly–gold link: rejected. |
| **2. Feature analysis outside TSG** | Raw spectra of BWD072 read with `pytsg`; our own hull-quotient white-mica feature (15) | Feature found in 82% of good spectra; matches TSG's own feature (position r = 0.90, median difference 1.1 nm; depth r = 0.99). Mineral order (paragonite < muscovite < phengite) reproduced. | Good (method validated on one hole). |
| **3. Boundary detection** | PELT change-points on mineralogy, split at scanning gaps (11) | 74% of our boundaries fall within ±3 m of a logged contact vs 43% for random boundaries (p = 0.0002); better than random in 22 of 23 holes. | Strong. |
| **4. Rock type from minerals** | Felsic index from minerals, cut-offs calibrated leave-one-hole-out; ternary + pseudologs (10) | 61% of held-out metres match the logged broad class (felsic / intermediate / mafic); felsic 85%, mafic 56%, intermediate 20%. | Moderate; intermediate is hard. |
| **5. Other parameters in bulk** | PLS and gradient boosting from mineralogy to 34 assay elements, tested on unseen holes and unseen companies (16) | 11 elements predictable even for an unseen company — best **MgO** (skill 0.75–0.76, median error 33%). Gold, Cu, As, S and other ore-related elements are **not** predictable. | Strong for MgO and rock-forming elements; none for gold. |

"Skill" = 1 − (model error / error of simply predicting the training average). 0 = no better than the average; 1 = perfect.

---

## 1. Features of interest

**Mineral endmembers (script 14).** Non-negative matrix factorisation on 25,044 metres split the mineralogy into
seven candidate endmembers. The number seven was chosen *by stability*: it is the largest number for which every
endmember is recovered when the model is refitted on half of the holes (median cosine ≥ 0.9; eight gives 0.50).

| Endmember | Share of metres | Recovered in refits | Within-hole gold (holes +/−) |
|---|---|---|---|
| E1 white mica + silica | 23.4% | 20/20 | 6 / 0 (p 0.031, BH q 0.22) |
| E2 chlorite | 19.9% | 20/20 | 1 / 6 (q 0.42) |
| E3 silica / dark mica / kaolin | 17.4% | 15/20 | — |
| E4 plagioclase / dark mica / epidote | 13.3% | 19/20 | — |
| E5 amphibole | 12.8% | 20/20 | 2 / 7 (q 0.42) |
| E6 carbonate | 8.6% | 15/20 | 6 / 2 (q 0.51) |
| E7 Mg-OH | 4.6% | 16/20 | — |

The white-mica/silica endmember was positively related to gold in all six eligible holes where the relationship was
clear, but this is **not significant after correcting for testing seven endmembers** (Benjamini–Hochberg q = 0.22).

**Assemblage clusters (script 05).** k-means on PCA scores gives 10 assemblages (silhouette 0.21: groups overlap,
as expected for gradational rocks; k capped at 10 for interpretability). Names are generated from the data
(`pipeline/cluster_names.py`).

**Anomalies (script 07).** Isolation Forest flags the most unusual 3% of metres and explains which minerals drive
each one. On 11 holes anomalous metres had ~2× the gold rate; **on 65 holes the lift is 0.54 — the link did not
replicate**, so we no longer claim it. Anomalies remain useful as a "look here" list for unusual mineralogy and for
data-quality problems.

## 2. Feature analysis outside TSG (BWD072, script 15)

* Read 76,160 SWIR spectra (380–2500 nm, ~156 per metre) directly from the `.tsg/.bip` files; TSG's Final Mask keeps 62,291.
* Our own method: hull-quotient, then the deepest local minimum in 2180–2235 nm, refined with a parabola; rejected if at
  the window edge or shallower than 0.02. Valid in 51,322 spectra (82%).
* **Check against TSG** (same window): position r = 0.90, median difference 1.1 nm; depth r = 0.99. Mean position by
  TSG mineral call: paragonite 2194.0, muscovite 2203.0, phengite 2212.9 nm — the expected order.
* Against the 1 m summary log: Spearman 0.66 (offset +2.5 nm); white-mica share 0.81.
* Gold metres in BWD072 have far more white mica (68% vs 11% of spectra) and a shorter wavelength (−5.5 nm, 90% CI
  −8.6 to −0.7). Using white-mica-only spectra the shift is −4.6 nm (−8.0 to +0.7): **a change in white-mica
  chemistry with gold is suggested but not established.** One hole only.

## 3. Boundary detection (script 11)

* PELT finds 236 boundaries in 47 holes. Validated where possible: 166 boundaries in 23 holes with ≥ 3 logged contacts.
* 122 of 166 (74%) are within ±3 m of a logged lithology contact, vs 43% (70.9) for random boundaries drawn from the
  same eligible logged positions with the same spacing (5,000 draws, p = 0.0002).
* Better than random in 22 of 23 holes; individually significant (p < 0.05) in 6.
* Use: pre-mark likely contacts for a geologist to confirm, and find contacts that were never logged.

## 4. Rock type from minerals (script 10)

* A felsic index from the mineral mix gives a broad class: felsic / intermediate / mafic. Ternary plot (`10a`) and
  pseudologs next to the logged geology (`10b`) show both small (metre) and large (whole-hole) scale.
* Cut-offs calibrated leaving one hole out at a time: **61% agreement on held-out metres** (5,066 metres, 43 holes;
  54% with fixed cut-offs). Mean per hole 43% — some holes agree well, others poorly.
* Recall: felsic 85%, mafic 56%, intermediate 20%. Suggested cut-offs for new holes: felsic ≥ 0.90, intermediate ≥ 0.80.
* Mafic rock called felsic has much higher TIR silica; veining is a plausible cause but is **not confirmed**
  (vein mentions in the log are similar).

## 5. Other parameters in bulk (scripts 16, 12)

* 34 assay elements (each tested on the metres where it was assayed). Tested on **unseen holes** and on **unseen companies** (GroupKFold, 10 folds),
  so the score shows how it would work on a new company's drilling.
* Skill ≥ 0.3 for 15 of 34 elements by hole and 11 of 34 by company (PLS); boosting 16 and 14.
* **Consistently predictable:** MgO, Rb, Cr, Ni, K₂O, Ba, Na₂O, Ga, Al, Zr, Co — the rock-forming and alteration
  chemistry minerals should carry. **MgO: skill 0.75 (new hole), 0.76 (new company), median relative error 33%.**
* **Not predictable:** Au, Cu, As, S, Sb, Bi, Mo, W, U, Th, Y, FeO (V and Sc borderline). Gold classifier ROC AUC 0.50.
* Within holes, gold goes with its usual pathfinders (As 5/6 holes positive, S 5/6, Cu 6/7, Sb 4/7) — chemistry
  works as a vector where mineralogy alone does not.

---

## Data-quality discoveries (useful to GSWA)

The flat assay table merges several lab methods into one column per element and does not say which method each row
came from (the metadata lists methods per hole, not per row). Script 03 detects and handles each case below and
prints what it removed; none of them is in GSWA's own `meta_data_quality` table in the SQLite release.

* **Handheld XRF rows mixed into lab assays** (12DHDD0001/0002: 408 rows; metadata method "Niton"). Au up to 257 ppm
  where fire assay in the same holes peaks at 0.17 ppm. Rows removed.
* **A second, non-fire-assay "Au" for the same samples** (KBD089: 801 values, KBD094: 146). Whole numbers 1–35 that do
  not track fire assay at the same depths (Spearman −0.03 over 780 depths); method not recorded. Ignored; fire assay kept.
* **Leach recovery percentages stored as gold grade** (18KLDD008: values 50–100 below 330 m, median 75.85; metadata
  lists LeachWELL methods). Gold set to missing for that hole.
* **−5000 used as a "no value" code** (ETG0202/0204/0226: 1,579 rows, almost every element). Treating it as
  "below detection" would have created 2,500 ppm values; now set to missing (with −999 and −9999).
* **A whole-hole assay row** (ABND198, 0–308.8 m with As 31%) — removed.
* **Physically impossible values remain in a few places** (e.g. KPDDH003 S up to 99.8%, above pure pyrite at ~53%).
  They do not affect our results (S is not predictable) but are worth a check by GSWA.
* **The index counts scans, not holes:** 2,019 scans of 1,907 holes (re-scans, wedges, cuttings and chips).
* Several holes have mostly low-confidence spectra (e.g. YNC350, 97%); 13.7% of all metres fail QC.

## Business value (no dollar figures claimed)

* **Chemistry where there are no assays:** MgO and ~10 rock-forming elements estimated from existing HyLogger scans.
  Value = metres without assays × lab cost per sample avoided or prioritised.
* **Faster logging:** pre-marked contacts and pseudologs for geologists to confirm rather than start from scratch.
* **Alteration screening:** white-mica vs chlorite as a first-pass vector in new gold holes.
* **Cleaner public data:** the checks above can run across GSWA's whole database.
* **Interactive explorer:** `app/streamlit_app.py` (Streamlit) shows every hole's minerals, endmembers, boundaries,
  rock type and assays side by side, reading only the pipeline outputs.
* **Open and scalable:** plain Python, any `SM_<hole>.csv`, runs in minutes; raw-spectra method needs no TSG licence.

## Gold: an alteration guide, not a gold predictor (exploratory)

Gold is not one of GSWA's five questions, but it is the first question most users ask, so we record the evidence here.


| Signal | Evidence | Status |
|---|---|---|
| More white mica (+ silica) near gold | within-hole: SWIR white mica 7 of 10 holes positive (median ρ 0.22); endmember E1 6/0; assemblage "White-Mica-rich, high Silica" gold lift 1.45 (90% CI 1.30–1.59, 18 holes) | Consistent but not significant after correction |
| Less chlorite / amphibole near gold | TIR chlorite median ρ −0.21 (6 of 10 holes negative); chlorite assemblage lift 0.81 (0.71–0.93); amphibole/epidote assemblage lift 0.32 (0.10–0.61) | Consistent, exploratory |
| Shorter white-mica wavelength near gold | BWD072 raw spectra −5.5 nm; white-mica-only −4.6 nm (CI crosses 0) | One hole; not established |
| Anomalies = gold | 11 holes: lift 1.97 → 65 holes: 0.54 | **Rejected** |
| Predict gold grade from minerals | skill ≈ 0, AUC 0.50 | **No** |

This is the classic sericite (white mica) – chlorite alteration pattern of orogenic gold. HyLogger shows it
consistently, but it is a guide to *alteration*, not a gold predictor.

## What changed from 11 to 65 holes

| Result | 11 holes | 65 holes | |
|---|---|---|---|
| Boundaries near logged contacts | yes | 74% vs 43%, p = 0.0002 | replicated, stronger |
| White mica up / chlorite down near gold | yes | yes | replicated |
| Anomaly → gold | lift 1.97 | lift 0.54 | **not replicated** |

We report the failed replication on purpose: more data overturned an attractive early result.

Re-running after the assay cleaning in script 03 (which removed false gold in KBD089 and KBD094: gold metres 489 → 392)
left every headline result unchanged and slightly strengthened the white-mica / chlorite pattern.

## Limitations

* 65 holes, chosen for rich data; 10 holes for the gold tests. Results may not hold for typical holes.
* HyLogger minerals are TSA algorithm interpretations, not measured mineral percentages.
* Depths are along the hole (down-hole), not true vertical depth; survey tables (WAMEX_Survey, WAPIMS_Survey) and
  WAPIMS_Lithology were not used.
* Raw-spectra analysis is one hole (BWD072).
* Gold associations are not significant after multiple-testing correction.

## Next steps

1. Run on the 1,115 holes that have a mineral log, assays and a geology log (the pipeline is ready; download is the limit).
   A team linkage audit of all 1,590 indexed WAMEX holes with a mineral log found 87% link to a collar, 81% to a
   geology log and 76% to assays (primary or alternate CollarId), so most of the archive can enter this pipeline.
   17 of our 65 holes also have an alternate CollarId that we have not yet used.
2. Raw-spectra white-mica wavelength on more gold holes to test the −5 nm shift.
3. Add survey data for true depth and 3-D maps; use WAPIMS lithology for petroleum wells.
4. Give geologists the boundary and pseudolog outputs to test in real logging.
