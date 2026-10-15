"""10d - Q4: which REAL rock types (as named in the geology logs) can HyLogger minerals recognise?

Script 10 tests GSWA's example scale (felsic / intermediate / mafic). This script uses the rock
types that are actually present in our 65 holes.

1) Rock-type labels (our rules, explicit words only):
   * the rock-name fields (company lithology name + GSWA rock name) are used first; the geologist's
     comments only when the name fields give no single answer (label source is recorded);
   * a metre matching two different rock types is 'mixed' and not used; company codes are not interpreted;
   * known traps handled: 'cordiorite' (= cordierite, a mineral) is NOT diorite; 'intermediate sediment'
     is a sediment, not an intermediate igneous rock.
2) Which rock types are tested: those present (>= 20 m) in at least MIN_HOLES holes, so that a model
   trained without one hole still has examples from other holes. Rarer types are listed, not tested.
3) Model: random forest on all SWIR + TIR mineral shares + wavelength features, class
   weights balanced. Tested leave-one-HOLE-out and leave-one-COMPANY-out (never sees the test hole /
   company). Baselines: always the most common type; chance = 1 / number of types.
4) Two levels: the named rock type, and 5 composition groups (felsic / intermediate / mafic /
   ultramafic / sedimentary) - what a mineral scanner can physically separate. Pairs such as basalt vs
   dolerite or sandstone vs siltstone differ in grain size / texture, which minerals do not show.
5) Checks: main rock of each hole vs minority (thin) units; an 'unsure' flag when the model's top
   probability is < UNSURE_P, with accuracy for sure vs unsure metres.
6) A model trained on all labelled metres predicts EVERY QC-passed metre (first-pass log), with
   its confidence; it can only answer with the tested rock types.
Outputs: outputs/10d_rock_type_labels.csv (census), outputs/10d_rock_type_scores.csv,
         outputs/10d_rock_type_recall.csv, outputs/10d_rock_type_predictions.csv,
         outputs/plots/10d_rock_types.png
"""
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from config import OUT

warnings.filterwarnings("ignore", category=UserWarning)
PLOTS = OUT / "plots"; PLOTS.mkdir(parents=True, exist_ok=True)
MIN_HOLES, MIN_M_PER_HOLE, UNSURE_P = 4, 20, 0.5

# rock type, composition group, pattern (explicit words only)
ROCKS = [
    ("Granite / granitic",            "Felsic",       r"granit|syenogranit|monzogranit|granodiorit|tonalit|pegmatit|aplite"),
    ("Felsic volcanic",               "Felsic",       r"rhyolit|dacit|felsic volcanic|felsic porphyr"),
    ("Felsic gneiss / granulite",     "Felsic",       r"felsic[- ](?:gneiss|granulite|migmatite)|quartzofeldspathic|quartz-?feldspar"),
    ("Intermediate (diorite, andesite)", "Intermediate", r"(?<!cor)diorit|(?<!\?)andesit|intermediate[- ](?:gneiss|granulite|intrusive|volcanic)"),
    ("Dolerite / gabbro",             "Mafic",        r"dolerit|gabbro|norite"),
    ("Basalt",                        "Mafic",        r"basalt"),
    ("Amphibolite / mafic gneiss",    "Mafic",        r"amphibolite|mafic[- ](?:gneiss|granulite|schist)"),
    ("Ultramafic",                    "Ultramafic",   r"serpentin|pyroxenit|peridotit|komatii|dunite|talc[- ]carbonate|ultramafic"),
    ("Sandstone / conglomerate",      "Sedimentary",  r"sandstone|conglomerat|arkose|greywacke|quartzite|intermediate sediment"),
    ("Siltstone / mudstone / shale",  "Sedimentary",  r"siltstone|mudstone|shale|claystone|argillite|slate|phyllite"),
    ("Carbonate",                     "Sedimentary",  r"dolomite|limestone|marble"),
    ("BIF / chert",                   "Sedimentary",  r"\bbif\b|banded iron|chert|jaspilite"),
]
GROUP = {r: g for r, g, _ in ROCKS}

def rock_of(text):
    text = text.replace("intermediate sediment", "sediment-intermediate")     # a sediment, not an igneous rock
    hits = []
    for r, _, p in ROCKS:
        pat = p.replace("|intermediate sediment", "")
        if re.search(pat, text): hits.append(r)
    if "sediment-intermediate" in text and not hits: return "Sandstone / conglomerate" if "sand" in text else None
    return hits[0] if len(hits) == 1 else ("mixed" if hits else None)

d = pd.read_csv(OUT / "10_rock_type_intervals.csv", low_memory=False)    # QC-passed metres (from 05 via 10)
name = d[["Lith_name", "GSWA_rock"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower().map(rock_of)
comm = d["Geology_comments"].fillna("").astype(str).str.lower().map(rock_of)
use_name = name.notna() & (name != "mixed")
d["Rock"] = np.where(use_name, name, comm)
d["Rock_source"] = np.where(use_name, "rock name", np.where(comm.notna() & (comm != "mixed"), "comment", ""))
d.loc[d["Rock"] == "mixed", "Rock"] = None

# ---- census + which rock types can be tested ----
lab_all = d[d["Rock"].notna()]
per_hole = lab_all.groupby(["Rock", "Hole"]).size()
cen = pd.DataFrame({"group": pd.Series(GROUP),
                    "metres": lab_all["Rock"].value_counts(),
                    "holes": lab_all.groupby("Rock")["Hole"].nunique(),
                    f"holes_with_{MIN_M_PER_HOLE}m": (per_hole >= MIN_M_PER_HOLE).groupby(level=0).sum(),
                    "from_comments_share": lab_all.groupby("Rock")["Rock_source"].apply(lambda s: (s == "comment").mean()).round(2)}).fillna(0)
cen["tested"] = cen[f"holes_with_{MIN_M_PER_HOLE}m"] >= MIN_HOLES
cen = cen.sort_values("metres", ascending=False)
cen.rename_axis("rock_type").to_csv(OUT / "10d_rock_type_labels.csv")
print(f"QC metres: {len(d):,} | with a single named rock type: {len(lab_all):,} | no usable name or mixed: {len(d) - len(lab_all):,}")
print(cen.to_string())
TYPES = [r for r in cen.index if cen.at[r, "tested"]]
print(f"\nTested: {len(TYPES)} rock types = {cen.loc[TYPES, 'metres'].sum() / cen['metres'].sum():.1%} of named metres."
      f" Not tested (too few holes): {[r for r in cen.index if not cen.at[r, 'tested']]}")

# ---- model ----
MINERALS = [c for c in d.columns if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c]
WAVE = [c for c in ["VIS-SWIR_Albedo", "900D_Fe", "900W_Fe", "1900D_Water", "2200D_AlOH", "2200W_AlOH", "2250D_FeOH",
                    "2250W_FeOH", "8625D_Quartz", "11300D_Carbonate", "11300W_Carbonate"] if c in d.columns]
FEATS = MINERALS + WAVE
lab = d[d["Rock"].isin(TYPES)].reset_index(drop=True)
company = pd.read_csv(OUT / "02_holes.csv").set_index("Hole")["Company"]
lab["Company_grp"] = lab["Hole"].map(company).fillna("hole:" + lab["Hole"])
y = lab["Rock"].to_numpy()
def model():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=0)

def cv_predict(groups):
    pred, conf = np.empty(len(y), dtype=object), np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(lab, y, groups):
        med = lab.iloc[tr][FEATS].median()
        m = model().fit(lab.iloc[tr][FEATS].fillna(med), y[tr])
        pr = m.predict_proba(lab.iloc[te][FEATS].fillna(med))
        pred[te], conf[te] = m.classes_[pr.argmax(1)], pr.max(1)
    return pred, conf

lab["Pred_new_hole"], lab["Conf_new_hole"] = cv_predict(lab["Hole"].to_numpy())
lab["Pred_new_company"], _ = cv_predict(lab["Company_grp"].to_numpy())

rows = []
def score(name, pred, truth, labels):
    r = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
    rows.append({"level": "rock type" if labels is TYPES else "composition group", "method": name, "n": len(truth),
                 "accuracy": round(np.mean(pred == truth), 3), "balanced_accuracy": round(balanced_accuracy_score(truth, pred), 3)})
    print(f"  {name:44s} accuracy {np.mean(pred == truth):4.0%} | balanced {balanced_accuracy_score(truth, pred):4.0%}")
    return r

print(f"\nLevel 1 - {len(TYPES)} rock types ({len(lab):,} m, {lab.Hole.nunique()} holes, {lab.Company_grp.nunique()} companies):")
major = pd.Series(y).value_counts().index[0]
score(f"Always '{major}' (most common)", np.full(len(y), major), y, TYPES)
print(f"  {'Chance':44s} balanced {1 / len(TYPES):4.0%}")
rec_hole = score("Random forest - new HOLE", lab["Pred_new_hole"].to_numpy(), y, TYPES)
rec_co = score("Random forest - new COMPANY", lab["Pred_new_company"].to_numpy(), y, TYPES)
cm = pd.DataFrame(confusion_matrix(y, lab["Pred_new_hole"], labels=TYPES), index=TYPES, columns=TYPES)
recall = pd.DataFrame({"group": [GROUP[t] for t in TYPES], "metres": [int((y == t).sum()) for t in TYPES],
                       "holes": [lab.loc[lab.Rock == t, "Hole"].nunique() for t in TYPES],
                       "recall_new_hole": np.round(rec_hole, 3), "recall_new_company": np.round(rec_co, 3),
                       "most_confused_with": [cm.loc[t].drop(t).idxmax() if cm.loc[t].drop(t).sum() else "" for t in TYPES]},
                      index=pd.Index(TYPES, name="rock_type"))
print(recall.to_string())

GROUPS = ["Felsic", "Intermediate", "Mafic", "Ultramafic", "Sedimentary"]
yg = lab["Rock"].map(GROUP).to_numpy()
print("\nLevel 2 - 5 composition groups (same predictions, grouped):")
score(f"Always '{pd.Series(yg).value_counts().index[0]}' (most common)", np.full(len(yg), pd.Series(yg).value_counts().index[0]), yg, GROUPS)
rg = score("Random forest - new HOLE", lab["Pred_new_hole"].map(GROUP).to_numpy(), yg, GROUPS)
rgc = score("Random forest - new COMPANY", lab["Pred_new_company"].map(GROUP).to_numpy(), yg, GROUPS)
print("  recall per group (new hole):", {g: f"{v:.0%}" for g, v in zip(GROUPS, rg)})
print("  recall per group (new company):", {g: f"{v:.0%}" for g, v in zip(GROUPS, rgc)})

# ---- checks ----
ok = lab["Pred_new_hole"] == lab["Rock"]
main = lab["Rock"] == lab["Hole"].map(lab.groupby("Hole")["Rock"].agg(lambda s: s.mode()[0]))
print(f"\nChecks (new hole): hole's main rock {ok[main].mean():.0%} (n={main.sum():,}) | minority / thin units {ok[~main].mean():.0%} (n={(~main).sum():,})")
sure = lab["Conf_new_hole"] >= UNSURE_P
print(f"  'unsure' flag (top probability < {UNSURE_P}): {(~sure).mean():.0%} of metres flagged; accuracy sure {ok[sure].mean():.0%} vs unsure {ok[~sure].mean():.0%}")
src = lab["Rock_source"] == "comment"
print(f"  label from rock name {ok[~src].mean():.0%} (n={(~src).sum():,}) | label from comments {ok[src].mean():.0%} (n={src.sum():,})")

pd.DataFrame(rows).to_csv(OUT / "10d_rock_type_scores.csv", index=False)
recall.to_csv(OUT / "10d_rock_type_recall.csv")

# ---- first-pass log for every QC metre ----
med = lab[FEATS].median()
final = model().fit(lab[FEATS].fillna(med), y)
pr = final.predict_proba(d[FEATS].fillna(med))
out = d[["Hole", "From_m", "To_m", "Rock", "Rock_source"]].rename(columns={"Rock": "Logged_rock"})
out["Predicted_rock"] = final.classes_[pr.argmax(1)]
out["Predicted_group"] = out["Predicted_rock"].map(GROUP)
out["Confidence"] = pr.max(1).round(2)
out["Unsure"] = out["Confidence"] < UNSURE_P
out = out.merge(lab[["Hole", "From_m", "Pred_new_hole", "Pred_new_company"]], on=["Hole", "From_m"], how="left")
out.to_csv(OUT / "10d_rock_type_predictions.csv", index=False)
print(f"\nFirst-pass log: {len(out):,} metres predicted ({out['Unsure'].mean():.0%} flagged unsure). "
      "Logged metres also carry their held-out prediction (Pred_new_hole).")

# ---- figure ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.25]})
order = recall.sort_values("recall_new_hole").index
yy = np.arange(len(order))
a1.barh(yy + 0.2, recall.loc[order, "recall_new_hole"], 0.4, color="#2B6CB0", label="new hole")
a1.barh(yy - 0.2, recall.loc[order, "recall_new_company"], 0.4, color="#B5652B", label="new company")
a1.axvline(1 / len(TYPES), ls=":", color="grey"); a1.text(1 / len(TYPES) + 0.01, -0.9, "chance", fontsize=8, color="grey")
a1.set_yticks(yy, [f"{t}  ({recall.at[t, 'metres']:,} m, {recall.at[t, 'holes']} holes)" for t in order], fontsize=8)
a1.set_xlim(0, 1); a1.set_xlabel("share of logged metres recognised (recall)"); a1.legend(fontsize=8, loc="lower right")
a1.set_title("Recognised on unseen holes, per rock type", loc="left", fontsize=10)
cmn = cm.div(cm.sum(axis=1), axis=0)
a2.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
short = [t.split(" (")[0].split(" /")[0] for t in TYPES]
a2.set_xticks(range(len(TYPES)), short, rotation=45, ha="right", fontsize=8); a2.set_yticks(range(len(TYPES)), short, fontsize=8)
for i in range(len(TYPES)):
    for j in range(len(TYPES)):
        if cmn.values[i, j] >= 0.05:
            a2.text(j, i, f"{cmn.values[i, j]:.0%}", ha="center", va="center", fontsize=7, color="white" if cmn.values[i, j] > 0.5 else "black")
a2.set_xlabel("predicted from minerals"); a2.set_ylabel("logged by geologist")
a2.set_title("Where the mistakes go (rows sum to 100%)", loc="left", fontsize=10)
acc = np.mean(lab["Pred_new_hole"] == lab["Rock"]); accg = np.mean(lab["Pred_new_hole"].map(GROUP) == yg)
fig.suptitle(f"Q4: {len(TYPES)} logged rock types from HyLogger minerals - {acc:.0%} exact, {accg:.0%} correct composition group "
             f"(unseen holes, {len(lab):,} m)", x=0.01, ha="left", weight="bold")
fig.tight_layout(); fig.savefig(PLOTS / "10d_rock_types.png", dpi=150); plt.close(fig)
print("Saved: 10d_rock_type_labels.csv, 10d_rock_type_scores.csv, 10d_rock_type_recall.csv, 10d_rock_type_predictions.csv, plots/10d_rock_types.png")
