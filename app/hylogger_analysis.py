"""
HyLogger extra analyses for the GSWA Data Crunch questions.

One module, used by both the notebook and the Streamlit app, so the numbers match everywhere.

Q1  endmembers + explained anomalies  -> nmf_endmembers(), anomaly_reasons()
Q2  features beyond mineral labels    -> FEATURE_FAMILIES, wavelength_by_group()
Q3  boundary detection                -> detect_boundaries(), score_boundaries()
Q4  rock type from minerals           -> rocktype_loho(), ternary_table()
Q5  other properties (geochemistry)   -> pls_loho()

All validation is leave-one-hole-out (LOHO): a model is always tested on a hole it never saw.
Everything here is exploratory and uses only the TSG summary logs (no raw spectra).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import kruskal, spearmanr
except Exception:  # scipy ships with scikit-learn, but fail soft
    kruskal = spearmanr = None


# ----------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------
def numeric_matrix(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Numeric feature matrix. In HyLogger logs 'N.A.' means 'not detected', so missing -> 0."""
    return df[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def pretty_feature(col: str) -> str:
    """'Kaolin_(System_SWIR)' -> 'Kaolin (SWIR)'."""
    s = str(col)
    s = s.replace("_(System_SWIR)", " (SWIR)").replace("_(System_TIR)", " (TIR)")
    return s.replace("_", " ")


def mineral_fraction_columns(df: pd.DataFrame):
    """Mineral-abundance columns from the TSG system classification (excluding the Invalid fraction)."""
    return [c for c in df.columns
            if (str(c).endswith("(System_SWIR)") or str(c).endswith("(System_TIR)")) and "invalid" not in str(c).lower()]


ROCK_GROUP_RULES = [
    ("Mafic (amphibolite, basalt, dolerite, gabbro)", r"amphibol|basalt|dolerite|gabbro|mafic"),
    ("Siltstone & claystone", r"^(siltstone|claystone)"),
    ("Sandstone & conglomerate", r"sandstone|conglomerate"),
    ("Carbonate (dolomite)", r"^dolomite$|limestone|^carbonate"),
    ("Felsic-intermediate & schist", r"quartzofeldspathic|micaceous|schist|phyllite|slate|granulite|granofels|hornfels|granit|diorite|trachyte|gneiss|felsic"),
    ("Regolith", r"regolith|pisolite|laterite|saprolite"),
]


def rock_group(name) -> str | None:
    """Broad rock group from the GSWA rock name. Veins / unknown / undivided sediment -> None (excluded)."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip().lower()
    for label, pattern in ROCK_GROUP_RULES:
        if re.search(pattern, s):
            return label
    return None


def add_rock_group(df: pd.DataFrame, rock_col: str = "Geo_GSWARockName") -> pd.DataFrame:
    out = df.copy()
    out["Rock_Group"] = out[rock_col].map(rock_group) if rock_col in out else None
    return out


# ----------------------------------------------------------------------------------------------
# Q3  Boundary detection
# ----------------------------------------------------------------------------------------------
def _prefix_sums(Z):
    S = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, axis=0)])
    T = np.concatenate([[0.0], np.cumsum((Z ** 2).sum(axis=1))])
    return S, T


def segment_series(Z: np.ndarray, penalty: float, min_size: int = 3):
    """Optimal partitioning of a multivariate series into constant-mean segments.

    Minimises  sum(within-segment squared error) + penalty * (number of segments).
    Exact dynamic programme (O(n^2) with prefix sums), so no extra library is needed.
    Returns the start indices of every segment after the first (the change points).
    """
    n = len(Z)
    if n < 2 * min_size:
        return []
    S, T = _prefix_sums(Z)
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    last = np.zeros(n + 1, dtype=int)
    for j in range(min_size, n + 1):
        i = np.arange(0, j - min_size + 1)
        i = i[(i == 0) | (i >= min_size)]
        seg = j - i
        cost = (T[j] - T[i]) - ((S[j] - S[i]) ** 2).sum(axis=1) / seg
        total = F[i] + cost + penalty
        k = int(np.argmin(total))
        F[j] = total[k]
        last[j] = i[k]
    cps, j = [], n
    while j > 0:
        i = int(last[j])
        if i > 0:
            cps.append(i)
        j = i
    return sorted(cps)


def _noise_variance(Z: np.ndarray) -> float:
    """Robust point-to-point noise level (sum over components), from MAD of first differences."""
    if len(Z) < 4:
        return float(np.var(Z, axis=0).sum()) or 1.0
    d = np.diff(Z, axis=0)
    mad = np.median(np.abs(d - np.median(d, axis=0)), axis=0) * 1.4826
    s2 = float(((mad ** 2) / 2.0).sum())
    return s2 if s2 > 1e-9 else float(np.var(Z, axis=0).sum()) or 1.0


def _smooth_rows(Z: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(Z) < window:
        return Z
    return pd.DataFrame(Z).rolling(window, center=True, min_periods=1).median().to_numpy()


def logged_contacts(hole_df: pd.DataFrame, level: str = "group", rock_col: str = "Geo_GSWARockName",
                    max_gap_m: float = 1.0) -> np.ndarray:
    """Depths where the *logged* geology changes (rock name, or broad rock group)."""
    need = {"Geo_FromDepth", "Geo_ToDepth", rock_col}
    if not need.issubset(hole_df.columns):
        return np.array([])
    g = (hole_df.loc[hole_df["Geo_Match"].fillna(False).astype(bool), ["Geo_FromDepth", "Geo_ToDepth", rock_col]]
         .dropna(subset=["Geo_FromDepth", rock_col]).drop_duplicates().sort_values("Geo_FromDepth"))
    if len(g) < 2:
        return np.array([])
    label = g[rock_col].map(rock_group) if level == "group" else g[rock_col].astype(str)
    label = label.where(label.notna(), "__none__").to_numpy()
    frm, to = g["Geo_FromDepth"].to_numpy(float), g["Geo_ToDepth"].to_numpy(float)
    out = []
    for a in range(len(g) - 1):
        if label[a] != label[a + 1] and "__none__" not in (label[a], label[a + 1]) and abs(frm[a + 1] - to[a]) <= max_gap_m:
            out.append(frm[a + 1])
    return np.array(sorted(set(out)))


def detect_boundaries(df: pd.DataFrame, feats, penalty_scale: float = 8.0, min_zone_intervals: int = 5,
                      smooth_window: int = 5, variance_kept: float = 0.90, max_gap_m: float = 5.0):
    """Detect stratigraphic-style zone boundaries from the mineral signature down each hole.

    Method: standardise the (N.A.->0) mineral features, keep the PCA components explaining
    `variance_kept` of the variance, smooth lightly, then split each hole into zones with
    optimal partitioning. Higher `penalty_scale` -> fewer, more confident boundaries.
    A gap in the log larger than `max_gap_m` also starts a new zone (nothing was scanned there).

    Returns (boundaries_df, zones_df).
    """
    d = df[df["Depth_QC_Valid"].fillna(False).astype(bool)].copy() if "Depth_QC_Valid" in df else df.copy()
    X = numeric_matrix(d, feats)
    keep = X.columns[X.std() > 0]
    scaler = StandardScaler().fit(X[keep])
    Zall = scaler.transform(X[keep])
    pca = PCA(n_components=variance_kept, svd_solver="full").fit(Zall)
    scores = pca.transform(Zall)
    d["_row"] = np.arange(len(d))

    b_rows, z_rows = [], []
    for hole, g in d.groupby(d["Hole_name"].astype(str)):
        g = g.sort_values("Depth_Mid")
        Z = _smooth_rows(scores[g["_row"].to_numpy()], smooth_window)
        n = len(g)
        s2 = float(Z.var(axis=0).sum()) or 1.0          # total variance of this hole's signal
        penalty = penalty_scale * s2 * np.log(max(n, 2))
        cps = set(segment_series(Z, penalty, min_zone_intervals))
        frm, to = g["Mineral_FromDepth"].to_numpy(float), g["Mineral_ToDepth"].to_numpy(float)
        gaps = {i for i in range(1, n) if frm[i] - to[i - 1] > max_gap_m}
        starts = [0] + sorted(cps | gaps)
        ends = starts[1:] + [n]
        for zi, (a, b) in enumerate(zip(starts, ends), 1):
            seg = g.iloc[a:b]
            z_rows.append({"Hole_name": hole, "Zone": zi, "From_depth": float(frm[a]), "To_depth": float(to[b - 1]),
                           "Intervals": b - a, "From_index": a, "To_index": b})
            if a > 0:
                left, right = Z[max(0, a - min_zone_intervals):a].mean(axis=0), Z[a:min(n, a + min_zone_intervals)].mean(axis=0)
                b_rows.append({"Hole_name": hole, "Boundary_depth": float(frm[a]),
                               "Contrast": float(np.linalg.norm(left - right) / np.sqrt(s2)),
                               "Reason": "log gap" if a in gaps and a not in cps else "mineral change"})
    return pd.DataFrame(b_rows), pd.DataFrame(z_rows)


def describe_zones(df: pd.DataFrame, zones: pd.DataFrame, mineral_cols, top_n: int = 3) -> pd.DataFrame:
    """Add dominant minerals, dominant logged rock and cluster to each zone."""
    d = df[df["Depth_QC_Valid"].fillna(False).astype(bool)].copy() if "Depth_QC_Valid" in df else df.copy()
    rows = []
    for hole, zg in zones.groupby("Hole_name"):
        g = d[d["Hole_name"].astype(str) == hole].sort_values("Depth_Mid").reset_index(drop=True)
        for _, z in zg.iterrows():
            seg = g.iloc[int(z["From_index"]):int(z["To_index"])]
            means = numeric_matrix(seg, mineral_cols).mean().sort_values(ascending=False)
            minerals = ", ".join(f"{pretty_feature(k)} {100*v:.0f}%" for k, v in means.head(top_n).items() if v > 0.02) or "none detected"
            rock = seg["Geo_GSWARockName"].dropna().value_counts() if "Geo_GSWARockName" in seg else pd.Series(dtype=int)
            cl = seg["Cluster"].dropna() if "Cluster" in seg else pd.Series(dtype=float)
            rows.append({**z.drop(["From_index", "To_index"]).to_dict(), "Dominant_minerals": minerals,
                         "Dominant_logged_rock": rock.index[0] if len(rock) else "no lithology matched",
                         "Dominant_cluster": f"Cluster {int(cl.mode().iloc[0])}" if len(cl) else "n/a"})
    return pd.DataFrame(rows)


def score_boundaries(df: pd.DataFrame, boundaries: pd.DataFrame, tol_m: float = 5.0, level: str = "group",
                     n_random: int = 500, seed: int = 0) -> dict:
    """Do detected boundaries land on logged geological contacts more often than chance?

    recall    = share of logged contacts that have a detected boundary within +/- tol_m
    precision = share of detected boundaries that have a logged contact within +/- tol_m
    The chance baseline places the same number of boundaries at random depths in each hole.
    """
    rng = np.random.default_rng(seed)
    per_hole = []
    for hole, g in df.groupby(df["Hole_name"].astype(str)):
        lo, hi = float(g["Mineral_FromDepth"].min()), float(g["Mineral_ToDepth"].max())
        contacts = logged_contacts(g, level=level)
        contacts = contacts[(contacts >= lo) & (contacts <= hi)]
        det = boundaries.loc[boundaries["Hole_name"] == hole, "Boundary_depth"].to_numpy(float) if len(boundaries) else np.array([])
        per_hole.append((hole, lo, hi, contacts, det))

    def counts(det, contacts):
        if len(contacts) == 0 or len(det) == 0:
            return 0, 0
        rec = int((np.abs(contacts[:, None] - det[None, :]).min(axis=1) <= tol_m).sum())
        prec = int((np.abs(det[:, None] - contacts[None, :]).min(axis=1) <= tol_m).sum())
        return rec, prec

    def pooled(draw=None):
        rec = prec = nc = nd = 0
        for hole, lo, hi, contacts, det in per_hole:
            if draw is not None:
                det = np.sort(rng.uniform(lo, hi, size=len(det)))
            r, p = counts(det, contacts)
            rec += r; prec += p; nc += len(contacts); nd += len(det)
        recall = rec / nc if nc else np.nan
        precision = prec / nd if nd else np.nan
        f1 = 2 * recall * precision / (recall + precision) if recall and precision and (recall + precision) else 0.0
        return recall, precision, f1, nc, nd

    r, p, f1, nc, nd = pooled()
    sims = np.array([pooled(draw=True)[:3] for _ in range(n_random)])
    return {"tolerance_m": tol_m, "contact_level": level, "logged_contacts": nc, "detected_boundaries": nd,
            "recall": r, "precision": p, "f1": f1,
            "chance_recall": float(np.nanmean(sims[:, 0])), "chance_precision": float(np.nanmean(sims[:, 1])),
            "chance_f1": float(np.nanmean(sims[:, 2])),
            "p_value_f1": float((np.sum(sims[:, 2] >= f1) + 1) / (n_random + 1)),
            "lift_precision": float(p / np.nanmean(sims[:, 1])) if np.nanmean(sims[:, 1]) else np.nan}


# ----------------------------------------------------------------------------------------------
# Q4  Rock type from minerals
# ----------------------------------------------------------------------------------------------
def _rolling_mode(labels, window: int):
    s = pd.Series(labels)
    out = []
    half = window // 2
    for i in range(len(s)):
        seg = s.iloc[max(0, i - half):i + half + 1]
        out.append(seg.mode().iloc[0])
    return out


def rocktype_loho(df: pd.DataFrame, feats, cluster_col: str = "Cluster", n_estimators: int = 300,
                  smooth_window: int = 5, seed: int = 0, min_group_rows: int = 40) -> dict:
    """Predict the broad logged rock group from mineral features, testing on unseen holes."""
    d = add_rock_group(df)
    d = d[d["Depth_QC_Valid"].fillna(False).astype(bool) & d["Geo_Match"].fillna(False).astype(bool) & d["Rock_Group"].notna()].copy()
    counts = d["Rock_Group"].value_counts()
    d = d[d["Rock_Group"].isin(counts[counts >= min_group_rows].index)].reset_index(drop=True)
    X = numeric_matrix(d, feats)
    y = d["Rock_Group"].to_numpy()
    groups = d["Hole_name"].astype(str).to_numpy()
    classes = sorted(set(y))

    pred = np.empty(len(d), dtype=object)
    conf = np.zeros(len(d))
    base_major = np.empty(len(d), dtype=object)
    base_cluster = np.empty(len(d), dtype=object)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", min_samples_leaf=3,
                                    random_state=seed, n_jobs=-1).fit(X.iloc[tr], y[tr])
        proba = rf.predict_proba(X.iloc[te])
        pred[te] = rf.classes_[proba.argmax(axis=1)]
        conf[te] = proba.max(axis=1)
        base_major[te] = pd.Series(y[tr]).mode().iloc[0]
        if cluster_col in d.columns and d[cluster_col].notna().any():
            tmp = pd.DataFrame({"c": d.loc[tr, cluster_col].to_numpy(), "y": y[tr]}).dropna()
            cmap = tmp.groupby("c")["y"].agg(lambda s: s.mode().iloc[0]).to_dict()
            base_cluster[te] = d.loc[te, cluster_col].map(cmap).fillna(pd.Series(y[tr]).mode().iloc[0]).to_numpy()
        else:
            base_cluster[te] = base_major[te]

    d["Predicted_group"], d["Prediction_confidence"] = pred, conf
    smooth = np.empty(len(d), dtype=object)
    for hole, idx in d.groupby("Hole_name").groups.items():
        order = d.loc[idx].sort_values("Depth_Mid").index.to_numpy()
        smooth[order] = _rolling_mode(d.loc[order, "Predicted_group"].to_list(), smooth_window)
    d["Predicted_group_smoothed"] = smooth

    def metrics(p):
        return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
                "macro_f1": f1_score(y, p, average="macro")}
    result = {
        "n_intervals": int(len(d)), "n_holes": int(d["Hole_name"].nunique()), "classes": classes,
        "model": metrics(pred), "model_smoothed": metrics(smooth),
        "baseline_majority": metrics(base_major), "baseline_cluster_only": metrics(base_cluster),
        "per_class": pd.DataFrame({
            "Rock_group": classes,
            "Intervals": [int((y == c).sum()) for c in classes],
            "Recall": [float(((pred == c) & (y == c)).sum() / max((y == c).sum(), 1)) for c in classes],
            "Precision": [float(((pred == c) & (y == c)).sum() / max((pred == c).sum(), 1)) for c in classes]}),
        "confusion": pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes),
        "per_hole": (d.assign(Correct=lambda t: t["Predicted_group"] == t["Rock_Group"])
                     .groupby("Hole_name").agg(Intervals=("Correct", "size"), Accuracy=("Correct", "mean")).reset_index()),
        "predictions": d[["Hole_name", "Depth_Mid", "Rock_Group", "Predicted_group", "Predicted_group_smoothed", "Prediction_confidence"]],
    }
    rf_full = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample", min_samples_leaf=3,
                                     random_state=seed, n_jobs=-1).fit(X, y)
    result["importance"] = (pd.DataFrame({"Feature": [pretty_feature(c) for c in X.columns], "Importance": rf_full.feature_importances_})
                            .sort_values("Importance", ascending=False).reset_index(drop=True))
    return result


def ternary_table(df: pd.DataFrame) -> pd.DataFrame:
    """MyLogger-style felsic / mafic / chlorite proportions.

    felsic = silica + K-feldspar, mafic = pyroxene + amphibole (all TIR), chlorite = chlorite (SWIR).
    Each interval is scaled so the three parts sum to 1; intervals with none of the three are dropped.
    """
    def col(name):
        return numeric_matrix(df, [name]).iloc[:, 0] if name in df.columns else pd.Series(0.0, index=df.index)
    felsic = col("Silica_(System_TIR)") + col("K-Feldspar_(System_TIR)")
    mafic = col("Pyroxene_(System_TIR)") + col("Amphibole_(System_TIR)")
    chl = col("Chlorite_(System_SWIR)")
    tot = felsic + mafic + chl
    out = pd.DataFrame({"Hole_name": df["Hole_name"].astype(str), "Depth_Mid": df["Depth_Mid"],
                        "Felsic": felsic / tot, "Mafic": mafic / tot, "Chlorite": chl / tot, "Total": tot})
    if "Geo_GSWARockName" in df.columns:
        out["Rock_Group"] = df["Geo_GSWARockName"].map(rock_group)
    return out[tot > 0.02].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Q5  Other properties (geochemistry) by PLS
# ----------------------------------------------------------------------------------------------
ASSAY_TARGETS = ["Assay_Cu_PPM", "Assay_Ni_PPM", "Assay_Co_PPM", "Assay_Cr_PPM", "Assay_Zn_PPM", "Assay_Pb_PPM",
                 "Assay_MgO_PCT", "Assay_CaO_PCT", "Assay_K2O_PCT", "Assay_Na2O_PCT", "Assay_Au_PPM", "Assay_As_PPM", "Assay_S_PPM"]


def _log_target(y: pd.Series) -> pd.Series:
    pos = y[y > 0]
    floor = pos.min() / 2 if len(pos) else 1e-6
    return np.log10(y.clip(lower=floor))


def pls_loho(df: pd.DataFrame, feats, targets=None, min_samples: int = 100, min_holes: int = 5,
             max_components: int = 8, subsets: dict | None = None):
    """Predict assay values from mineral features with PLS, tested on unseen holes (log10 scale).

    One row per assay *sample* (features are averaged over the HyLogger intervals that share a sample),
    so repeated intervals are not double-counted. `subsets` maps a name to a boolean mask over df.rows
    (e.g. one lab/report at a time) because assay units and methods can differ between reports.
    Returns (summary_df, predictions_df).
    """
    targets = [t for t in (targets or ASSAY_TARGETS) if t in df.columns]
    subsets = subsets or {"All holes": pd.Series(True, index=df.index)}
    base = df[df["Assay_Match"].fillna(False).astype(bool) & df["Depth_QC_Valid"].fillna(False).astype(bool)].copy()
    if base.empty or "AssayFlatDataId" not in base.columns:
        return pd.DataFrame(), pd.DataFrame()
    Xall = numeric_matrix(base, feats)
    Xall["_sample"] = base["AssayFlatDataId"].to_numpy()
    xs = Xall.groupby("_sample").mean()
    meta = base.drop_duplicates("AssayFlatDataId").set_index("AssayFlatDataId")
    summary, preds = [], []
    for sub_name, mask in subsets.items():
        sub_ids = set(base.loc[mask.reindex(base.index, fill_value=False), "AssayFlatDataId"])
        for t in targets:
            ys = pd.to_numeric(meta[t], errors="coerce").dropna()
            ys = ys[ys.index.isin(sub_ids)]
            ys = ys[ys.index.isin(xs.index)]
            holes = meta.loc[ys.index, "Hole_name"].astype(str)
            if len(ys) < min_samples or holes.nunique() < min_holes:
                continue
            X = xs.loc[ys.index]
            keep = X.columns[X.std() > 0]
            X = X[keep]
            y = _log_target(ys)
            yhat = pd.Series(np.nan, index=y.index)
            comps = []
            for tr, te in LeaveOneGroupOut().split(X, y, holes):
                sc = StandardScaler().fit(X.iloc[tr])
                Xtr, Xte = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])
                ytr = y.iloc[tr].to_numpy()
                n_tr_holes = holes.iloc[tr].nunique()
                best_k, best_err = 1, np.inf
                for k in range(1, min(max_components, Xtr.shape[1]) + 1):
                    err = []
                    for a, b in GroupKFold(n_splits=min(5, n_tr_holes)).split(Xtr, ytr, holes.iloc[tr]):
                        m = PLSRegression(n_components=k, scale=False).fit(Xtr[a], ytr[a])
                        err.append(np.mean((m.predict(Xtr[b]).ravel() - ytr[b]) ** 2))
                    if np.mean(err) < best_err:
                        best_k, best_err = k, np.mean(err)
                comps.append(best_k)
                m = PLSRegression(n_components=best_k, scale=False).fit(Xtr, ytr)
                yhat.iloc[te] = m.predict(Xte).ravel()
            sse = float(((y - yhat) ** 2).sum()); sst = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - sse / sst if sst > 0 else np.nan
            rho = float(spearmanr(y, yhat).correlation) if spearmanr is not None else np.nan
            verdict = "predictable" if r2 >= 0.5 else ("weak signal" if r2 >= 0.2 else "not predictable")
            summary.append({"Subset": sub_name, "Target": t.replace("Assay_", ""), "Samples": int(len(y)),
                            "Holes": int(holes.nunique()), "R2_unseen_holes": r2, "Spearman_rho": rho,
                            "RMSE_log10": float(np.sqrt(sse / len(y))), "Typical_components": int(np.median(comps)),
                            "Verdict": verdict})
            preds.append(pd.DataFrame({"Subset": sub_name, "Target": t.replace("Assay_", ""), "Hole_name": holes.values,
                                       "Depth": meta.loc[y.index, "Assay_MidDepth"].values,
                                       "Measured_log10": y.values, "Predicted_log10": yhat.values}))
    return (pd.DataFrame(summary), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


# ----------------------------------------------------------------------------------------------
# Q1  Endmembers and explained anomalies
# ----------------------------------------------------------------------------------------------
def nmf_endmembers(df: pd.DataFrame, mineral_cols=None, k: int | None = None, k_range=range(2, 9),
                   elbow: float = 0.08, seed: int = 0, unexplained_pct: float = 97.0):
    """Extract spectral-mineral endmembers in bulk with non-negative matrix factorisation (NMF).

    Every interval is expressed as a mix of k endmembers (a "pure" mineral assemblage each).
    k is the smallest value where adding another endmember improves the fit by less than `elbow`.
    Intervals the endmembers explain badly (top 3% residual) are flagged as 'unexplained'.
    """
    mineral_cols = mineral_cols or mineral_fraction_columns(df)
    X = numeric_matrix(df, mineral_cols).clip(lower=0)
    X = X.loc[:, X.sum() > 0]
    norm_x = np.linalg.norm(X.to_numpy())
    errs = {}
    models = {}
    for kk in k_range:
        m = NMF(n_components=kk, init="nndsvda", max_iter=600, random_state=seed)
        W = m.fit_transform(X)
        errs[kk] = float(np.linalg.norm(X.to_numpy() - W @ m.components_) / norm_x)
        models[kk] = (m, W)
    if k is None:
        ks = sorted(errs)
        k = ks[-1]
        for a, b in zip(ks[:-1], ks[1:]):
            if (errs[a] - errs[b]) / errs[a] < elbow:
                k = a
                break
    m, W = models[k]
    H = m.components_
    Hn = H / H.sum(axis=1, keepdims=True)
    labels = []
    for r in range(k):
        top = pd.Series(Hn[r], index=X.columns).sort_values(ascending=False)
        names = [pretty_feature(c) for c in top.index[:3] if top[c] >= 0.12]
        labels.append(f"EM{r + 1}: " + " + ".join(names or [pretty_feature(top.index[0])]))
    comp = pd.DataFrame(Hn.T, index=[pretty_feature(c) for c in X.columns], columns=labels)
    Wn = W / np.where(W.sum(axis=1, keepdims=True) == 0, 1, W.sum(axis=1, keepdims=True))
    ab = pd.DataFrame(Wn, index=df.index, columns=[f"EM{r + 1}_share" for r in range(k)])
    resid = np.linalg.norm(X.to_numpy() - W @ H, axis=1)
    out = pd.DataFrame({"Dominant_endmember": [labels[i] for i in Wn.argmax(axis=1)],
                        "Endmember_purity": Wn.max(axis=1), "Residual": resid}, index=df.index)
    out["Unexplained"] = out["Residual"] >= np.percentile(resid, unexplained_pct)
    out["Dominant_endmember"] = out["Dominant_endmember"].where(X.sum(axis=1) > 0, "none detected")
    return {"k": int(k), "error_curve": pd.DataFrame({"k": list(errs), "relative_error": list(errs.values())}),
            "composition": comp, "labels": labels, "abundance": pd.concat([ab, out], axis=1)}


def anomaly_reasons(df: pd.DataFrame, feats, z_min: float = 2.5, top_n: int = 2) -> pd.Series:
    """Plain-language reason each interval looks unusual: its most extreme mineral/feature values."""
    X = numeric_matrix(df, feats)
    sd = X.std().replace(0, np.nan)
    Z = (X - X.mean()) / sd
    reasons = []
    for _, row in Z.iterrows():
        hi = row[row > z_min].sort_values(ascending=False).head(top_n)
        reasons.append("; ".join(f"high {pretty_feature(k)} (z={v:.1f})" for k, v in hi.items()) or "combination of features")
    return pd.Series(reasons, index=df.index)


# ----------------------------------------------------------------------------------------------
# Q2  Spectral feature positions (what TSG extracted beyond the mineral labels)
# ----------------------------------------------------------------------------------------------
FEATURE_FAMILIES = {
    "Fe absorption (~900 nm)": ("900W_Fe", "900D_Fe"),
    "Al-OH (~2200 nm)": ("2200W_AlOH", "2200D_AlOH"),
    "Fe-OH (~2250 nm)": ("2250W_FeOH", "2250D_FeOH"),
    "Carbonate (~11300 nm, TIR)": ("11300W_Carbonate", "11300D_Carbonate"),
}
DEPTH_ONLY_FEATURES = {"Water (~1900 nm)": "1900D_Water", "Quartz (~8625 nm, TIR)": "8625D_Quartz"}


def wavelength_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Does the position of each absorption feature differ between logged rock groups?"""
    d = add_rock_group(df)
    d = d[d["Rock_Group"].notna()]
    rows = []
    for fam, (wcol, _) in FEATURE_FAMILIES.items():
        if wcol not in d.columns:
            continue
        v = d[["Rock_Group", wcol]].copy()
        v[wcol] = pd.to_numeric(v[wcol], errors="coerce")
        v = v.dropna()
        groups = [g[wcol].to_numpy() for _, g in v.groupby("Rock_Group") if len(g) >= 15]
        p = float(kruskal(*groups).pvalue) if kruskal is not None and len(groups) >= 2 else np.nan
        for rg, g in v.groupby("Rock_Group"):
            if len(g) >= 15:
                rows.append({"Feature": fam, "Wavelength_column": wcol, "Rock_group": rg, "Intervals": len(g),
                             "Median_nm": g[wcol].median(), "P25_nm": g[wcol].quantile(0.25), "P75_nm": g[wcol].quantile(0.75),
                             "Kruskal_p_across_groups": p})
    return pd.DataFrame(rows)
