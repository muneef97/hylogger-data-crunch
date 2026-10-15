"""HyLogger Insight Explorer, merged with the HyLogger Data Crunch pipeline (65 holes).

Interactive analyses (boundary penalty, rock-type model, PLS, endmember count) come from a teammate's
Insight Explorer (hylogger_analysis.py); the validated pipeline results (scripts 03-16) are shown beside them.

Run from the repo folder, after ./run_all.sh:
    python app/build_app_data.py          # converts our outputs into the app's format (~5 s)
    python -m streamlit run app/streamlit_app.py
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent))   # hylogger_analysis.py sits beside this file
try:
    import hylogger_analysis as hla
except ImportError:
    hla = None

st.set_page_config(page_title="HyLogger Insight Explorer", page_icon="🪨", layout="wide")
st.title("HyLogger Insight Explorer")
st.caption("Exploratory mineralogical characterisation • depth-aligned context • auditable anomaly review")

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parent
PIPE = REPO / "outputs"                      # the pipeline's own tables and figures
PLOTS = PIPE / "plots"
sys.path.insert(0, str(REPO / "pipeline"))
ROOT = Path.cwd()
CANDIDATE_DIRS = [PIPE / "app", ROOT / "outputs", ROOT.parent / "outputs"]
OUTPUT_DIR = next((p for p in CANDIDATE_DIRS if (p / "processed" / "hylogger_modelled_intervals.csv").exists()), None)
if OUTPUT_DIR is None:
    st.error("App data not found. Run the pipeline, then convert its outputs for the app.")
    st.code("./run_all.sh\npython app/build_app_data.py\npython -m streamlit run app/streamlit_app.py")
    st.stop()

PROCESSED_DIR = OUTPUT_DIR / "processed"
REPORT_DIR = OUTPUT_DIR / "reports"
DATA_FILE = PROCESSED_DIR / "hylogger_modelled_intervals.csv"
CONFIG_FILE = REPORT_DIR / "feature_config.json"

@st.cache_data(show_spinner=False)
def load_data(path, mtime):
    return pd.read_csv(path, low_memory=False)

@st.cache_data(show_spinner=False)
def load_config(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

modelled = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
config = load_config(CONFIG_FILE)

if "Depth_Mid" in modelled:
    modelled["Depth_Mid"] = pd.to_numeric(modelled["Depth_Mid"], errors="coerce")
if "Cluster" in modelled:
    modelled["Cluster"] = pd.to_numeric(modelled["Cluster"], errors="coerce")
if "AnomalyScore" in modelled:
    modelled["AnomalyScore"] = pd.to_numeric(modelled["AnomalyScore"], errors="coerce")

holes = sorted(modelled["Hole_name"].dropna().astype(str).unique()) if "Hole_name" in modelled else []
feature_cols = [c for c in config.get("display_numeric_features", []) if c in modelled.columns]
assay_cols = [c for c in config.get("assay_columns", []) if c in modelled.columns and pd.api.types.is_numeric_dtype(modelled[c])]
assay_cols = [c for c in assay_cols if c not in {"Assay_MidDepth", "Assay_Match_Distance"} and "Match" not in c]
rock_col = "Geo_GSWARockName" if "Geo_GSWARockName" in modelled.columns else None
model_features = [c for c in config.get("model_feature_columns", feature_cols) if c in modelled.columns] or feature_cols
DATA_MTIME = DATA_FILE.stat().st_mtime
SWIR_MINERALS = [c for c in modelled.columns if str(c).endswith("(System_SWIR)") and "invalid" not in str(c).lower()]
PALETTE = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
# One fixed colour per rock group, used on every page (same rock = same colour everywhere).
# Loosely follows geological-map conventions and stays distinct on light and dark backgrounds.
ROCK_COLORS = {
    "Felsic-intermediate & schist": "#FF5C8A",                   # pink   (granite-like)
    "Mafic (amphibolite, basalt, dolerite, gabbro)": "#2FBF71",  # green  (basalt / dolerite)
    "Sandstone & conglomerate": "#F5C518",                       # yellow (sand)
    "Siltstone & claystone": "#F28C28",                          # orange (fine sediment)
    "Carbonate (dolomite)": "#3DA5F4",                           # blue   (carbonate)
    "Regolith": "#B88A5A",                                       # brown  (weathered cover)
    "(not classified)": "#BBBBBB",
}
ROCK10D_COLORS = {  # script 10d rock types: pinks = felsic, purple = intermediate, greens = mafic, teal = ultramafic, yellow/orange = sedimentary
    "Granite / granitic": "#FF5C8A", "Felsic gneiss / granulite": "#F7A1C0", "Intermediate (diorite, andesite)": "#9B6BD6",
    "Basalt": "#2FBF71", "Dolerite / gabbro": "#1E7D4E", "Amphibolite / mafic gneiss": "#9BD8A6", "Ultramafic": "#14A3B8",
    "Sandstone / conglomerate": "#F5C518", "Siltstone / mudstone / shale": "#F28C28",
}
def rock_color(group, i=0):
    return ROCK_COLORS.get(group, px.colors.qualitative.T10[i % 10])


def pipe_csv(name):
    """A table from the pipeline's outputs/ folder, or None if that script has not been run."""
    p = PIPE / name
    return pd.read_csv(p, low_memory=False) if p.exists() else None


def pipe_image(name, caption=None):
    p = PLOTS / name
    if p.exists():
        st.image(str(p), caption=caption)
        return True
    return False


def need_module():
    if hla is None:
        st.error("This page needs hylogger_analysis.py in the same folder as streamlit_app.py.")
        st.stop()


def stacked_mineral_traces(d, cols, depth_col="Depth_Mid", label_fn=None):
    """Horizontal stacked bars (one trace per mineral/endmember) along depth."""
    h = (d["Mineral_ToDepth"] - d["Mineral_FromDepth"]).clip(lower=0.1).to_numpy() if "Mineral_ToDepth" in d else np.full(len(d), 1.0)
    cum = np.zeros(len(d))
    traces = []
    for i, c in enumerate(cols):
        v = pd.to_numeric(d[c], errors="coerce").fillna(0).to_numpy()
        traces.append(go.Bar(x=v, y=d[depth_col], base=cum.copy(), width=h, orientation="h", name=(label_fn(c) if label_fn else str(c)),
                             marker=dict(color=PALETTE[i % len(PALETTE)], line=dict(width=0)),
                             hovertemplate="%{y:.1f} m: %{x:.2f}<extra>" + (label_fn(c) if label_fn else str(c)) + "</extra>"))
        cum = cum + v
    return traces


@st.cache_data(show_spinner=False)
def cached_reasons(mtime):
    return hla.anomaly_reasons(modelled, model_features)

# Intro and KPIs
n_rows = len(modelled)
n_holes = modelled["Hole_name"].nunique() if "Hole_name" in modelled else 0
valid_depth = modelled["Depth_Mid"].notna().sum() if "Depth_Mid" in modelled else 0
geo_rate = 100 * modelled.get("Geo_Match", pd.Series(False, index=modelled.index)).fillna(False).astype(bool).mean() if n_rows else 0
assay_rate = 100 * modelled.get("Assay_Match", pd.Series(False, index=modelled.index)).fillna(False).astype(bool).mean() if n_rows else 0
anom_count = modelled.get("IsAnomaly", pd.Series(False, index=modelled.index)).fillna(False).astype(bool).sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Drillholes", f"{n_holes:,}")
k2.metric("Log rows", f"{n_rows:,}")
k3.metric("Rows with depth", f"{valid_depth:,}")
k4.metric("Geology match", f"{geo_rate:.1f}%")
k5.metric("Anomaly flags", f"{anom_count:,}")

if config.get("feature_mode") == "categorical_mineral_labels_fallback":
    st.warning("This run clustered categorical mineral labels. Interpret it as mineral-assemblage co-occurrence, not as discovery from raw spectral values.")
else:
    st.info("Clusters and anomaly scores are exploratory. Inspect the raw interval and independent geological evidence before assigning geological meaning.")

page = st.sidebar.radio("Explore", ["Overview", "GSWA question scorecard", "Map", "Downhole explorer", "Downhole tracks (pipeline)", "Clusters & validation", "Q1 Endmembers & anomalies", "Q2 Spectral features", "Q3 Zone boundaries", "Q4 Rock type", "Q5 Geochemistry", "Anomaly review", "Data quality", "Export & provenance"], key="page")
st.sidebar.caption(f"Scope: {config.get('scope', 'unknown')} • Feature mode: {config.get('feature_mode', 'unknown')}")

if page == "Overview":
    st.subheader("Data coverage and quality")
    c1, c2 = st.columns(2)
    with c1:
        if "Hole_name" in modelled:
            counts = modelled.groupby("Hole_name").size().sort_values(ascending=False).head(25).sort_values()
            st.write("**Rows per drillhole (top 25)**")
            st.bar_chart(counts)
    with c2:
        coverage_path = REPORT_DIR / "per_hole_data_coverage.csv"
        if coverage_path.exists():
            coverage = pd.read_csv(coverage_path)
            show_cols = [c for c in ["Hole_name", "mineral_rows", "rows_with_valid_depth", "geology_match_pct", "assay_match_pct"] if c in coverage.columns]
            st.write("**Coverage by hole**")
            st.dataframe(coverage[show_cols].sort_values("rows_with_valid_depth", ascending=False), use_container_width=True, height=350)
    st.subheader("How to read this application")
    st.markdown("""
    - **GSWA question scorecard:** what each of the five GSWA questions now has. Pages Q1 to Q5 hold the detail.
    - **Map:** see where each drillhole is, coloured by company, and click a hole for its summary.
    - **Downhole tracks (pipeline):** one hole's SWIR/TIR minerals, endmembers, assemblage, rock class vs log, an assay and the pipeline's boundaries side by side.
    - **Data quality:** the assay problems the pipeline found and set aside before any model saw them.
    - **Downhole explorer:** compare mineral-log values with lithology and nearby assays at consistent depth.
    - **Clusters & validation:** inspect statistical groups and how they compare with independently logged rock names.
    - **Anomaly review:** prioritise observations that are unusual relative to this cohort; flags are not proof of mineralisation.
    - **Export & provenance:** review source links, model feature selection and limitations.
    """)

elif page == "Map":
    import plotly.express as px

    st.subheader("Drillhole map")

    def find_collar_file():
        if (OUTPUT_DIR / "collars.csv").exists():
            return OUTPUT_DIR / "collars.csv"
        roots = [ROOT, ROOT / "data", ROOT.parent, ROOT.parent / "data", ROOT.parent.parent, ROOT.parent.parent / "data",
                 OUTPUT_DIR.parent, OUTPUT_DIR.parent / "data"]
        for r in roots:
            p = r / "WAMEX_Collar.csv"
            if p.exists():
                return p
        return None

    collar_file = find_collar_file()
    if collar_file is None or "Join_CollarId" not in modelled.columns:
        st.error("The map needs WAMEX_Collar.csv (for coordinates and company) and a Join_CollarId column in the modelled export.")
        st.caption("Looked for WAMEX_Collar.csv in the app folder, its parent, and their 'data' sub-folders.")
        st.stop()

    @st.cache_data(show_spinner=False)
    def load_collars(path, mtime):
        want = {"CollarId", "CompanyHoleId", "Longitude", "Latitude", "HoleType", "MaxDepth", "Elevation", "Company"}
        c = pd.read_csv(path, usecols=lambda col: col in want, low_memory=False)
        c["CollarId"] = pd.to_numeric(c["CollarId"], errors="coerce")
        for col in ["Longitude", "Latitude", "MaxDepth", "Elevation"]:
            if col in c:
                c[col] = pd.to_numeric(c[col], errors="coerce")
        return c.dropna(subset=["CollarId"]).drop_duplicates("CollarId")

    def as_bool(col):
        if col in modelled:
            return modelled[col].fillna(False).astype(bool)
        return pd.Series(False, index=modelled.index)

    # SWIR mineral-fraction columns describe "what is in the rock"; fall back to the display features.
    swir_cols = [c for c in modelled.columns if str(c).endswith("(System_SWIR)") and "invalid" not in str(c).lower()]
    mineral_cols = swir_cols or feature_cols[:8]

    def pretty_mineral(col):
        return str(col).replace("_(System_SWIR)", "").replace("_", " ")

    def top_minerals(frame, n=3):
        if not mineral_cols:
            return "n/a"
        means = frame[mineral_cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean().sort_values(ascending=False)
        parts = [f"{pretty_mineral(k)} {100*v:.0f}%" for k, v in means.head(n).items() if v > 0.01]
        return ", ".join(parts) if parts else "none detected"

    def top_rocks(frame, n=3):
        if rock_col is None:
            return "n/a"
        vc = frame[rock_col].dropna().value_counts(normalize=True).head(n)
        return ", ".join(f"{k} {100*v:.0f}%" for k, v in vc.items()) if len(vc) else "no lithology matched"

    work = modelled.copy()
    work["_geo"], work["_assay"], work["_anom"] = as_bool("Geo_Match"), as_bool("Assay_Match"), as_bool("IsAnomaly")
    rows = []
    for hole, g in work.groupby(work["Hole_name"].astype(str)):
        cl = g["Cluster"].dropna() if "Cluster" in g else pd.Series(dtype=float)
        rows.append({
            "Hole_name": hole,
            "CollarId": pd.to_numeric(g["Join_CollarId"], errors="coerce").dropna().iloc[0] if g["Join_CollarId"].notna().any() else np.nan,
            "Log_intervals": len(g),
            "Log_depth_min": g["Depth_Mid"].min(), "Log_depth_max": g["Depth_Mid"].max(),
            "Geology_match_pct": 100 * g["_geo"].mean(), "Assay_match_pct": 100 * g["_assay"].mean(),
            "Anomalies": int(g["_anom"].sum()),
            "Dominant_cluster": f"Cluster {int(cl.mode().iloc[0])}" if len(cl) else "n/a",
            "Top_minerals": top_minerals(g), "Top_rocks": top_rocks(g),
        })
    hole_tbl = pd.DataFrame(rows).merge(load_collars(str(collar_file), collar_file.stat().st_mtime), on="CollarId", how="left")
    hole_tbl["Company"] = hole_tbl["Company"].fillna("Unknown")
    hole_tbl["HoleType"] = hole_tbl["HoleType"].fillna("Unknown")
    hole_tbl["Depth_flag"] = hole_tbl["Log_depth_max"] > (hole_tbl["MaxDepth"] + 1.0)
    mappable = hole_tbl.dropna(subset=["Latitude", "Longitude"]).copy()
    missing = hole_tbl[hole_tbl["Latitude"].isna() | hole_tbl["Longitude"].isna()]

    if mappable.empty:
        st.warning("None of the modelled holes have coordinates in WAMEX_Collar.csv.")
        st.stop()

    cA, cB, cC = st.columns(3)
    colour_by = cA.selectbox("Colour holes by", ["Company", "Hole type", "Dominant cluster", "Geology match %"])
    basemap = cB.selectbox("Basemap", ["No basemap (works offline)", "Australia outline (downloads outline)", "Online street map (needs internet)"])
    view = cC.selectbox("View", ["Fit to holes", "Whole of Australia"])
    colour_col = {"Company": "Company", "Hole type": "HoleType", "Dominant cluster": "Dominant_cluster", "Geology match %": "Geology_match_pct"}[colour_by]
    continuous = colour_col == "Geology_match_pct"

    mappable["Depth_text"] = mappable["MaxDepth"].map(lambda v: f"{v:,.1f} m" if pd.notna(v) else "n/a")
    hover = {"Company": True, "HoleType": True, "Depth_text": True, "Top_minerals": True, "Top_rocks": True,
             "Geology_match_pct": ":.0f", "Latitude": False, "Longitude": False, "Hole_name": False,
             "Dominant_cluster": False, "Depth_flag": False}
    common = dict(lat="Latitude", lon="Longitude", color=colour_col, hover_name="Hole_name",
                  hover_data={k: v for k, v in hover.items() if k != colour_col or v is not False}, custom_data=["Hole_name"])
    if continuous:
        common["color_continuous_scale"] = "Viridis"; common["range_color"] = (0, 100)

    lat_lo, lat_hi = mappable["Latitude"].min(), mappable["Latitude"].max()
    lon_lo, lon_hi = mappable["Longitude"].min(), mappable["Longitude"].max()
    clat, clon = (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2

    if basemap.startswith("No basemap"):
        # plain longitude/latitude scatter: nothing is downloaded, so it works with no internet
        if continuous:
            fig = px.scatter(mappable, x="Longitude", y="Latitude", color=colour_col, hover_name="Hole_name",
                             custom_data=["Hole_name"], color_continuous_scale="Viridis", range_color=(0, 100))
        else:
            fig = px.scatter(mappable, x="Longitude", y="Latitude", color=colour_col, hover_name="Hole_name", custom_data=["Hole_name"])
        idx_file = REPO / "outputs" / "13_dataset_overview.csv"
        if idx_file.exists():
            ov = pd.read_csv(idx_file, usecols=["Lat", "Lon"]).dropna()
            ov = ov[ov["Lat"].between(-36, -12) & ov["Lon"].between(111, 130)]
            fig.add_trace(go.Scatter(x=ov["Lon"], y=ov["Lat"], mode="markers", marker=dict(size=4, color="#d4d3ce"),
                                     name="Other indexed holes", hoverinfo="skip"))
            fig.data = fig.data[-1:] + fig.data[:-1]   # grey context dots underneath
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")), selector=dict(mode="markers", hoverinfo=None))
        if view == "Whole of Australia":
            fig.update_xaxes(range=[112, 155]); fig.update_yaxes(range=[-45, -9])
        else:
            fig.update_xaxes(range=[111.5, 129.5]); fig.update_yaxes(range=[-35.5, -12.5])
        fig.update_yaxes(scaleanchor="x", scaleratio=1.1)
        fig.update_layout(plot_bgcolor="#fcfcfb")
    elif basemap.startswith("Online"):
        scatter_map = getattr(px, "scatter_map", None) or px.scatter_mapbox
        new_api = hasattr(px, "scatter_map")
        fig = scatter_map(mappable, **common)
        if view == "Whole of Australia":
            centre, zoom = {"lat": -25.5, "lon": 134.0}, 3.0
        else:
            span = max(lon_hi - lon_lo, (lat_hi - lat_lo) * 1.2, 0.02)
            centre, zoom = {"lat": clat, "lon": clon}, float(np.clip(np.log2(1260 / (span * 1.5)), 3, 13))
        if new_api:
            fig.update_layout(map_style="open-street-map", map_center=centre, map_zoom=zoom)
        else:
            fig.update_layout(mapbox_style="open-street-map", mapbox_center=centre, mapbox_zoom=zoom)
        fig.update_traces(marker=dict(size=13))
    else:
        fig = px.scatter_geo(mappable, **common)
        if view == "Whole of Australia":
            lon_rng, lat_rng = [112, 155], [-45, -9]
        else:
            half = max((lon_hi - lon_lo) / 2, (lat_hi - lat_lo) / 2, 4.0)   # keep regional context visible
            lon_rng, lat_rng = [clon - half * 1.3, clon + half * 1.3], [clat - half, clat + half]
        fig.update_geos(scope="world", resolution=50, lonaxis_range=lon_rng, lataxis_range=lat_rng,
                        showland=True, landcolor="#f1ead7", showocean=True, oceancolor="#dfeaf2",
                        showcountries=True, countrycolor="#999999", showsubunits=True, subunitcolor="#b5a98a",
                        showcoastlines=True, coastlinecolor="#555555", showlakes=False, projection_type="mercator")
        fig.update_traces(marker=dict(size=13, line=dict(width=1, color="white")))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0), legend=dict(title=colour_by, bgcolor="rgba(255,255,255,0.8)"))

    st.caption("Click a hole to see its summary below. If several holes sit at the same spot, use the hole picker under the map.")
    try:
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="hole_map")
    except TypeError:
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key="hole_map")

    clicked = None
    try:
        pts = event.selection.points if event is not None else []
        if pts and pts[0].get("customdata"):
            clicked = str(pts[0]["customdata"][0])
    except Exception:
        clicked = None

    options = sorted(mappable["Hole_name"].astype(str))
    default_idx = options.index(clicked) if clicked in options else 0
    picked = st.selectbox("Selected hole (click the map, or choose here)", options, index=default_idx,
                          key=f"map_pick_{clicked or 'none'}")
    h = mappable[mappable["Hole_name"].astype(str) == picked].iloc[0]

    st.markdown(f"### {h['Hole_name']}  ·  {h['Company']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Hole type", str(h["HoleType"]))
    m2.metric("Recorded hole depth", h["Depth_text"])
    m3.metric("HyLogger intervals", f"{int(h['Log_intervals']):,}")
    m4.metric("Logged depth range", f"{h['Log_depth_min']:.0f}–{h['Log_depth_max']:.0f} m")
    n1, n2, n3 = st.columns(3)
    n1.metric("Lithology matched", f"{h['Geology_match_pct']:.0f}%")
    n2.metric("Assay matched", f"{h['Assay_match_pct']:.0f}%")
    n3.metric("Anomaly flags", f"{int(h['Anomalies'])}")
    st.write(f"**Main minerals (SWIR average):** {h['Top_minerals']}")
    st.write(f"**Logged rock types (share of intervals):** {h['Top_rocks']}")
    st.write(f"**Most common mineral cluster:** {h['Dominant_cluster']}  ·  **Location:** {h['Latitude']:.4f}, {h['Longitude']:.4f}")
    if bool(h["Depth_flag"]):
        st.warning(f"Data-quality flag: the HyLogger log reaches {h['Log_depth_max']:.0f} m but the collar records a maximum depth of {h['MaxDepth']:.1f} m. "
                   "Check which record is correct before relying on the lithology and assay match for this hole.")
    elif h["Geology_match_pct"] < 80:
        st.warning("Lithology match is below 80% for this hole; part of the log has no logged lithology interval at the same depth.")

    def _open_downhole(hole):
        st.session_state["hole_select"] = hole
        st.session_state["page"] = "Downhole explorer"
    st.button("Open this hole in the Downhole explorer", on_click=_open_downhole, args=(str(h["Hole_name"]),))

    with st.expander("Holes by company"):
        by_co = hole_tbl.groupby("Company").agg(Holes=("Hole_name", "nunique"), Log_intervals=("Log_intervals", "sum"),
                                                Geology_match_pct=("Geology_match_pct", "mean"), Assay_match_pct=("Assay_match_pct", "mean")).round(1)
        st.dataframe(by_co.sort_values("Holes", ascending=False), use_container_width=True)
    if len(missing):
        st.caption(f"{len(missing)} modelled hole(s) have no coordinates in WAMEX_Collar.csv and are not shown: " + ", ".join(missing["Hole_name"].astype(str).head(10)))
    st.caption("Coordinates, company and recorded depth come from WAMEX_Collar.csv; mineral, lithology and assay figures come from the pipeline output. "
               "Colour by company shows who drilled where; it is descriptive, not evidence that one company's holes differ geologically.")

elif page == "Downhole explorer":
    st.subheader("Depth-aligned drillhole view")
    if not holes:
        st.warning("No drillhole identifiers found in the modelled file.")
        st.stop()
    selected_hole = st.selectbox("Select drillhole", holes, key="hole_select")
    hole_df = modelled[modelled["Hole_name"].astype(str).eq(selected_hole)].copy()
    hole_df = hole_df[hole_df["Depth_Mid"].notna()].sort_values("Depth_Mid")
    if hole_df.empty:
        st.warning("This hole has no usable depth values in the current export.")
        st.stop()
    dmin = float(hole_df["Depth_Mid"].min())
    dmax = float(hole_df["Depth_Mid"].max())
    if dmax > dmin:
        depth_min, depth_max = st.slider("Depth range (m)", min_value=float(dmin), max_value=float(dmax), value=(float(dmin), float(dmax)))
        hole_df = hole_df[hole_df["Depth_Mid"].between(depth_min, depth_max)]
    else:
        st.caption(f"Single depth value: {dmin:.2f} m")
    left, right = st.columns([3, 1])
    with right:
        display_feats = st.multiselect("Mineral-log features", feature_cols, default=feature_cols[:min(3, len(feature_cols))])
        available_assays = [c for c in assay_cols if hole_df[c].notna().any()]
        selected_assay = st.selectbox("Assay track", ["(none)"] + available_assays, index=1 if available_assays else 0)
        show_anomalies = st.checkbox("Show anomaly flags", value=True)
    with left:
        n_feature_tracks = max(1, len(display_feats))
        ncols = n_feature_tracks + 2  # lithology + cluster/anomaly track
        titles = [f"{c}" for c in display_feats] if display_feats else ["Mineral-log feature"]
        titles += ["Logged lithology", "Cluster / anomaly"]
        fig = make_subplots(rows=1, cols=ncols, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.04)
        col_num = 1
        for f in display_feats:
            x = pd.to_numeric(hole_df[f], errors="coerce")
            fig.add_trace(go.Scatter(x=x, y=hole_df["Depth_Mid"], mode="lines+markers", marker={"size": 4}, name=f, customdata=np.stack([hole_df.get("Mineral_FromDepth", pd.Series(np.nan, index=hole_df.index)), hole_df.get("Mineral_ToDepth", pd.Series(np.nan, index=hole_df.index))], axis=-1), hovertemplate=f"{f}=%{{x}}<br>Depth=%{{y:.2f}} m<extra></extra>"), row=1, col=col_num)
            fig.update_xaxes(title_text=f, row=1, col=col_num)
            col_num += 1
        if not display_feats:
            fig.add_annotation(text="No numeric mineral features were selected/available. Inspect source mineral-name columns in the export.", x=0.5, y=0.5, xref="x domain", yref="y domain", showarrow=False)
        # Lithology strip represented with source-label hover points.
        if rock_col and hole_df[rock_col].notna().any():
            rocks = hole_df[rock_col].fillna("Unlogged").astype(str)
            cats = {v: i for i, v in enumerate(sorted(rocks.unique()))}
            fig.add_trace(go.Scatter(x=rocks.map(cats), y=hole_df["Depth_Mid"], mode="markers", marker={"size": 7}, text=rocks, name="Lithology", hovertemplate="Lithology=%{text}<br>Depth=%{y:.2f} m<extra></extra>"), row=1, col=col_num)
            fig.update_xaxes(title_text="Rock code", tickvals=list(cats.values()), ticktext=list(cats.keys()), row=1, col=col_num)
        else:
            fig.add_trace(go.Scatter(x=np.zeros(len(hole_df)), y=hole_df["Depth_Mid"], mode="markers", marker={"size": 3}, name="No geology match", hovertemplate="Depth=%{y:.2f} m<extra></extra>"), row=1, col=col_num)
            fig.update_xaxes(title_text="Lithology not available", row=1, col=col_num)
        col_num += 1
        if "Cluster" in hole_df.columns:
            cluster_text = hole_df["Cluster"].astype("string").fillna("not modelled")
            marker = {"size": 6, "symbol": "circle"}
            if show_anomalies and "IsAnomaly" in hole_df.columns:
                marker["symbol"] = np.where(hole_df["IsAnomaly"].fillna(False).astype(bool), "x", "circle")
                marker["size"] = np.where(hole_df["IsAnomaly"].fillna(False).astype(bool), 10, 6)
            fig.add_trace(go.Scatter(x=pd.to_numeric(hole_df["Cluster"], errors="coerce"), y=hole_df["Depth_Mid"], mode="markers", marker=marker, text=cluster_text, customdata=hole_df.get("AnomalyScore", pd.Series(np.nan, index=hole_df.index)), name="Cluster / anomaly", hovertemplate="Cluster=%{text}<br>Depth=%{y:.2f} m<br>Anomaly score=%{customdata}<extra></extra>"), row=1, col=col_num)
            fig.update_xaxes(title_text="Cluster ID", row=1, col=col_num)
        else:
            fig.add_trace(go.Scatter(x=np.zeros(len(hole_df)), y=hole_df["Depth_Mid"], mode="markers", name="No clusters"), row=1, col=col_num)
        fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=1, col=1)
        fig.update_layout(height=850, showlegend=False, margin={"l": 30, "r": 20, "t": 70, "b": 40})
        st.plotly_chart(fig, use_container_width=True)
    if selected_assay != "(none)":
        st.subheader(f"Assay detail: {selected_assay}")
        assay_plot = go.Figure()
        assay_plot.add_trace(go.Scatter(x=hole_df[selected_assay], y=hole_df["Depth_Mid"], mode="lines+markers", name=selected_assay, hovertemplate=f"{selected_assay}=%{{x}}<br>Depth=%{{y:.2f}} m<extra></extra>"))
        assay_plot.update_yaxes(autorange="reversed", title="Depth (m)")
        assay_plot.update_xaxes(title=selected_assay)
        assay_plot.update_layout(height=500)
        st.plotly_chart(assay_plot, use_container_width=True)
    detail_cols = [c for c in ["Hole_name", "Depth_Mid", "Mineral_FromDepth", "Mineral_ToDepth", "Geo_GSWARockName", "Cluster", "AnomalyScore", "IsAnomaly", "Assay_Match_Distance"] + display_feats + ([selected_assay] if selected_assay != "(none)" else []) if c in hole_df.columns]
    st.dataframe(hole_df[detail_cols], use_container_width=True, height=300)
    st.download_button("Download selected hole rows", data=hole_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{selected_hole}_downhole_rows.csv", mime="text/csv")

elif page == "Clusters & validation":
    st.subheader("Cluster structure")
    if {"PCA1", "PCA2", "Cluster"}.issubset(modelled.columns):
        plot_df = modelled.dropna(subset=["PCA1", "PCA2", "Cluster"]).copy()
        if len(plot_df) > 10000:
            plot_df = plot_df.sample(10000, random_state=42)
        hover_cols = [c for c in ["Hole_name", "Depth_Mid", "Geo_GSWARockName", "AnomalyScore"] if c in plot_df.columns]
        hover_data = {c: True for c in hover_cols}
        fig = go.Figure()
        for cl in sorted(plot_df["Cluster"].dropna().unique()):
            part = plot_df[plot_df["Cluster"].eq(cl)]
            fig.add_trace(go.Scatter(x=part["PCA1"], y=part["PCA2"], mode="markers", name=f"Cluster {int(cl)}", customdata=part[hover_cols].to_numpy() if hover_cols else None, hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>"))
        fig.update_layout(height=600, xaxis_title="Principal component 1", yaxis_title="Principal component 2")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("PCA/cluster outputs are missing. Check Step 8 and the feature audit.")
    crosstab_path = REPORT_DIR / "cluster_vs_gswa_rockname_counts.csv"
    if crosstab_path.exists():
        st.subheader("Cluster vs GSWA logged rock name (counts)")
        st.dataframe(pd.read_csv(crosstab_path), use_container_width=True)
    st.caption("Cluster IDs are arbitrary statistical labels, not validated lithology classes. Inspect dominant and mixed rock names for each cluster.")

elif page == "GSWA question scorecard":
    st.subheader("The five GSWA Data Crunch questions")
    st.write("Each question has its own page. Everything is validated on holes the model did not see, or against an independent baseline.")
    sc_file = REPORT_DIR / "gswa_questions_scorecard.csv"
    if sc_file.exists():
        st.dataframe(pd.read_csv(sc_file), use_container_width=True, hide_index=True)
    else:
        st.info("Run python app/build_app_data.py to generate the scorecard from the pipeline outputs.")
    st.markdown("""
    | Question | Page |
    |---|---|
    | 1. Detect anomalies or endmembers in bulk | Q1 Endmembers & anomalies |
    | 2. Analyse features outside TSG | Q2 Spectral features |
    | 3. Map divisions in stratigraphy | Q3 Zone boundaries |
    | 4. Translate minerals to rock type | Q4 Rock type |
    | 5. Extract other properties in bulk | Q5 Geochemistry |
    """)
    st.caption("The table above holds the pipeline's validated results (run_all.sh). Each Q page also has interactive analyses you can tune; those are exploratory.")
    st.warning("Limits: 65 holes chosen for rich data (3.4% of the archive); raw spectra for one hole only (BWD072); mineral shares are TSA interpretations; gold and copper sit in sulphides the scanner cannot see; geophysics is listed as TBC in the index so it could not be tested.")

elif page == "Q3 Zone boundaries":
    need_module()
    st.subheader("Automatic zone boundaries")
    st.caption("The mineral signature down each hole is split into zones by optimal partitioning. Dashed red lines are detected boundaries; black triangles are depths where the logged rock group changes.")
    c1, c2, c3 = st.columns(3)
    hole = c1.selectbox("Hole", holes, key="zone_hole")
    scale = c2.select_slider("Boundary penalty (higher = fewer boundaries)", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
    tol = c3.slider("Match tolerance (m)", 2, 10, 5)

    @st.cache_data(show_spinner="Detecting boundaries…")
    def run_boundaries(scale_, tol_, mtime):
        b, z = hla.detect_boundaries(modelled, model_features, penalty_scale=scale_)
        zt = hla.describe_zones(modelled, z, hla.mineral_fraction_columns(modelled))
        sc = hla.score_boundaries(modelled, b, tol_m=tol_, level="group", n_random=200)
        return b, zt, sc

    bnd, zone_tbl, sc = run_boundaries(scale, tol, DATA_MTIME)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Boundaries detected", f"{int(sc['detected_boundaries']):,}", f"{zone_tbl.groupby('Hole_name').size().median():.0f} zones per hole (median)", delta_color="off")
    m2.metric(f"Boundaries within ±{tol} m of a logged contact", f"{100*sc['precision']:.0f}%", f"random: {100*sc['chance_precision']:.0f}%", delta_color="off")
    m3.metric("Logged contacts that have a boundary", f"{100*sc['recall']:.0f}%", f"random: {100*sc['chance_recall']:.0f}%", delta_color="off")
    m4.metric("Better than random?", "Yes" if sc["p_value_f1"] < 0.05 else "Not shown", f"p ≈ {sc['p_value_f1']:.3f}", delta_color="off")

    d = modelled[(modelled["Hole_name"].astype(str) == hole) & modelled["Depth_QC_Valid"].fillna(False).astype(bool)].sort_values("Depth_Mid")
    b_h = bnd.loc[bnd["Hole_name"] == hole, "Boundary_depth"].to_numpy() if len(bnd) else np.array([])
    z_h = zone_tbl[zone_tbl["Hole_name"] == hole]
    contacts = hla.logged_contacts(d, level="group")
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, column_widths=[0.55, 0.2, 0.25], horizontal_spacing=0.02,
                        subplot_titles=("SWIR mineral fractions", "Logged rock group", "Detected zones"))
    for t in stacked_mineral_traces(d, SWIR_MINERALS, label_fn=lambda c: hla.pretty_feature(c).replace(" (SWIR)", "")):
        fig.add_trace(t, row=1, col=1)
    rg = d["Geo_GSWARockName"].map(hla.rock_group).fillna("(not classified)") if "Geo_GSWARockName" in d else pd.Series("(none)", index=d.index)
    hh = (d["Mineral_ToDepth"] - d["Mineral_FromDepth"]).clip(lower=0.1)
    for i, g in enumerate(sorted(rg.unique())):
        m = rg == g
        fig.add_trace(go.Bar(x=np.ones(int(m.sum())), y=d.loc[m, "Depth_Mid"], width=hh[m], orientation="h", name=g.split(" (")[0],
                             marker=dict(color=rock_color(g, i), line=dict(width=0)),
                             legendgroup="rock", hovertemplate="%{y:.1f} m<extra>" + g + "</extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=np.full(len(contacts), 1.0), y=contacts, mode="markers", marker=dict(symbol="triangle-left", size=11, color="black"),
                             name="logged contact", hovertemplate="%{y:.1f} m<extra>logged contact</extra>"), row=1, col=2)
    for _, r in z_h.iterrows():
        mid = (r["From_depth"] + r["To_depth"]) / 2
        fig.add_trace(go.Bar(x=[1], y=[mid], width=[r["To_depth"] - r["From_depth"]], orientation="h", showlegend=False,
                             marker=dict(color="#d9e2ef" if int(r["Zone"]) % 2 == 0 else "#f2f2f2", line=dict(color="white", width=1)),
                             text=f"Z{int(r['Zone'])}", textposition="inside",
                             hovertemplate=f"Zone {int(r['Zone'])}: {r['From_depth']:.0f}–{r['To_depth']:.0f} m<br>{r['Dominant_minerals']}<br>Logged: {r['Dominant_logged_rock']}<extra></extra>"), row=1, col=3)
    for depth in b_h:
        fig.add_hline(y=depth, line_dash="dash", line_color="crimson", line_width=1.2, row="all", col="all")
    fig.update_layout(barmode="overlay", height=720, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.04), bargap=0)
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2); fig.update_xaxes(showticklabels=False, row=1, col=3)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Zone summary for this hole**")
    st.dataframe(z_h[["Zone", "From_depth", "To_depth", "Intervals", "Dominant_minerals", "Dominant_logged_rock", "Dominant_cluster"]].round(1), use_container_width=True, hide_index=True)
    st.download_button("Download all zones", data=zone_tbl.to_csv(index=False).encode("utf-8-sig"), file_name="hylogger_detected_zones.csv", mime="text/csv")
    sweep_file = REPORT_DIR / "boundary_sensitivity.csv"
    if sweep_file.exists():
        with st.expander("How the result changes with the penalty setting (from the notebook)"):
            sw = pd.read_csv(sweep_file)
            st.dataframe(sw[sw["contact_level"] == "group"][["penalty_scale", "median_zones_per_hole", "detected_boundaries", "precision", "chance_precision", "recall", "chance_recall"]].round(2), hide_index=True, use_container_width=True)
    st.caption("A boundary counts as correct if a logged rock-group contact is within the tolerance. Logged contacts are frequent, so the random baseline matters: it shows how often that would happen by chance. Exploratory; the logged geology is the only reference.")
    with st.expander("Pipeline result (script 11): PELT boundaries vs logged lithology contacts, 5,000 random draws", expanded=True):
        v11 = pipe_csv("11_boundary_validation.csv")
        if v11 is not None:
            w11 = v11["our_boundaries"]
            b1, b2, b3 = st.columns(3)
            b1.metric("Tested boundaries", int(w11.sum()), f"{len(v11)} holes", delta_color="off")
            b2.metric("Within ±3 m of a logged contact", f"{(v11['ours_near_contact']*w11).sum()/w11.sum():.1%}", f"{(v11['ours_near_contact']*w11).sum():.0f} of {int(w11.sum())}", delta_color="off")
            b3.metric("Random boundaries, same rule", f"{(v11['random_near_contact']*w11).sum()/w11.sum():.1%}", "5,000 random placements; none matched ours", delta_color="off")
            st.dataframe(v11, hide_index=True, use_container_width=True)
            st.caption("Contacts here are any change in the logged lithology code (not only rock group), tested only at positions that have a code. "
                       "Robustness: with zones of 1-5 m and looser or stricter settings, the match stayed at 66-75% vs 40-48% for random boundaries.")

elif page == "Q4 Rock type":
    st.subheader("Can minerals tell the rock type?")
    st.caption("Script 10d: the 9 rock types the geologists actually named in our 65 holes. A random forest learns each rock's mineral recipe "
               "from all SWIR + TIR minerals; every hole is predicted by a model that never saw it.")
    sc10d, rec10d, lab10d = pipe_csv("10d_rock_type_scores.csv"), pipe_csv("10d_rock_type_recall.csv"), pipe_csv("10d_rock_type_labels.csv")
    if sc10d is None or rec10d is None:
        st.info("Run pipeline/10d_rock_types_from_minerals.py (part of run_all.sh) to see this result.")
    else:
        def _row(level, method):
            r = sc10d[(sc10d["level"] == level) & (sc10d["method"] == method)]
            return r.iloc[0] if len(r) else None
        g_h, t_h, t_c = _row("composition group", "Random forest - new HOLE"), _row("rock type", "Random forest - new HOLE"), _row("rock type", "Random forest - new COMPANY")
        m1, m2, m3, m4 = st.columns(4)
        if g_h is not None: m1.metric("Right composition group (5 groups)", f"{g_h['accuracy']:.0%}", help="felsic / intermediate / mafic / ultramafic / sedimentary")
        if t_h is not None: m2.metric(f"Exact rock type (chance {1 / len(rec10d):.0%})", f"{t_h['accuracy']:.0%}")
        if t_c is not None: m3.metric("Exact type, unseen company", f"{t_c['accuracy']:.0%}")
        m4.metric(f"Labelled metres ({len(rec10d)} rock types)", f"{int(rec10d['metres'].sum()):,}")
        pipe_image("10d_rock_types.png", "Left: share of each logged rock type recognised on unseen holes. Right: where the mistakes go.")
        show = rec10d.rename(columns={"rock_type": "Rock type", "group": "Composition group", "metres": "Metres", "holes": "Holes",
                                      "recall_new_hole": "Recognised (new hole)", "recall_new_company": "Recognised (new company)",
                                      "most_confused_with": "Most often called"})
        for c_ in ["Recognised (new hole)", "Recognised (new company)"]:
            show[c_] = (show[c_] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.markdown("**Why not higher?** (1) *Twin rocks*: basalt and dolerite, sandstone and siltstone, granite and felsic gneiss have the same "
                    "minerals and differ in grain size or texture, which a mineral scanner cannot see. (2) *Noisy labels*: about half the labels come "
                    "from geologists' comments, which can mention a neighbouring rock. (3) *Few examples*: intermediate rocks have only ~480 m.")
        if lab10d is not None and "tested" in lab10d:
            rare = lab10d[~lab10d["tested"].astype(bool)]
            if len(rare):
                st.caption("Not tested (too few holes): " + ", ".join(f"{r['rock_type']} ({int(r['metres'])} m)" for _, r in rare.iterrows()))

        @st.cache_data(show_spinner=False)
        def _pred10d(mtime):
            p = PIPE / "10d_rock_type_predictions.csv"
            return pd.read_csv(p, low_memory=False) if p.exists() else None
        pr10 = _pred10d((PIPE / "10d_rock_type_predictions.csv").stat().st_mtime if (PIPE / "10d_rock_type_predictions.csv").exists() else 0)
        if pr10 is not None:
            st.markdown("#### First-pass rock log, hole by hole")
            h_opts = sorted(pr10.loc[pr10["Logged_rock"].notna(), "Hole"].unique())
            h_pick = st.selectbox("Hole", h_opts, index=h_opts.index("LEFD006") if "LEFD006" in h_opts else 0, key="rt10d_hole")
            ph = pr10[pr10["Hole"] == h_pick].sort_values("From_m").copy()
            ph["Shown_prediction"] = ph["Pred_new_hole"].fillna(ph["Predicted_rock"])
            types = list(rec10d["rock_type"])
            fig = make_subplots(rows=1, cols=3, shared_yaxes=True, column_widths=[0.35, 0.35, 0.3], horizontal_spacing=0.03,
                                subplot_titles=("Geologist's log", "Predicted from minerals", "Model confidence"))
            for col_i, key in enumerate(["Logged_rock", "Shown_prediction"], 1):
                for i, t_ in enumerate(types):
                    m = ph[key] == t_
                    if m.any():
                        fig.add_trace(go.Bar(x=np.ones(int(m.sum())), y=ph.loc[m, "From_m"] + 0.5, width=1.0, orientation="h", name=t_,
                                             marker=dict(color=ROCK10D_COLORS.get(t_, px.colors.qualitative.T10[i % 10]), line=dict(width=0)),
                                             legendgroup=t_, showlegend=(col_i == 1 or not (ph["Logged_rock"] == t_).any()),
                                             hovertemplate="%{y:.0f} m<extra>" + t_ + "</extra>"), row=1, col=col_i)
            fig.add_trace(go.Scatter(x=ph["Confidence"], y=ph["From_m"] + 0.5, mode="lines", line=dict(color="#888", width=1), name="confidence",
                                     showlegend=False), row=1, col=3)
            fig.add_vline(x=0.5, line_dash="dash", line_color="#B5652B", row=1, col=3)
            fig.update_layout(barmode="overlay", height=650, bargap=0, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.12))
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
            fig.update_xaxes(showticklabels=False, row=1, col=1); fig.update_xaxes(showticklabels=False, row=1, col=2)
            fig.update_xaxes(range=[0, 1], title_text="top probability", row=1, col=3)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Logged metres show the prediction from a model that never saw this hole; metres without a named rock show the first-pass "
                       "prediction. Below the dashed line (probability < 0.5) the model is unsure — check those metres first.")
    st.divider()
    show_other_q4 = st.toggle("Show other approaches (GSWA's 3-class felsic / intermediate / mafic example, and the interactive model)", value=False)
    if show_other_q4:
        need_module()
        st.subheader("Can minerals predict the logged rock type?")
        st.caption("A random-forest model predicts the broad rock group from mineral features. Every prediction is for a hole the model never saw (leave-one-hole-out).")

        @st.cache_data(show_spinner="Testing the model on unseen holes (about 40 seconds, first time only)…", persist="disk")
        def run_rocktype(mtime):
            return hla.rocktype_loho(modelled, model_features)

        rt = run_rocktype(DATA_MTIME)
        a, b, c, d_ = st.columns(4)
        a.metric("Accuracy on unseen holes", f"{100*rt['model']['accuracy']:.0f}%")
        b.metric("Balanced accuracy", f"{100*rt['model']['balanced_accuracy']:.0f}%", "each rock group counts equally", delta_color="off")
        c.metric("Always guess commonest group", f"{100*rt['baseline_majority']['accuracy']:.0f}%")
        d_.metric("Earlier clusters only", f"{100*rt['baseline_cluster_only']['accuracy']:.0f}%")
        st.write(f"{rt['n_intervals']:,} intervals from {rt['n_holes']} holes, {len(rt['classes'])} rock groups. Veins, unknown lithology and undivided sediment are excluded.")
        with st.expander("Pipeline result (script 10): felsic / intermediate / mafic from TIR minerals, cut-offs calibrated leaving one hole out", expanded=True):
            cal = pipe_csv("10_rock_type_calibration_loho.csv")
            if cal is not None:
                p1, p2, p3 = st.columns(3)
                p1.metric("Agreement on unseen holes", f"{(cal['accuracy_on_held_out']*cal['n']).sum()/cal['n'].sum():.0%}")
                p2.metric("With fixed cut-offs", f"{(cal['accuracy_fixed_cutoffs']*cal['n']).sum()/cal['n'].sum():.0%}")
                p3.metric("Labelled metres / holes", f"{int(cal['n'].sum()):,} / {len(cal)}")
                lab = modelled[modelled["Logged_broad"].isin(["Felsic", "Intermediate", "Mafic"])] if "Logged_broad" in modelled else pd.DataFrame()
                if len(lab):
                    st.dataframe(pd.crosstab(lab["Logged_broad"], lab["HyLogger_broad"], normalize="index").round(2), use_container_width=True)
                    st.caption("Rows = geologist's broad class; columns = class from TIR minerals (fixed cut-offs, all labelled metres).")
            pipe_image("10b_pseudologs.png", "Pseudologs: rock class from minerals beside the geology log")

        t1, t2, t3, t4 = st.tabs(["Where it works", "Hole by hole", "Felsic / mafic / chlorite", "What drives it"])
        with t1:
            cm = rt["confusion"]; cmn = (cm.div(cm.sum(axis=1), axis=0) * 100).round(0)
            short = [x.split(" (")[0] for x in cm.index]
            fig = px.imshow(cmn.values, x=short, y=short, text_auto=".0f", color_continuous_scale="Blues", zmin=0, zmax=100,
                            labels=dict(x="Predicted from minerals", y="Logged rock group", color="% of row"))
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rt["per_class"].round(2), hide_index=True, use_container_width=True)
            st.caption("Rows = logged rock group; columns = group predicted from minerals on a hole the model never saw. Read the diagonal for what is recognised well.")
        with t2:
            ph = rt["per_hole"].sort_values("Accuracy")
            fig = px.bar(ph, x="Hole_name", y="Accuracy", labels={"Accuracy": "Accuracy on this unseen hole"})
            fig.update_yaxes(tickformat=".0%", range=[0, 1]); fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            hole = st.selectbox("Compare logged vs predicted for a hole", holes, key="rt_hole")
            pr = rt["predictions"][rt["predictions"]["Hole_name"].astype(str) == hole].sort_values("Depth_Mid")
            if pr.empty:
                st.info("No classified lithology for this hole.")
            else:
                groups = sorted(set(pr["Rock_Group"]) | set(pr["Predicted_group_smoothed"]))
                gcol = {g: rock_color(g, i) for i, g in enumerate(groups)}
                fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Logged rock group", "Predicted from minerals (smoothed)"), horizontal_spacing=0.03)
                for col_i, key in enumerate(["Rock_Group", "Predicted_group_smoothed"], 1):
                    for g in groups:
                        m = pr[key] == g
                        if m.any():
                            fig.add_trace(go.Bar(x=np.ones(int(m.sum())), y=pr.loc[m, "Depth_Mid"], width=1.0, orientation="h", name=g.split(" (")[0], marker=dict(color=gcol[g], line=dict(width=0)),
                                                 legendgroup=g, showlegend=(col_i == 1), hovertemplate="%{y:.1f} m<extra>" + g + "</extra>"), row=1, col=col_i)
                fig.update_layout(barmode="overlay", height=620, bargap=0, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.05))
                fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1); fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True)
        with t3:
            tern_all = hla.ternary_table(modelled).dropna(subset=["Rock_Group"])
            counts = tern_all["Rock_Group"].value_counts()
            order = [g for g in ROCK_COLORS if g in counts.index] + [g for g in counts.index if g not in ROCK_COLORS]
            c1, c2 = st.columns([3, 1])
            show_groups = c1.multiselect("Rock groups to show", order, default=order,
                                         format_func=lambda g: f"{g} ({counts[g]:,} m)", key="tern_groups")
            per_group = c2.slider("Max points per group", 200, 3000, 800, 100, key="tern_n",
                                  help="Same cap for every group, so the common mafic rocks do not hide the rarer groups.")
            layout_mode = st.radio("Layout", ["All groups in one triangle", "One triangle per group"], horizontal=True, key="tern_layout")
            tern = pd.concat([tern_all[tern_all["Rock_Group"] == g].sample(min(per_group, int(counts[g])), random_state=0)
                              for g in show_groups]) if show_groups else tern_all.iloc[:0]
            # largest group drawn first, rarest last, so small groups stay visible on top
            draw = sorted(show_groups, key=lambda g: -counts[g])
            def tern_trace(g, i, show_centre=True, legend=True):
                q = tern[tern["Rock_Group"] == g]
                pts = go.Scatterternary(a=q["Felsic"], b=q["Mafic"], c=q["Chlorite"], mode="markers", name=g.split(" (")[0],
                                        legendgroup=g, showlegend=legend,
                                        marker=dict(size=5, color=rock_color(g, i), opacity=0.75, line=dict(width=0.3, color="rgba(0,0,0,0.5)")),
                                        customdata=q[["Hole_name", "Depth_Mid"]],
                                        hovertemplate="<b>" + g + "</b><br>%{customdata[0]} @ %{customdata[1]:.1f} m<br>"
                                                      "Felsic %{a:.2f} · Mafic %{b:.2f} · Chlorite %{c:.2f}<extra></extra>")
                ctr = tern_all[tern_all["Rock_Group"] == g][["Felsic", "Mafic", "Chlorite"]].median()
                centre = go.Scatterternary(a=[ctr["Felsic"]], b=[ctr["Mafic"]], c=[ctr["Chlorite"]], mode="markers", legendgroup=g, showlegend=False,
                                           marker=dict(size=16, symbol="diamond", color=rock_color(g, i), line=dict(width=2, color="white")),
                                           hovertemplate="<b>" + g + " - median</b><br>Felsic %{a:.2f} · Mafic %{b:.2f} · Chlorite %{c:.2f}<extra></extra>")
                return [pts, centre] if show_centre else [pts]
            axes = dict(aaxis=dict(title="Felsic"), baxis=dict(title="Mafic"), caxis=dict(title="Chlorite"))
            if layout_mode == "All groups in one triangle":
                fig = go.Figure()
                for i, g in enumerate(draw):
                    for t in tern_trace(g, order.index(g)): fig.add_trace(t)
                fig.update_layout(ternary=axes, height=600, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.08, itemsizing="constant"))
            else:
                ncol = 3; nrow = max(1, -(-len(show_groups) // ncol))
                fig = make_subplots(rows=nrow, cols=ncol, specs=[[{"type": "ternary"}] * ncol] * nrow,
                                    subplot_titles=[f"{g.split(' (')[0]} ({counts[g]:,} m)" for g in show_groups],
                                    vertical_spacing=0.14, horizontal_spacing=0.1)
                for k, g in enumerate(show_groups):
                    for t in tern_trace(g, order.index(g), legend=False): fig.add_trace(t, row=k // ncol + 1, col=k % ncol + 1)
                fig.update_ternaries(aaxis=dict(title=dict(text="Felsic", font=dict(size=11))),
                                     baxis=dict(title=dict(text="Mafic", font=dict(size=11))),
                                     caxis=dict(title=dict(text="Chl", font=dict(size=11))))
                fig.update_layout(height=360 * nrow, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Felsic = silica + K-feldspar, mafic = pyroxene + amphibole (TIR), chlorite = chlorite (SWIR). Each colour is one rock group "
                       "(same colours on every page); the large diamond is that group's median position. Each group is capped at the same number "
                       "of points so the common mafic rocks do not hide the others. Click legend items to hide or show a group. Sedimentary rocks sit "
                       "at the silica corner; mafic rocks and schists overlap, which matches the confusion above.")
        with t4:
            imp = rt["importance"].head(12).iloc[::-1]
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h"); fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Importance shows which features the model relied on, not which cause the rock type. Rock groups come from the GSWA rock name where logged, otherwise the company's rock name.")

elif page == "Q5 Geochemistry":
    st.subheader("Can minerals estimate the rock's chemistry?")
    st.caption("Lab assays cost money for every sample. Minerals are made of elements, so we test whether HyLogger minerals can estimate "
               "the chemistry — always on companies the model never saw (scripts 16 and 16b; PLS, as GSWA suggested).")
    s16, mg16 = pipe_csv("16_bulk_prediction_scores.csv"), pipe_csv("16b_mgo_predictions.csv")
    if s16 is None:
        st.info("Run pipeline/16_bulk_property_prediction.py (part of run_all.sh) to see this result.")
    else:
        comp = s16[(s16["model"] == "PLS") & (s16["held_out"] == "company")].set_index("element")
        tier = pd.cut(comp["skill"], [-np.inf, 0.3, 0.5, np.inf], labels=["No", "Partly", "Yes"], right=False)
        k1, k2, k3 = st.columns(3)
        if "MgO" in comp.index:
            k1.metric("MgO: typical error vs the lab", f"within {comp.at['MgO', 'median_rel_error']:.0%}", help="If the lab says 9% MgO, we say about 6-12%.")
            k2.metric("MgO: R² on an unseen company", f"{comp.at['MgO', 'skill']:.2f}", help="1 = perfect, 0 = no better than always guessing the average.")
        k3.metric(f"Elements that work (of {len(comp)})", f"{int((tier == 'Yes').sum())} + {int((tier == 'Partly').sum())} partly")
        pipe_image("16b_mgo_example.png", "MgO from minerals vs the lab, for companies the model never saw. Left: every assayed metre. Right: one example hole.")
        WHY = {"MgO": "in amphibole, chlorite, serpentine (seen well)", "K2O": "in mica and K-feldspar", "Rb": "follows potassium",
               "Na2O": "in plagioclase", "Ba": "follows potassium feldspar and mica", "Al": "in clays, mica, feldspar",
               "Ni": "travels with magnesium-rich rocks (indirect)", "Cr": "travels with magnesium-rich rocks (indirect)",
               "Co": "travels with magnesium-rich rocks (indirect)", "Au": "in sulphides and veins - invisible to infrared",
               "Cu": "in sulphides - invisible to infrared", "As": "in sulphides - invisible to infrared",
               "S": "in sulphides - invisible to infrared", "FeO": "much is in magnetite / sulphides, barely seen"}
        tl = pd.DataFrame({"Works?": tier.astype(str), "Element": comp.index, "R² (unseen company)": comp["skill"].round(2),
                           "Typical error": (comp["median_rel_error"] * 100).round(0).astype(int).astype(str) + "%",
                           "Why": [WHY.get(e, "") for e in comp.index]}).sort_values("R² (unseen company)", ascending=False)
        st.dataframe(tl, hide_index=True, use_container_width=True, height=420)
        if mg16 is not None:
            st.markdown("#### MgO down a hole: lab vs prediction")
            mg_opts = sorted(mg16["Hole"].unique())
            hp = st.selectbox("Hole", mg_opts, index=mg_opts.index("THDD024") if "THDD024" in mg_opts else 0, key="mg16_hole")
            q = mg16[mg16["Hole"] == hp].sort_values("From_m")
            fig = go.Figure([go.Scatter(x=q["MgO_PCT"], y=q["From_m"], mode="lines", name="Lab assay", line=dict(color="#1F2A2E")),
                             go.Scatter(x=q["MgO_predicted"], y=q["From_m"], mode="lines", name="Predicted from minerals", line=dict(color="#B5652B"))])
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)"); fig.update_xaxes(type="log", title_text="MgO (%)")
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Limits: MgO is easiest where rocks vary a lot (without our ultramafic holes R² drops from 0.75 to 0.63). These are screening "
                   "estimates for unassayed metres and for choosing lab samples, not a replacement for the lab.")
    st.divider()
    show_other_q5 = st.toggle("Show the interactive, region-by-region PLS run (exploratory)", value=False)
    if show_other_q5:
        need_module()
        st.subheader("Can minerals predict assay values?")
        st.caption("PLS regression predicts assay values (log10) from mineral features, tested on holes the model never saw. Geophysics (density, magnetics, conductivity) is listed as TBC in the index for every hole, so only geochemistry could be tested.")

        @st.cache_data(show_spinner="Fitting PLS models on unseen holes (1–2 minutes on 65 holes, first time only)…", persist="disk")
        def run_pls(mtime):
            subs = {"All holes": pd.Series(True, index=modelled.index)}
            if "Report_key" in modelled:
                for rk, g in modelled.groupby("Report_key"):
                    if g["Hole_name"].nunique() >= 10:   # small regions (5-6 holes) give unstable, misleading scores
                        subs[(f"Report {rk} only" if config.get("report_key_kind", "report") == "report" else f"{rk} only")] = modelled["Report_key"] == rk
            return hla.pls_loho(modelled, model_features, subsets=subs)

        pls_sum, pls_pred = run_pls(DATA_MTIME)
        if pls_sum.empty:
            st.warning("Not enough matched assay samples to fit models.")
            st.stop()
        if config.get("report_key_kind", "report") == "report" and "Report_key" in modelled.columns and "AssayFlatDataId" in modelled.columns:
            mt = modelled[modelled["Assay_Match"].fillna(False).astype(bool)].drop_duplicates("AssayFlatDataId")
            mg = mt.groupby("Report_key")["Assay_MgO_PCT"].median()
            ca = mt.groupby("Report_key")["Assay_CaO_PCT"].median()
            if len(mg) > 1 and mg.max() > 0 and mg.max() / max(mg.min(), 1e-9) > 100:
                st.warning("Data-quality flag: median MgO differs by more than 100× between reports (" + ", ".join(f"report {k}: {v:.4g}%" for k, v in mg.items()) +
                           ") while median CaO is comparable (" + ", ".join(f"{v:.2f}%" for v in ca.values) + "). That points to a unit or labelling problem in the Mg field of one report, so MgO from the all-holes run is unreliable. Compare the single-report results.")
        best = pls_sum.sort_values("R2_unseen_holes", ascending=False).iloc[0]
        st.info("Exploratory: leave-one-hole-out PLS on all holes and on regions with at least 10 holes. Use the validated result above for reporting.")
        fig = px.bar(pls_sum, x="R2_unseen_holes", y="Target", color="Subset", orientation="h", barmode="group",
                     labels={"R2_unseen_holes": "R² on unseen holes (0 = no better than the average, 0.5 = useful)"})
        fig.add_vline(x=0.5, line_dash="dash", line_color="green"); fig.update_xaxes(range=[-0.6, 1]); fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        sub_pick = c1.selectbox("Subset", sorted(pls_sum["Subset"].unique()), index=sorted(pls_sum["Subset"].unique()).index(best["Subset"]))
        tg_opts = pls_sum[pls_sum["Subset"] == sub_pick]["Target"].tolist()
        tg_pick = c2.selectbox("Element / oxide", tg_opts, index=tg_opts.index(best["Target"]) if best["Subset"] == sub_pick and best["Target"] in tg_opts else 0)
        p = pls_pred[(pls_pred["Subset"] == sub_pick) & (pls_pred["Target"] == tg_pick)]
        row = pls_sum[(pls_sum["Subset"] == sub_pick) & (pls_sum["Target"] == tg_pick)].iloc[0]
        fig = px.scatter(p, x="Measured_log10", y="Predicted_log10", color="Hole_name", hover_data=["Depth"], opacity=0.7,
                         labels={"Measured_log10": "Measured assay (log10)", "Predicted_log10": "Predicted from minerals (log10)"})
        lo, hi = float(min(p["Measured_log10"].min(), p["Predicted_log10"].min())), float(max(p["Measured_log10"].max(), p["Predicted_log10"].max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="black", width=1))
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=30, b=10), title=f"{tg_pick}, {sub_pick}: R² {row['R2_unseen_holes']:.2f}, Spearman {row['Spearman_rho']:.2f}, {int(row['Holes'])} holes")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pls_sum.sort_values("R2_unseen_holes", ascending=False).round(2), hide_index=True, use_container_width=True)
        st.caption("Interactive PLS above: leave-one-hole-out on the cleaned assays from script 03. Assays are attached to each HyLogger metre from the shortest covering assay interval.")
        with st.expander("Pipeline result (script 16): 34 elements, PLS and boosting, held out by hole and by company", expanded=True):
            sc16 = pipe_csv("16_bulk_prediction_scores.csv")
            if sc16 is not None:
                q1_, q2_ = st.columns(2)
                held = q1_.radio("Tested on", ["company", "hole"], format_func=lambda x: {"company": "unseen companies", "hole": "unseen holes"}[x], horizontal=True, key="p16_held")
                mdl = q2_.radio("Model", sorted(sc16["model"].unique(), key=lambda v: v != "PLS"), horizontal=True, key="p16_model")
                t16 = sc16[(sc16["held_out"] == held) & (sc16["model"] == mdl)].sort_values("skill")
                fig = go.Figure(go.Bar(y=t16["element"] + " (" + t16["unit"] + ")", x=t16["skill"].clip(lower=-0.2), orientation="h",
                                       marker_color=np.where(t16["skill"] >= 0.3, "#2a78d6", "#cde2fb"), hovertemplate="%{y}: skill %{x:.2f}<extra></extra>"))
                fig.add_vline(x=0.3, line_dash="dash", line_color="#777")
                fig.update_layout(height=max(500, 20 * len(t16)), xaxis_title="Skill vs predicting the training average (clipped at -0.2); dashed = 0.3", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Rock-forming elements carried by HyLogger-visible minerals predict well (MgO, Rb, Cr, Ni, K2O ...); gold, copper, arsenic and sulphur sit in sulphides the scanner cannot see.")

elif page == "Q1 Endmembers & anomalies":
    need_module()
    st.subheader("Endmembers and explained anomalies")
    st.caption("Non-negative matrix factorisation expresses every interval as a mix of a few 'pure' mineral assemblages (endmembers).")
    k_choice = st.select_slider("Number of endmembers", options=["Auto"] + list(range(3, 9)), value="Auto")

    @st.cache_data(show_spinner="Extracting endmembers…")
    def run_nmf(k_, mtime):
        return hla.nmf_endmembers(modelled, k=None if k_ == "Auto" else int(k_))

    em = run_nmf(k_choice, DATA_MTIME)
    ab = em["abundance"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Endmembers", em["k"])
    m2.metric("Median interval purity", f"{ab['Endmember_purity'].median():.2f}", "share of the largest endmember", delta_color="off")
    m3.metric("Poorly explained intervals", f"{int(ab['Unexplained'].sum()):,}", "top 3% residual", delta_color="off")
    c1, c2 = st.columns([1, 2.2])
    with c1:
        ec = em["error_curve"]
        fig = px.line(ec, x="k", y=ec["relative_error"] * 100, markers=True, labels={"y": "Unexplained signal (%)", "k": "Endmembers"})
        fig.add_vline(x=em["k"], line_dash="dash", line_color="crimson"); fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="Choosing the number")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        comp = em["composition"]; rows_ = comp.max(axis=1).sort_values(ascending=False).head(14).index
        fig = px.imshow(comp.loc[rows_].T.values, x=list(rows_), y=[x[:44] for x in comp.columns], color_continuous_scale="Viridis", zmin=0, zmax=1, aspect="auto")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), title="What each endmember is made of")
        st.plotly_chart(fig, use_container_width=True)
    hole = st.selectbox("Hole", holes, key="em_hole")
    dd = pd.concat([modelled[["Hole_name", "Depth_Mid", "Mineral_FromDepth", "Mineral_ToDepth"]], ab], axis=1)
    dd = dd[dd["Hole_name"].astype(str) == hole].sort_values("Depth_Mid")
    share_cols = [c for c in dd.columns if c.endswith("_share")]
    fig = go.Figure(stacked_mineral_traces(dd, share_cols, label_fn=lambda c: em["labels"][int(c.split("_")[0][2:]) - 1][:48]))
    fig.update_layout(barmode="overlay", height=640, bargap=0, margin=dict(l=10, r=10, t=30, b=10), title=f"{hole}: endmember mixture down the hole", legend=dict(orientation="h", y=-0.08))
    fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Anomalies, with the reason each was flagged")
    if "IsAnomaly" in modelled.columns:
        an = modelled[modelled["IsAnomaly"].fillna(False).astype(bool)].copy()
        an["Why_flagged"] = cached_reasons(DATA_MTIME).loc[an.index]
        an["Also_poorly_explained_by_endmembers"] = ab.loc[an.index, "Unexplained"].to_numpy()
        cols_ = [c for c in ["Hole_name", "Depth_Mid", "Cluster", "Geo_GSWARockName", "Why_flagged", "Also_poorly_explained_by_endmembers", "AnomalyScore"] if c in an.columns]
        st.write(f"{len(an):,} anomaly flags; {int(an['Also_poorly_explained_by_endmembers'].sum())} are also poorly explained by the endmembers, so the two methods catch different things.")
        st.dataframe(an.sort_values("AnomalyScore")[cols_], hide_index=True, use_container_width=True, height=320)
    st.caption("Endmembers are statistical mineral assemblages from the TSG summary logs, not pure spectra. Name them with a geologist before reporting.")
    with st.expander("Pipeline result (script 14): 7 endmembers, number chosen by refit stability, within-hole gold tests", expanded=True):
        e14 = pipe_csv("14_endmembers.csv")
        if e14 is not None:
            show = [c for c in ["endmember", "mean_share", "metres_dominant(>=50%)", "holes_with_dominant", "refits_matched(cos>=0.9)",
                                "gold_holes_pos", "gold_holes_neg", "gold_sign_test_q_BH"] if c in e14]
            st.dataframe(e14[show], hide_index=True, use_container_width=True)
        pipe_image("14a_endmembers.png")

elif page == "Q2 Spectral features":
    need_module()
    st.subheader("Absorption-feature positions beyond the mineral labels")
    st.info("The top of this page maps the feature positions TSG already extracted (summary logs, all holes). The bottom shows the pipeline's own hull-quotient analysis of raw spectra for hole BWD072 (script 15).")
    hole = st.selectbox("Hole", holes, key="sf_hole")
    d = modelled[(modelled["Hole_name"].astype(str) == hole) & modelled["Depth_QC_Valid"].fillna(False).astype(bool)].sort_values("Depth_Mid")
    fams = [(f, w, dcol) for f, (w, dcol) in hla.FEATURE_FAMILIES.items() if w in d.columns and dcol in d.columns]
    fig = make_subplots(rows=1, cols=len(fams), shared_yaxes=True, subplot_titles=[f[0] for f in fams], horizontal_spacing=0.04)
    for i, (fam, w, dcol) in enumerate(fams, 1):
        wv, dv = pd.to_numeric(d[w], errors="coerce"), pd.to_numeric(d[dcol], errors="coerce")
        ok = wv.notna()
        fig.add_trace(go.Scatter(x=wv[ok], y=d.loc[ok, "Depth_Mid"], mode="markers", showlegend=False,
                                 marker=dict(size=6, color=dv[ok], colorscale="Turbo", showscale=(i == len(fams)), colorbar=dict(title="feature depth")),
                                 hovertemplate="%{x:.1f} nm at %{y:.1f} m<extra></extra>"), row=1, col=i)
        fig.update_xaxes(title_text="wavelength (nm)", row=1, col=i)
    fig.update_yaxes(autorange="reversed", row=1, col=1, title_text="Depth (m)")
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Do feature positions differ between rock groups?")

    @st.cache_data(show_spinner=False)
    def wl_table(mtime):
        return hla.wavelength_by_group(modelled)

    wl = wl_table(DATA_MTIME)
    fam_pick = st.selectbox("Feature", sorted(wl["Feature"].unique()), index=sorted(wl["Feature"].unique()).index("Al-OH (~2200 nm)") if "Al-OH (~2200 nm)" in set(wl["Feature"]) else 0)
    sub = wl[wl["Feature"] == fam_pick].sort_values("Median_nm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Median_nm"], y=sub["Rock_group"].str.split(" \\(").str[0], mode="markers", marker=dict(size=10, color="black"),
                             error_x=dict(type="data", symmetric=False, array=(sub["P75_nm"] - sub["Median_nm"]).to_numpy(), arrayminus=(sub["Median_nm"] - sub["P25_nm"]).to_numpy(), thickness=4, color="#3b6ea5"),
                             hovertemplate="%{y}: median %{x:.1f} nm<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Wavelength (nm): median and middle 50%")
    st.plotly_chart(fig, use_container_width=True)
    pval = float(sub["Kruskal_p_across_groups"].iloc[0]) if len(sub) else np.nan
    st.caption(f"Kruskal–Wallis p across groups: {pval:.2g}. With thousands of intervals a small shift is statistically significant, so look at the size of the shift. Wavelength shifts are commonly used as composition indicators (for example white-mica Al-OH position), but interpreting them needs expert review.")
    st.markdown("### Outside TSG: raw spectra of BWD072 (script 15)")
    st.write("76,160 SWIR spectra read with the open-source `pytsg` reader; white-mica absorption found with our own hull-quotient method "
             "(deepest minimum in 2180–2235 nm, parabola refinement). Against TSG's own feature: position r = 0.90, median difference 1.1 nm; "
             "mineral order paragonite < muscovite < phengite reproduced.")
    shown = [pipe_image(f) for f in ["15a_example_spectra.png", "15b_feature_map.png", "15c_tsg_check_and_gold.png"]]
    if not any(shown):
        st.caption("Run the optional script 15 (needs the BWD072 TSG files and pytsg) to show these figures.")

elif page == "Downhole tracks (pipeline)":
    st.subheader("Downhole tracks: minerals, endmembers, assemblage, rock class and assays side by side")
    CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#8a5a2b", "#0097a7", "#6d6d00", "#a23b72"]
    ROCK_COL = {"Felsic": "#e34948", "Intermediate": "#eda100", "Mafic": "#2a78d6", "Mixed": "#a23b72",
                "Unclassified (low signal)": "#cfcec9", "not logged": "#efeeea"}
    SW = [c for c in ["White-Mica", "Chlorite", "Amphibole", "Dark-Mica", "Kaolin", "Carbonate", "Epidote", "Smectite"] if f"{c}_(System_SWIR)" in modelled]
    TI = [c for c in ["Silica", "K-Feldspar", "Plagioclase", "Amphibole", "Chlorite", "White-Mica", "Carbonate", "Pyroxene"] if f"{c}_(System_TIR)" in modelled]

    @st.cache_data(show_spinner=False)
    def track_frame(mtime):
        d = modelled[modelled["Depth_QC_Valid"].fillna(False).astype(bool)].copy()
        try:
            from cluster_names import cluster_names
            ours = d.rename(columns=lambda c: ("SWIR_" + c[:-14]) if str(c).endswith("_(System_SWIR)") else (("TIR_" + c[:-13]) if str(c).endswith("_(System_TIR)") else c))
            names = cluster_names(ours.dropna(subset=["Cluster"]))
        except Exception:
            names = {}
        d["Cluster_name"] = d["Cluster"].map(lambda k: f"{int(k)}: {names.get(int(k), '')}" if pd.notna(k) else "n/a")
        d["Logged_broad"] = d["Logged_broad"].fillna("not logged") if "Logged_broad" in d else "not logged"
        return d

    D = track_frame(DATA_MTIME)
    em_cols = [c for c in D.columns if c.startswith("Pipeline_E")]
    e14 = pipe_csv("14_endmembers.csv")
    em_label = {("Pipeline_" + r.split(":")[0]): r for r in e14["endmember"]} if e14 is not None else {}
    hole_list = sorted(D["Hole_name"].astype(str).unique())
    c1, c2, c3 = st.columns(3)
    hole = c1.selectbox("Hole", hole_list, index=hole_list.index("LEFD006") if "LEFD006" in hole_list else 0, key="trk_hole")
    g = D[D["Hole_name"].astype(str) == hole].sort_values("Mineral_FromDepth")
    avail = ["(none)"] + [c for c in assay_cols if g[c].notna().any()]
    assay = c2.selectbox("Assay track", avail, index=avail.index("Assay_Au_PPM") if "Assay_Au_PPM" in avail else 0, key="trk_assay")
    lo, hi = float(g["Mineral_FromDepth"].min()), float(g["Mineral_ToDepth"].max())
    rng = c3.slider("Depth (m)", lo, hi, (lo, hi), key="trk_rng") if hi > lo else (lo, hi)
    g = g[(g["Mineral_ToDepth"] >= rng[0]) & (g["Mineral_FromDepth"] <= rng[1])]
    gap = g["Mineral_FromDepth"].shift(-1) - g["Mineral_ToDepth"] > 1.5
    zc = [f"{c}_(System_SWIR)" for c in SW] + [f"{c}_(System_TIR)" for c in TI] + em_cols
    pads = pd.concat([pd.DataFrame({"Depth_Mid": g.loc[gap, "Mineral_ToDepth"] + 0.01}), pd.DataFrame({"Depth_Mid": g["Mineral_FromDepth"].shift(-1)[gap] - 0.01})])
    for c in zc:
        pads[c] = 0.0
    ga = pd.concat([g[["Depth_Mid"] + zc], pads]).sort_values("Depth_Mid")
    tracks = ["SWIR minerals", "TIR minerals"] + (["Endmembers"] if em_cols else []) + ["Assem-<br>blage", "Rock<br>HyLogger", "Rock<br>logged"] + ([assay.replace("Assay_", "")] if assay != "(none)" else [])
    fig = make_subplots(rows=1, cols=len(tracks), shared_yaxes=True, subplot_titles=tracks, horizontal_spacing=0.02,
                        column_widths=[3, 3] + ([3] if em_cols else []) + [1, 1, 1] + ([2.5] if assay != "(none)" else []))

    def area(col_i, names, suffix, group, pal):
        for i, nm in enumerate(names):
            c = f"{nm}_(System_{suffix})"
            if ga[c].fillna(0).max() <= 0.02:
                continue
            fig.add_trace(go.Scatter(x=ga[c].fillna(0), y=ga["Depth_Mid"], stackgroup=group, orientation="h", mode="lines", line_width=0,
                                     fillcolor=pal[i % len(pal)], name=nm, legendgroup=group, legendgrouptitle_text=group,
                                     hovertemplate=f"{nm}: %{{x:.2f}}<br>%{{y:.1f}} m<extra></extra>"), row=1, col=col_i)
        fig.update_xaxes(range=[0, 1], row=1, col=col_i)

    def strip(col_i, cat_col, colours, group, legend=True):
        for i, (cat, gg) in enumerate(g.groupby(cat_col, sort=True)):
            colour = colours.get(cat, "#cfcec9") if isinstance(colours, dict) else colours[i % len(colours)]
            fig.add_trace(go.Bar(x=np.ones(len(gg)), y=gg["Depth_Mid"], width=(gg["Mineral_ToDepth"] - gg["Mineral_FromDepth"]), orientation="h",
                                 marker_color=colour, marker_line_width=0, name=str(cat), legendgroup=group, legendgrouptitle_text=group if legend else None,
                                 showlegend=legend, hovertemplate=f"{cat}<br>%{{y:.1f}} m<extra></extra>"), row=1, col=col_i)
        fig.update_xaxes(showticklabels=False, range=[0, 1], row=1, col=col_i)

    col = 1
    area(col, SW, "SWIR", "SWIR", CAT); col += 1
    area(col, TI, "TIR", "TIR", CAT[3:] + CAT[:3]); col += 1
    if em_cols:
        for i, e in enumerate(em_cols):
            fig.add_trace(go.Scatter(x=ga[e].fillna(0), y=ga["Depth_Mid"], stackgroup="EM", orientation="h", mode="lines", line_width=0,
                                     fillcolor=CAT[i % len(CAT)], name=em_label.get(e, e)[:38], legendgroup="Endmembers", legendgrouptitle_text="Endmembers",
                                     hovertemplate=f"{em_label.get(e, e)}: %{{x:.2f}}<br>%{{y:.1f}} m<extra></extra>"), row=1, col=col)
        col += 1
    strip(col, "Cluster_name", CAT, "Assemblage"); asm_col = col; col += 1
    strip(col, "HyLogger_broad", ROCK_COL, "Rock class"); col += 1
    strip(col, "Logged_broad", ROCK_COL, "Rock class", legend=False); col += 1
    if assay != "(none)":
        x = g[assay]
        fig.add_trace(go.Scatter(x=x, y=g["Depth_Mid"], mode="lines+markers", marker_size=3, line_width=1, line_color="#0b0b0b", showlegend=False,
                                 hovertemplate=f"{assay}: %{{x}}<br>%{{y:.1f}} m<extra></extra>"), row=1, col=col)
        if x.dropna().gt(0).all() and len(x.dropna()) and x.max() / max(x[x > 0].min(), 1e-9) > 100:
            fig.update_xaxes(type="log", row=1, col=col)
    b11 = pipe_csv("11_boundaries.csv")
    if b11 is not None:
        for y in b11.loc[(b11["Hole"] == hole) & b11["Boundary_m"].between(*rng), "Boundary_m"]:
            fig.add_hline(y=y, line_color="#0b0b0b", line_width=1.2, line_dash="dot")
    an = g[g["IsAnomaly"].fillna(False).astype(bool)] if "IsAnomaly" in g else g.iloc[:0]
    if len(an):
        fig.add_trace(go.Scatter(x=np.full(len(an), 0.5), y=an["Depth_Mid"], mode="markers", name="Unusual metre", marker=dict(symbol="x", size=7, color="#0b0b0b"),
                                 text=an["Anomaly_reason"], hovertemplate="%{text}<br>%{y:.1f} m<extra></extra>"), row=1, col=asm_col)
    fig.update_yaxes(range=[rng[1] + 2, rng[0] - 2], title_text="Depth (m)", row=1, col=1)
    fig.update_layout(height=900, barmode="overlay", bargap=0, margin=dict(l=40, r=10, t=60, b=20), legend=dict(groupclick="toggleitem", font_size=11))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted lines = pipeline boundaries (script 11). × = unusual metre (script 07; hover for why). Rock HyLogger = class from TIR minerals (script 10); "
               "Rock logged = broad class from the geology log. Mineral shares are TSA algorithm interpretations, not measured percentages.")

elif page == "Data quality":
    st.subheader("Assay problems found and set aside before any model saw them (script 03)")
    st.write("The public assay table merges several lab methods into one column per element and does not say which method each row came from. "
             "Script 03 detects each case below, sets the doubtful values aside (never 'corrects' them) and prints what it removed in run_log.txt.")
    st.table(pd.DataFrame([
        ["Handheld XRF rows beside lab assays", "12DHDD0001/0002, 408 rows", "metadata 'Niton'; Au up to 257 ppm vs fire assay max 0.17", "rows removed"],
        ["Second 'Au' from a multi-element package", "KBD089 (801), KBD094 (146)", "whole numbers 1-35; Spearman -0.03 with fire assay", "that Au ignored; fire assay kept"],
        ["Leach recovery % stored as gold", "18KLDD008", "50-100 below 330 m, median 75.85", "gold set to missing for the hole"],
        ["-5000 used as 'no value'", "ETG0202/0204/0226, 1,579 rows", "would read as 2,500 ppm if treated as below detection", "set to missing (with -999, -9999)"],
        ["One row covering a whole hole", "ABND198", "0-308.8 m, As 31%", "removed"],
        ["Impossible values", "KPDDH003", "S up to 99.8% (pure pyrite ~53%)", "flagged, not used in results"],
    ], columns=["Problem", "Where", "Evidence", "Handling"]))
    st.caption("Effect: gold metres fell from 489 to 392 and every headline result held. None of these is in GSWA's own meta_data_quality table.")
    st.subheader("HyLogger mineral logs (scripts 01-02)")
    st.table(pd.DataFrame([
        ["Not proper drill-core scans", "6 of 71 holes (GBD011_chips; MRDH400-407 with no automated results and no TIR)", "excluded by rule, not by name"],
        ["Blank mineral cells", "all holes", "checked that each metre's shares sum to 1, then blank = not detected = 0"],
        ["Poor readings", "13.7% of metres (SWIR invalid > 50%)", "flagged and kept out of the models"],
        ["Same column, two spellings", "INVALID_(System_TIR) in 49 files, Invalid_(System_TIR) in 16", "merged into one column"],
    ], columns=["Problem", "Where", "Handling"]))

elif page == "Anomaly review":
    st.subheader("Intervals flagged as unusual")
    if "IsAnomaly" not in modelled.columns:
        st.warning("No anomaly flags were generated.")
        st.stop()
    flagged = modelled[modelled["IsAnomaly"].fillna(False).astype(bool)].copy()
    flagged = flagged.sort_values("AnomalyScore", ascending=False)
    if hla is not None:
        flagged["Why_flagged"] = cached_reasons(DATA_MTIME).loc[flagged.index]
    c1, c2 = st.columns(2)
    with c1:
        all_holes = st.checkbox("All holes", value=True)
    with c2:
        chosen_hole = None if all_holes else st.selectbox("Hole", holes)
    if chosen_hole:
        flagged = flagged[flagged["Hole_name"].astype(str).eq(chosen_hole)]
    display_cols = [c for c in ["Hole_name", "Depth_Mid", "Mineral_FromDepth", "Mineral_ToDepth", "AnomalyScore", "Anomaly_reason", "Why_flagged", "Cluster", "Geo_GSWARockName", "Assay_Au_PPM", "Assay_Cu_PPM", "Assay_Ni_PPM", "Assay_Match_Distance", "MINERALLOG_URL"] if c in flagged.columns]
    st.write(f"Flagged intervals in this view: {len(flagged):,}")
    st.dataframe(flagged[display_cols], use_container_width=True, height=450)
    st.download_button("Download flagged intervals", data=flagged[display_cols].to_csv(index=False).encode("utf-8-sig"), file_name="hylogger_anomaly_review.csv", mime="text/csv")
    st.caption("Anomaly scores are relative to this model/cohort. Flags can reflect real geology, measurement quality, or data-processing artefacts.")

else:
    st.subheader("Export & provenance")
    st.write("**Feature mode:**", config.get("feature_mode", "unknown"))
    st.write("**Features used in the model:**")
    st.code("\n".join(config.get("model_feature_columns", [])) or "No feature list available")
    st.write("**PCA explained variance ratio:**", config.get("pca_explained_variance_ratio", []))
    st.write("**Adjusted Rand Index vs GSWA rock names:**", config.get("adjusted_rand_index_vs_gswa_rockname", "not available"))
    st.warning(config.get("warning", "Explore and validate the findings before interpreting them."))
    manifest_path = REPORT_DIR / "mineral_log_download_manifest.csv"
    schema_path = REPORT_DIR / "mineral_log_schema_audit.csv"
    feat_path = REPORT_DIR / "model_feature_audit.csv"
    for label, path in [("Download manifest", manifest_path), ("Mineral-log schema audit", schema_path), ("Model feature audit", feat_path)]:
        if path.exists():
            st.markdown(f"**{label}:** `{path}`")
            data = path.read_bytes()
            st.download_button(f"Download {label}", data=data, file_name=path.name, mime="text/csv", key=label)
    st.markdown("""
    **Known limitations**
    - Minerals are linked to supporting WAMEX data by collar alias and midpoint/depth; source sampling intervals may differ.
    - Nearest assay matching uses a configured tolerance and records match distance; it is not exact sample identity.
    - Negative assay values are treated as non-numeric sentinel/non-detect values for the `Assay_*` analysis tracks while raw source columns are retained.
    - The cluster count is an exploratory setting. Compare settings and test stability before reporting results.
    - Geology labels are used for independent comparison, not as clustering features.
    """)
