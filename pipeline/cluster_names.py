"""Shared cluster naming for script 08 and the app (one rule, so figures and pages always agree).

Name = dominant mineral of the cluster + the most ENRICHED other mineral (cluster mean vs dataset
mean, in standard deviations). The 'high X' part is only added when that enrichment is >= 0.5 SD,
otherwise it would be noise. These are short descriptions of statistical mineral assemblages,
not validated rock types.
"""
MIN_ENRICH_SD = 0.5

def short(c):
    return c.replace("SWIR_", "").replace("TIR_", "")

def cluster_names(d):
    mins = [c for c in d if c.startswith(("SWIR_", "TIR_")) and "Invalid" not in c and d[c].mean() > 0.005]
    prof = d.groupby("Cluster")[mins].mean()
    enrich = (prof - d[mins].mean()) / d[mins].std()
    names = {}
    for k in prof.index:
        dom = short(prof.loc[k].idxmax())
        e = enrich.loc[k].sort_values(ascending=False)
        e = e[[short(c) != dom for c in e.index]]
        names[int(k)] = f"{dom}-rich" + (f", high {short(e.index[0])}" if e.iloc[0] >= MIN_ENRICH_SD else "")
    return names
