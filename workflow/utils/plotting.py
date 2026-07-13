from matplotlib import pyplot as plt

plt.rcParams["figure.constrained_layout.use"] = True

# Default line width
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["lines.markersize"] = 5
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["axes.linewidth"] = 0.5

# Font
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"

## Axes text
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["axes.labelsize"] = 7

# Configure tick parameters
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7

## Legends and annotations text
plt.rcParams["font.size"] = 6
plt.rcParams["legend.fontsize"] = 6

## Keep text as text in SVG output
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

COLORS = {
    "lnc": "#9467bd",
    "pc": "#d95f02",
    "rna": "#1b7837",  # spliced RNA TE features
    "dna": "#762a83",  # unspliced DNA TE features
    "entropy_class": {"low": "#2ecc71", "other": "#95a5a6", "high": "#e74c3c"},
    "entropy_class_separated": {
        "low_coding": "#1a9850",
        "high_coding": "#d73027",
        "low_lncRNA": "#91cf60",
        "high_lncRNA": "#fc8d59",
        "middle": "#95a5a6",
    },
}

# ── RNA/DNA feature provenance helpers ─────────────────────────────────────────
# Every feature is computed on either the spliced RNA transcript or the genomic
# DNA. TE features exist as both variants and carry an rna_/dna_ prefix; non-B DNA
# features are genomic (listed in entropy_figures.DNA_ONLY); everything else
# defaults to RNA. We tag each label with a group colour and a greyscale-safe
# superscript marker (R/D). The marker is mathtext, not a Unicode superscript
# glyph, because Arial (the publication font) lacks U+1D3F/U+1D30 — matplotlib
# would silently drop them. Mathtext synthesises the superscript in any font.
_PREFIXES = {"rna_": "rna", "dna_": "dna"}
_MARK = {"rna": r"$^{\mathrm{R}}$", "dna": r"$^{\mathrm{D}}$"}


def feature_group(raw):
    """Provenance group of a feature id: "rna" or "dna".

    Resolution order: explicit rna_/dna_ prefix → non-B DNA membership → RNA
    default (features are computed on the spliced transcript unless stated).
    """
    for pre, group in _PREFIXES.items():
        if raw.startswith(pre):
            return group
    from utils.entropy_figures import DNA_ONLY

    return "dna" if raw in DNA_ONLY else "rna"


def feature_label(raw, *, marker=True):
    """Human label for a feature id, provenance-aware.

    Returns (label, group). The label comes from an exact FEATURE_LABEL_DICT
    match (so prefixed keys keep their own names), falling back to the
    prefix-stripped base name. A superscript rna/dna marker is appended unless
    ``marker=False``.
    """
    # Imported lazily so plot_shap_figures' in-place FEATURE_LABEL_DICT.update()
    # is reflected regardless of import order.
    from utils.entropy_figures import FEATURE_LABEL_DICT

    group = feature_group(raw)
    label = FEATURE_LABEL_DICT.get(raw)
    if label is None:  # unknown key — try stripping a known prefix
        for pre in _PREFIXES:
            if raw.startswith(pre):
                label = FEATURE_LABEL_DICT.get(raw[len(pre) :], raw[len(pre) :])
                break
        else:
            label = raw
    if group and marker:
        label = f"{label} {_MARK[group]}"
    return label, group


def color_feature_ticklabels(ax, raw_names, axis="y"):
    """Colour tick labels by rna/dna provenance. Call AFTER labels are set.

    raw_names must be in the same order as the drawn tick labels.
    """
    ticks = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lbl, raw in zip(ticks, raw_names):
        group = feature_group(raw)
        if group:
            lbl.set_color(COLORS[group])


if __name__ == "__main__":
    # ponytail: self-check for the provenance resolution (the non-trivial part)
    lbl, grp = feature_label("rna_te_count")
    assert grp == "rna" and lbl == f"TE count {_MARK['rna']}", (lbl, grp)
    lbl, grp = feature_label("dna_te_count", marker=False)
    assert grp == "dna" and lbl == "TE count", (lbl, grp)
    lbl, grp = feature_label("total_nonb_count")  # unprefixed non-B → dna
    assert grp == "dna" and lbl.endswith(_MARK["dna"]), (lbl, grp)
    lbl, grp = feature_label("ORF_l_cpat")  # sequence feature → rna default
    assert grp == "rna" and lbl.endswith(_MARK["rna"]), (lbl, grp)
    lbl, grp = feature_label("dna_te_foobar")  # unknown prefixed → strip fallback
    assert grp == "dna" and lbl == f"te_foobar {_MARK['dna']}", (lbl, grp)
    print("plotting self-check OK")
