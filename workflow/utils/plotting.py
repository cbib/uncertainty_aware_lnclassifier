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
    "entropy_class": {"low": "#2ecc71", "other": "#95a5a6", "high": "#e74c3c"},
    "entropy_class_separated": {
        "low_coding": "#1a9850",
        "high_coding": "#d73027",
        "low_lncRNA": "#91cf60",
        "high_lncRNA": "#fc8d59",
        "middle": "#95a5a6",
    },
}
