#!/usr/bin/env python3
"""
Generate timeline visualization combining ML tools and GENCODE releases.
Outputs both static (PDF/PNG) and interactive (HTML) formats.
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Try to import plotly, but don't fail if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print(
        "Warning: plotly not available, skipping interactive HTML output",
        file=sys.stderr,
    )


def parse_date(date_str, default_day=15):
    """Parse date strings in MM.YYYY or YYYY-MM format."""
    if pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Handle MM.YYYY format
    if "." in date_str:
        parts = date_str.split(".")
        if len(parts) == 2:
            month, year = int(parts[0]), int(parts[1])
            return datetime(year, month, default_day)

    # Handle YYYY-MM format
    if "-" in date_str:
        parts = date_str.split("-")
        if len(parts) == 2:
            year, month = int(parts[0]), int(parts[1])
            return datetime(year, month, default_day)

    # Handle just year
    try:
        year = int(date_str)
        return datetime(year, 6, default_day)  # Middle of year
    except:
        pass

    return None


def load_data(tools_file, gencode_file):
    """Load and prepare timeline data."""
    # Load tools
    tools_df = pd.read_csv(tools_file)
    tools_df["date"] = tools_df["year"].apply(lambda y: datetime(int(y), 6, 15))
    tools_df["type"] = "Tool"
    tools_df["label"] = tools_df["tool"]

    # Load GENCODE versions
    gencode_df = pd.read_csv(gencode_file)
    gencode_df["date"] = gencode_df["public_date"].apply(parse_date)
    gencode_df["type"] = "Database"
    gencode_df["label"] = "GENCODE v" + gencode_df["version"].astype(str)

    # Combine
    combined = pd.concat(
        [
            tools_df[["date", "type", "label", "year"]].rename(
                columns={"year": "year_only"}
            ),
            gencode_df[["date", "type", "label", "version"]].assign(
                year_only=lambda x: x["date"].dt.year
            ),
        ],
        ignore_index=True,
    )

    combined = combined.dropna(subset=["date"]).sort_values("date")

    return combined, tools_df, gencode_df


def create_matplotlib_timeline(combined_df, output_file):
    """Create publication-quality timeline using matplotlib."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define colors
    colors = {"Tool": "#2E86AB", "Database": "#A23B72"}  # Blue  # Purple

    # Plot timeline
    y_positions = {"Tool": 1, "Database": 0}

    for idx, row in combined_df.iterrows():
        y = y_positions[row["type"]]
        date = row["date"]

        # Plot point
        ax.scatter(date, y, c=colors[row["type"]], s=100, zorder=3, alpha=0.7)

        # Add label with rotation and offset to avoid overlap
        label = row["label"]
        rotation = 45 if row["type"] == "Tool" else -45
        va = "bottom" if row["type"] == "Tool" else "top"
        y_offset = 0.05 if row["type"] == "Tool" else -0.05

        ax.text(
            date,
            y + y_offset,
            label,
            rotation=rotation,
            ha="left" if row["type"] == "Tool" else "right",
            va=va,
            fontsize=8,
            alpha=0.8,
        )

    # Draw horizontal lines for each track
    ax.axhline(y=1, color=colors["Tool"], linewidth=2, alpha=0.3, zorder=1)
    ax.axhline(y=0, color=colors["Database"], linewidth=2, alpha=0.3, zorder=1)

    # Formatting
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GENCODE Releases", "ML Tools"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Year", fontsize=14, fontweight="bold")
    ax.set_title(
        "Timeline of lncRNA Classification Tools and GENCODE Releases",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["Tool"], label="ML Tools"),
        mpatches.Patch(color=colors["Database"], label="GENCODE Releases"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved matplotlib timeline to {output_file}")
    plt.close()


def create_plotly_timeline(combined_df, tools_df, gencode_df, output_file):
    """Create interactive timeline using plotly."""
    if not PLOTLY_AVAILABLE:
        return

    fig = go.Figure()

    # Add tools
    fig.add_trace(
        go.Scatter(
            x=tools_df["date"],
            y=[1] * len(tools_df),
            mode="markers+text",
            name="ML Tools",
            text=tools_df["tool"],
            textposition="top center",
            marker=dict(size=10, color="#2E86AB"),
            hovertemplate="<b>%{text}</b><br>Year: %{x|%Y}<extra></extra>",
        )
    )

    # Add GENCODE releases
    fig.add_trace(
        go.Scatter(
            x=gencode_df["date"],
            y=[0] * len(gencode_df),
            mode="markers+text",
            name="GENCODE Releases",
            text=gencode_df["label"],
            textposition="bottom center",
            marker=dict(size=8, color="#A23B72"),
            hovertemplate="<b>%{text}</b><br>Date: %{x|%B %Y}<extra></extra>",
        )
    )

    # Layout
    fig.update_layout(
        title="Timeline of lncRNA Classification Tools and GENCODE Releases",
        xaxis_title="Year",
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["GENCODE Releases", "ML Tools"],
            range=[-0.5, 1.5],
        ),
        height=800,
        hovermode="closest",
        showlegend=True,
        template="plotly_white",
    )

    fig.write_html(output_file)
    print(f"Saved interactive timeline to {output_file}")


def main():
    if len(sys.argv) < 4:
        print("Usage: generate_timeline.py <tools_csv> <gencode_csv> <output_prefix>")
        sys.exit(1)

    tools_file = sys.argv[1]
    gencode_file = sys.argv[2]
    output_prefix = sys.argv[3]

    # Create output directory if needed
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {tools_file} and {gencode_file}")
    combined_df, tools_df, gencode_df = load_data(tools_file, gencode_file)

    print(f"Loaded {len(tools_df)} tools and {len(gencode_df)} GENCODE releases")

    # Generate matplotlib version (PDF and PNG)
    create_matplotlib_timeline(combined_df, f"{output_prefix}.pdf")
    create_matplotlib_timeline(combined_df, f"{output_prefix}.png")

    # Generate plotly version (HTML)
    if PLOTLY_AVAILABLE:
        create_plotly_timeline(
            combined_df, tools_df, gencode_df, f"{output_prefix}.html"
        )

    print("Timeline generation complete!")


if __name__ == "__main__":
    main()
