#!/usr/bin/env python3
"""Comparison proposal: reuse panels A–D and show E as explicit dependency rows.

Run from revision-wt: python docs/build_supplementary_pipeline_dag_v2.py
The existing builder and SVG are never modified. Repeated input names reference
the same artifacts, avoiding cross-panel routing. Input lists are not files.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import build_supplementary_pipeline_dag as base


@dataclass(frozen=True)
class Stage:
    id: str
    inputs: tuple[str, ...]
    rules: tuple[str, ...]
    publication: bool = False


# Logical artifact bundles, not newly invented merged files. Each stage's id is
# also its output reference; inputs therefore retain a checkable dependency DAG.
STAGES = (
    Stage(
        "supplementary",
        ("features", "benchmark_index"),
        ("merge_supplementary_features",),
    ),
    Stage(
        "uncertainty",
        ("benchmark",),
        ("compute_entropy_metrics", "compute_entropy_groups"),
    ),
    Stage("clusters", ("benchmark", "supplementary"), ("feature_clustering",)),
    Stage(
        "entropy_figures",
        ("benchmark", "supplementary", "uncertainty", "clusters"),
        ("statistical_tests", "entropy_main_figures"),
        True,
    ),
    Stage(
        "tsne",
        ("benchmark", "supplementary", "uncertainty"),
        ("compute_embeddings", "tsne_figure"),
        True,
    ),
    Stage(
        "shap",
        ("benchmark", "supplementary", "clusters", "splits"),
        ("shap_fold", "shap_aggregate", "shap_figures"),
        True,
    ),
    Stage("performance", ("benchmark",), ("performance_figures", "upset_figure"), True),
)

DEFAULT_TEXT = {
    **base.DEFAULT_TEXT,
    "handoff.title": "E · Feature integration and downstream analysis",
    "handoff.note": "Supplementary features are used in downstream analysis and RF/SHAP, independently of benchmark classifier training.",
    "v2.reading": "Read each row left to right. Repeated input names refer to the same data; each input list contains separate artifacts.",
    "v2.inputs": "REQUIRED INPUTS · REFERENCES",
    "v2.computation": "COMPUTATION",
    "v2.results": "RESULTS",
    "v2.publication": "Publication outputs",
    "v2.benchmark": "Benchmark tables (A)",
    "v2.benchmark_index": "Benchmark transcript index (A; coverage check)",
    "v2.features": "REP (B), NBD (C), rG4 + ScanFold2 (D) features",
    "v2.splits": "Dataset splits (A)",
    "v2.supplementary": "Supplementary matrix",
    "v2.uncertainty": "Uncertainty results",
    "v2.clusters": "Feature clusters",
    "v2.entropy_figures": "Entropy figures + statistical tables",
    "v2.tsne": "t-SNE figure",
    "v2.shap": "SHAP figures",
    "v2.performance": "Performance + UpSet figures",
    "v2.supplementary.step": "Merge supplementary features",
    "v2.supplementary.detail": "Clean features and check coverage",
    "v2.supplementary.format": ".tsv",
    "v2.uncertainty.step": "Compute uncertainty\nand define groups",
    "v2.uncertainty.detail": "Metrics and group assignments",
    "v2.uncertainty.format": ".tsv · metrics + groups",
    "v2.clusters.step": "Cluster features",
    "v2.clusters.detail": "Correlations and feature selection",
    "v2.clusters.format": ".csv + .txt",
    "v2.entropy_figures.step": "Compare groups\nand plot results",
    "v2.entropy_figures.detail": "Statistical tests and entropy plots",
    "v2.entropy_figures.format": ".pdf + .csv + .tsv",
    "v2.tsne.step": "Compute embeddings\nand plot t-SNE",
    "v2.tsne.detail": "Uncertainty annotates the plot",
    "v2.tsne.format": ".pdf",
    "v2.shap.step": "Train RF and compute SHAP",
    "v2.shap.detail": "Aggregate explanations and plot",
    "v2.shap.format": ".pdf",
    "v2.performance.step": "Plot performance and consensus",
    "v2.performance.detail": "Benchmark predictions",
    "v2.performance.format": ".pdf",
    "svg.desc": "Independent benchmark and feature-generation workflows followed by explicit input-to-analysis-to-output rows. Uncertainty uses benchmark tables; statistical testing requires uncertainty and clusters; embeddings and RF/SHAP are separate computations.",
}

# Grid columns: references, computation, results. No geometry in STAGES.
WIDTHS = (500, 360, 400)
GAP, PADDING, ROW_HEIGHT, ROW_GAP = 65, 30, 92, 28
FIRST_ROW = 155
PANEL_WIDTH = 2 * PADDING + sum(WIDTHS) + 2 * GAP
PANEL_HEIGHT = FIRST_ROW + len(STAGES) * (ROW_HEIGHT + ROW_GAP) - ROW_GAP + PADDING
PANEL_TOP = base.PANEL_Y["E"]
LEGEND_Y = PANEL_TOP + PANEL_HEIGHT + 38
HEIGHT = LEGEND_Y + 40


def boxes():
    result = {}
    for row, stage in enumerate(STAGES):
        for col, role in enumerate(("inputs", "step", "output")):
            node = base.NodeSpec(
                f"{stage.id}-{role}",
                "E",
                f"v2.{stage.id}.step" if role == "step" else f"v2.{stage.id}",
                f"v2.{stage.id}.detail" if role == "step" else f"v2.{stage.id}.format",
                "step" if role != "output" else "produced_file",
                "E",
                row,
                col,
            )
            result[node.id] = base.Box(
                PADDING + sum(WIDTHS[:col]) + col * GAP,
                FIRST_ROW + row * (ROW_HEIGHT + ROW_GAP),
                WIDTHS[col],
                ROW_HEIGHT,
                node,
            )
    return result


def render_analysis(text):
    grid = boxes()
    content = [
        f'<rect width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>',
        base.txt(PADDING, 37, text["handoff.title"], "panel-title"),
        base.txt(PADDING, 62, text["handoff.note"], "panel-note"),
        base.txt(PADDING, 85, text["v2.reading"], "panel-note"),
    ]
    for col, key in enumerate(("inputs", "computation", "results")):
        content.append(
            base.txt(
                PADDING + sum(WIDTHS[:col]) + col * GAP, 125, text[f"v2.{key}"], "small"
            )
        )
    # A background enclosure groups publication results; it is not a produced
    # file or a fictitious final processing operation.
    outputs = [grid[f"{s.id}-output"] for s in STAGES if s.publication]
    first, last = outputs[0], outputs[-1]
    content.append(
        f'<rect x="{first.x - 14}" y="{first.y - 22}" width="{first.width + 28}" '
        f'height="{last.y + last.height - first.y + 36}" rx="12" fill="#e8f4f2"/>'
    )
    content.append(base.txt(first.x + 8, first.y - 6, text["v2.publication"], "small"))
    for stage in STAGES:
        refs, step, output = (
            grid[f"{stage.id}-{role}"] for role in ("inputs", "step", "output")
        )
        for source, target in ((refs, step), (step, output)):
            start, end = source.anchor("right"), target.anchor("left")
            base.assert_endpoint_on_box(start, source)
            content.append(base.edge_path((start, end), "solid", destination=target))
        # Square-corner input-list enclosure deliberately differs from both
        # document shapes and rounded computation nodes.
        content.append(
            f'<rect x="{refs.x}" y="{refs.y}" width="{refs.width}" height="{refs.height}" '
            'fill="#ffffff" stroke="#cbd5e1"/>'
        )
        top = refs.y + (refs.height - len(stage.inputs) * 19) // 2 + 14
        for index, ref in enumerate(stage.inputs):
            content.append(
                base.txt(refs.x + 16, top + index * 19, text[f"v2.{ref}"], "small")
            )
        content.extend((base.render_node(step, text), base.render_node(output, text)))
    return (
        f'<g transform="translate({(base.W - PANEL_WIDTH) // 2} {PANEL_TOP})">'
        + "".join(content)
        + "</g>"
    )


def svg(overrides=None):
    text = {**DEFAULT_TEXT, **(overrides or {})}
    # Reuse the established document header, styles, marker and panels A–D.
    document = base.svg(
        {key: value for key, value in text.items() if key in base.DEFAULT_TEXT}
    )
    old_panel = base.render_panel(base.PANELS[-1], text)
    document = document.replace(old_panel, render_analysis(text))
    document = document.replace(f'height="{base.H}"', f'height="{HEIGHT}"')
    document = document.replace(
        f'viewBox="0 0 {base.W} {base.H}"', f'viewBox="0 0 {base.W} {HEIGHT}"'
    )
    document = document.replace(
        base.render_legend(text, base.legend_baseline()),
        base.render_legend(text, LEGEND_Y),
    )
    return document


def comparison_svg(before, after):
    """Standalone panel-E comparison at the same scale, preserving both SVGs."""
    from xml.etree import ElementTree as ET

    frames = []
    frame_width = 950
    crop_top = PANEL_TOP - 20
    crop_height = HEIGHT - crop_top
    for index, (label, document) in enumerate(
        (("V1 · Current", before), ("V2 · Explicit dependencies", after))
    ):
        root = ET.fromstring(document)
        # Both embedded documents use an arrow marker. Namespace their ids so
        # each image remains independent in SVG viewers.
        for item in root.iter():
            for key, value in list(item.attrib.items()):
                if key == "id":
                    item.set(key, f"v{index}-{value}")
                elif "url(#arrow)" in value:
                    item.set(key, value.replace("url(#arrow)", f"url(#v{index}-arrow)"))
            if item.tag.endswith("}style") and item.text:
                item.text = item.text.replace("url(#arrow)", f"url(#v{index}-arrow)")
        root.set("x", str(index * frame_width))
        root.set("y", "50")
        root.set("width", str(frame_width))
        root.set("height", str(crop_height // 2))
        root.set("viewBox", f"0 {crop_top} {base.W} {crop_height}")
        frames.append(
            f'<text x="{index * frame_width + 25}" y="30" font-family="Arial" font-size="20">{label}</text>'
        )
        frames.append(ET.tostring(root, encoding="unicode"))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1900" height="{crop_height // 2 + 60}">'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(frames) + "</svg>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("supplementary_pipeline_dag_v2.svg"),
    )
    parser.add_argument(
        "--comparison", type=Path, help="Also write a panel-E comparison SVG"
    )
    args = parser.parse_args()
    args.output.write_text(svg(), encoding="utf-8")
    print(f"wrote {args.output}")
    if args.comparison:
        before = (
            Path(__file__)
            .with_name("supplementary_pipeline_dag.svg")
            .read_text(encoding="utf-8")
        )
        args.comparison.write_text(comparison_svg(before, svg()), encoding="utf-8")
        print(f"wrote {args.comparison}")


if __name__ == "__main__":
    main()
