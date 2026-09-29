#!/usr/bin/env python3
"""Compact file-reference variant of v2. Run from revision-wt:

    python docs/build_supplementary_pipeline_dag_v3.py

Reuses v2's scientific stages and panels A–D. Repeated document boxes refer
to the same artifacts; Features denotes the collection of B–D output files.
"""

import argparse
from pathlib import Path

import build_supplementary_pipeline_dag_v2 as v2

base = v2.base
STAGES = v2.STAGES
DEFAULT_TEXT = {
    **v2.DEFAULT_TEXT,
    "handoff.note": "Repeated file boxes reference the same data. Features = B–D outputs; Tables and Splits = A outputs.",
    "v3.note": "Supp. = supplementary matrix; Groups = uncertainty metrics and groups. Tables supply the index for merging.",
    "v2.inputs": "INPUT FILES",
    "v2.benchmark": "Tables",
    "v2.benchmark_index": "Tables",
    "v2.features": "Features",
    "v2.splits": "Splits",
    "v2.supplementary": "Supp.",
    "v2.uncertainty": "Groups",
    "v2.clusters": "Clusters",
    "v2.entropy_figures": "Entropy figures\n+ statistical tables",
    "v2.performance": "Performance\n+ UpSet figures",
    "v2.supplementary.step": "Merge supplementary features",
    "v2.uncertainty.step": "Compute uncertainty + groups",
    "v2.entropy_figures.step": "Compare groups + plot",
    "v2.tsne.step": "Embed features + plot",
    "v2.shap.step": "Train RF + SHAP + aggregate",
    "v2.performance.step": "Plot performance + consensus",
    "svg.desc": "Independent benchmark and feature workflows, followed by seven analysis rows with compact document-shaped input references and dependency arrows. Repeated references denote the same files.",
}

PADDING, GAP = 30, 44
FILE_WIDTH, FILE_HEIGHT = 106, 40
FILE_COL_GAP, FILE_ROW_GAP = 20, 14
INPUT_WIDTH = 2 * FILE_WIDTH + FILE_COL_GAP
STEP_WIDTH, OUTPUT_WIDTH = 320, 300
ROW_HEIGHT = 2 * FILE_HEIGHT + FILE_ROW_GAP
ROW_GAP, FIRST_ROW = 26, 145
STEP_X = PADDING + INPUT_WIDTH + GAP
OUTPUT_X = STEP_X + STEP_WIDTH + GAP
PANEL_WIDTH = OUTPUT_X + OUTPUT_WIDTH + PADDING
PANEL_HEIGHT = FIRST_ROW + len(STAGES) * (ROW_HEIGHT + ROW_GAP) - ROW_GAP + PADDING
LEGEND_Y = v2.PANEL_TOP + PANEL_HEIGHT + 38
HEIGHT = LEGEND_Y + 40


def boxes():
    result = {}
    for row, stage in enumerate(STAGES):
        top = FIRST_ROW + row * (ROW_HEIGHT + ROW_GAP)
        for index, ref in enumerate(stage.inputs):
            # One or two references share the right column; larger sets use
            # a compact two-by-two grid with a clear routing gap in the middle.
            col = 1 if len(stage.inputs) <= 2 else index % 2
            subrow = index if len(stage.inputs) <= 2 else index // 2
            y = (
                top + (ROW_HEIGHT - FILE_HEIGHT) // 2
                if len(stage.inputs) == 1
                else top + subrow * (FILE_HEIGHT + FILE_ROW_GAP)
            )
            node = base.NodeSpec(
                f"{stage.id}-input-{ref}",
                "E",
                f"v2.{ref}",
                None,
                "produced_file",
                "A" if ref in {"benchmark", "benchmark_index", "splits"} else "E",
                row,
                col,
            )
            result[node.id] = base.Box(
                PADDING + col * (FILE_WIDTH + FILE_COL_GAP),
                y,
                FILE_WIDTH,
                FILE_HEIGHT,
                node,
            )
        for role, x, width in (
            ("step", STEP_X, STEP_WIDTH),
            ("output", OUTPUT_X, OUTPUT_WIDTH),
        ):
            node = base.NodeSpec(
                f"{stage.id}-{role}",
                "E",
                f"v2.{stage.id}.step" if role == "step" else f"v2.{stage.id}",
                None if role == "step" else f"v2.{stage.id}.format",
                "step" if role == "step" else "produced_file",
                "E",
                row,
                2 if role == "step" else 3,
            )
            result[node.id] = base.Box(x, top + (ROW_HEIGHT - 76) // 2, width, 76, node)
    return result


def routes(stage, grid):
    """Join input branches in the gap between their rows, then enter the step."""
    step, output = grid[f"{stage.id}-step"], grid[f"{stage.id}-output"]
    inputs = [grid[f"{stage.id}-input-{ref}"] for ref in stage.inputs]
    target = step.anchor("left")
    paths = []
    junctions = []
    for source in inputs:
        start = source.anchor("right")
        base.assert_endpoint_on_box(start, source)
        branch_x = start[0] + (FILE_COL_GAP // 2 if source.node.col == 0 else GAP // 2)
        junctions.append(branch_x)
        points = (start, (branch_x, start[1]))
        if start[1] != target[1]:
            points += ((branch_x, target[1]),)
        paths.append((points, False, None))
    paths.append((((min(junctions), target[1]), target), True, step))
    paths.append(((step.anchor("right"), output.anchor("left")), True, output))
    return paths


def render_analysis(text):
    grid = boxes()
    content = [
        f'<rect width="{PANEL_WIDTH}" height="{PANEL_HEIGHT}" rx="18" fill="#f8fafc" stroke="#cbd5e1"/>',
        base.txt(PADDING, 37, text["handoff.title"], "panel-title"),
        base.txt(PADDING, 62, text["handoff.note"], "panel-note"),
        base.txt(PADDING, 84, text["v3.note"], "panel-note"),
    ]
    for x, key in ((PADDING, "inputs"), (STEP_X, "computation"), (OUTPUT_X, "results")):
        content.append(base.txt(x, 120, text[f"v2.{key}"], "small"))
    outputs = [grid[f"{s.id}-output"] for s in STAGES if s.publication]
    first, last = outputs[0], outputs[-1]
    content.extend(
        (
            f'<rect x="{first.x - 12}" y="{first.y - 23}" width="{first.width + 24}" height="{last.y + last.height - first.y + 35}" rx="12" fill="#e8f4f2"/>',
            base.txt(first.x + 8, first.y - 7, text["v2.publication"], "small"),
        )
    )
    for stage in STAGES:
        for points, arrow, destination in routes(stage, grid):
            content.append(base.edge_path(points, "solid", arrow, destination))
    content.extend(base.render_node(box, text) for box in grid.values())
    return (
        f'<g transform="translate({(base.W - PANEL_WIDTH) // 2} {v2.PANEL_TOP})">'
        + "".join(content)
        + "</g>"
    )


def svg(overrides=None):
    text = {**DEFAULT_TEXT, **(overrides or {})}
    document = base.svg(
        {key: value for key, value in text.items() if key in base.DEFAULT_TEXT}
    )
    document = document.replace(
        base.render_panel(base.PANELS[-1], text), render_analysis(text)
    )
    document = document.replace(f'height="{base.H}"', f'height="{HEIGHT}"')
    document = document.replace(
        f'viewBox="0 0 {base.W} {base.H}"', f'viewBox="0 0 {base.W} {HEIGHT}"'
    )
    return document.replace(
        base.render_legend(text, base.legend_baseline()),
        base.render_legend(text, LEGEND_Y),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("supplementary_pipeline_dag_v3.svg"),
    )
    args = parser.parse_args()
    args.output.write_text(svg(), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
