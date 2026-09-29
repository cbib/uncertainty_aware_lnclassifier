#!/usr/bin/env python3
"""Build the supplementary LNClassifier workflow overview SVG.

Requires Python 3.9+ and the standard library only. The script has no external
dependencies and does not read workflow outputs. The layout is deliberately
stage-level: the rule-level DOT/SnakeVision files remain the detailed technical
record, while this figure communicates the independent pipelines and their
handoff into revision-wt.

=============================================================================
TABLE OF CONTENTS
=============================================================================
  1. Quick start
  2. What the figure looks like (panel map)
  3. How the script is organized (top to bottom)
  4. Recipes: how to change things
       4.1 Change wording only (no code edit)
       4.2 Change colors
       4.3 Change fonts / line styles
       4.4 Resize the canvas or move panels
       4.5 Add / remove / move a node
       4.6 Add / remove an edge
       4.7 Add a new panel
       4.8 Adjust the legend
  5. Coordinate system and geometry rules
  6. Edge groups (the "bus" connectors) explained
  7. Hard-coded values you will trip over (read before editing panels A / E)
  8. Validation checklist
  9. Concurrent editing (lock convention)

=============================================================================
1. QUICK START (from revision-wt)
=============================================================================
    python docs/build_supplementary_pipeline_dag.py
    python docs/build_supplementary_pipeline_dag.py --dump-text /tmp/dag-text.json
    # Edit the JSON values, keeping the keys unchanged, then render a variant:
    python docs/build_supplementary_pipeline_dag.py --text /tmp/dag-text.json -o /tmp/dag.svg

Command-line options (``--help`` prints only the summary line and a pointer to
this docstring; the full guide is here and in
supplementary_pipeline_dag_generation.md):
    -o / --output PATH   Where to write the SVG. Default: supplementary_pipeline_dag.svg
                         next to this script.
    --text JSON          Partial or complete text overrides (see 4.1).
    --dump-text JSON     Write the complete text catalog to JSON and exit
                         WITHOUT rendering. Use it as a starting template.

=============================================================================
2. WHAT THE FIGURE LOOKS LIKE (panel map)
=============================================================================
Canvas is W x H = 1800 x 1800 SVG units.

    +--------------------------------------------------------------+
    | title / subtitle / versions line (y = 58, 86, 108)           |
    | +----------------------------------------------------------+ |
    | | A  Model benchmark  (one horizontal row of 6 nodes plus   | |
    | |    one extra node below the last one)          y = 120    | |
    | +----------------------------------------------------------+ |
    |                                                              |
    | +--------+   +--------+   +----------------+     y = 400     |
    | | B  REP |   | C  NBD |   | D  RNA / rG4   |                 |
    | | (vert) |   | (vert) |   | (branching)    |                 |
    | +--------+   +--------+   +----------------+                 |
    |                                                              |
    | +----------------------------------------------------------+ |
    | | E  Feature integration -> analysis -> publication files   | |
    | |    (row of 6 input files, row of 5 analysis steps, and    | |
    | |    one wide output "stack of sheets")         y = 1000    | |
    | +----------------------------------------------------------+ |
    |                       legend row                             |
    +--------------------------------------------------------------+

Panels A-D are independent pipelines. Panel E merges their outputs. Panel E's
top row DUPLICATES the "produced_file" nodes of A-D (ids ``e-*``), which is how
the cross-panel handoff is drawn without any cross-panel edges.

=============================================================================
3. HOW THE SCRIPT IS ORGANIZED (top to bottom)
=============================================================================
  W, H, PANEL_PADDING, COLORS
                      Canvas size, shared panel margin, and the color palette.
  DEFAULT_TEXT        EVERY string shown in the figure (labels, legend, title,
                      accessibility text). No geometry.
  STYLES              Shared CSS injected into the SVG <style> block.
  PanelSpec           Dataclass: one rounded panel background.
  NodeSpec            Dataclass: one box (step or file) inside a panel. The
                      panel is derived from the id, the two text keys from one
                      text prefix.
  EdgeSpec            Dataclass: one arrow between two nodes.
  chain(), fan_in(), fan_out()
                      Small helpers that build runs of EdgeSpecs.
  Box                 Computed rectangle for a node (x, y, width, height).
  GridLayout          Per-panel grid parameters.
  PANELS/NODES/EDGES  The declarative description of the workflow (the data).
  NODE_WIDTHS, GRID, PANEL_Y, A_TOPS, E_* constants, BUS_DROP, FANOUT_GAP,
  STACK_OFFSETS       Layout constants that turn (row, col) into coordinates.
  txt()               Helper that emits one escaped <text> element.
  layout()            (row, col) -> Box for every node in a panel.
  panel_size()        Panel width/height derived from its boxes.
  PANEL_X / panel_origin()  Where each panel sits on the canvas.
  edge_path(), orthogonal_route()                   Single-edge drawing.
  _fan_out_bus(), _merge_bus(), _bus_*(), GROUP_DRAWERS, render_edges()
                                                    Bus (group) drawing.
  workflow_colors(), render_node()                  Node drawing.
  render_panel(), render_legend()                   Panel and legend drawing.
  resolve_text(), svg(), main()                     Assembly and CLI.

Data flow:  NODES/EDGES + DEFAULT_TEXT  --layout()-->  Boxes  --render_*-->  SVG string

Rule of thumb: change WORDS in DEFAULT_TEXT, change STRUCTURE in NODES/EDGES,
change POSITIONS in GRID / PANEL_Y / A_TOPS / the E_* constants.

=============================================================================
4. RECIPES
=============================================================================
4.1 Change wording only (no code edit)
    Run --dump-text, edit the values in the JSON, keep every key, then run
    with --text. You may supply only the keys you want to change; missing keys
    keep their defaults (DEFAULT_TEXT itself is never modified). Unknown keys
    or non-string values raise an error. Text is plain Unicode, NOT SVG markup
    ("&" and "<" are escaped for you).
    Use "\\n" inside a node title/detail for an explicit line break (this
    applies to node titles and details only). There is NO automatic fitting or
    layout: after lengthening a label, inspect the SVG and widen the box (see
    4.5), adjust spacing or font sizes, or shorten the text.
    Key naming: "<prefix>.<name>.title" is the bold first line(s),
    "<prefix>.<name>.detail" is the lighter second line(s). Prefixes:
    revision.* = panel A (and panel E steps), te.* = B, nbd.* = C,
    rna.* = D, handoff.* = panel E inputs / legend, svg.* = document-level.
    Every key in DEFAULT_TEXT is used by some node, the legend or the document
    header (the old unused "spare" keys were removed).

4.2 Change colors
    COLORS maps a workflow kind to a (fill, border) pair of hex colors.
    workflow_colors() decides which pair a node uses:
        workflow "A" -> COLORS["rev"] (purple)
        "B" -> COLORS["te"]   "C" -> COLORS["nbd"]   "D" -> COLORS["rna"]
        "E" and kind == "step" -> COLORS["analysis"]
        anything else in E    -> COLORS["output"]
    Note the workflow of a NodeSpec (not its panel) picks the color: the
    panel-E copies of B/C/D/A files keep their source workflow's color (the
    workflow defaults to the node's own panel; it is set explicitly only on
    those copies). Steps use the fill; files are always white with a colored
    border. Panel backgrounds are the ``fill`` field on each PanelSpec.
    The legend samples use COLORS["te"] (LEGEND_FILL / LEGEND_STROKE).

4.3 Change fonts / line styles
    Edit the CSS in STYLES. Classes: .title, .subtitle, .panel-title, .node,
    .label (node title), .small (step detail), .file-format (monospace
    file-type line), .edge (solid arrow), .edge-dash (dashed arrow), .legend.
    Text HEIGHT used for vertical centering inside nodes is NOT read from CSS:
    it is LABEL_LINE (20) and DETAIL_LINE (18) next to render_node(). If you
    change font sizes noticeably, update those constants too.

4.4 Resize the canvas or move panels
    * Canvas: W, H (top of file). The white background and viewBox follow.
    * Vertical position of a panel: PANEL_Y[panel_id].
    * Horizontal position of B, C, D: computed by _middle_row_positions(),
      which centers the three panels as a group with ``gutter`` = 40 units
      between them. Change the gutter argument's default to space them out.
    * Panel A is centered horizontally (W - width) // 2 by panel_origin().
    * Panel E's x is E_X (97), used in panel_origin().
    * Panel heights: panel_size(). A returns a fixed 265, E a fixed
      (1695, 721 + E_DY); B, C, D fit their content plus PANEL_PADDING.
    * The legend baseline is derived from panel E's bottom edge + 38.

4.5 Add / remove / move a node
    Add a NodeSpec to NODES:
        NodeSpec(id, text, kind, row, col, workflow="", width=None)
      id        Unique string, referenced by EdgeSpec. Convention:
                "<panel letter>-<name>". The FIRST LETTER of the id (upper-
                cased) IS the panel the node is drawn in.
      text      Prefix into DEFAULT_TEXT: "<text>.title" is the bold title and
                "<text>.detail" the detail line. Add both keys there!
      kind      "step"          rounded rectangle, colored fill (a process).
                "file"          white page with an open dog-ear (an INPUT file).
                "produced_file" white page with a FILLED dog-ear (a file the
                                pipeline PRODUCES).
      row, col  Grid slot. Pixel position = f(GRID[panel], row, col).
      workflow  Color family (see 4.2); empty means "same as the panel".
      width     Optional per-node width override in units; otherwise
                NODE_WIDTHS[kind] (or A_COLUMN_WIDTHS for panel A).
    Moving a node: change row/col. For panels A and E some positions are
    overridden by A_TOPS and _e_position() (see section 7).
    Removing: delete the NodeSpec AND every EdgeSpec that references its id
    (a dangling id raises KeyError in render_edges()).

4.6 Add / remove an edge
    Add an EdgeSpec to EDGES (or use a helper):
        EdgeSpec(src, dst, style="solid", group="", src_anchor="bottom", dst_anchor="top")
        chain(a, b, c, ...)          a->b, b->c, ... (keyword args are passed
                                     to every EdgeSpec, e.g. the anchors)
        fan_in((a, b, c), dst, group)   many sources -> one target (merge bus)
        fan_out(src, (a, b), group)     one source -> many targets (fan-out bus)
      src, dst      Node ids. IMPORTANT: an edge is rendered by the panel that
                    contains its SOURCE node, and both nodes must be in that
                    panel (boxes are looked up per panel). No cross-panel edges.
      style         "solid" or "dashed" (see edge_path()). Only ungrouped edges
                    honor it; grouped edges are always solid.
      group         "" for a plain point-to-point arrow. A group name makes
                    several edges share a bus (section 6).
      src_anchor /  Where the arrow leaves/enters a box: "top", "bottom",
      dst_anchor    "left", "right" (midpoint of that side).
    Plain edges are routed orthogonally by orthogonal_route(): straight if the
    anchors line up, otherwise an elbow through a mid-way channel.

4.7 Add a new panel
    1. Add a PanelSpec to PANELS (id letter, title key, background color).
    2. Add its title key to DEFAULT_TEXT.
    3. Add a GridLayout entry in GRID for the new id.
    4. Add its y coordinate to PANEL_Y.
    5. Give it an x: either add it to _middle_row_positions() or special-case
       it in panel_origin().
    6. Add NodeSpecs / EdgeSpecs with ids starting with the new panel letter.
    7. Extend the workflow_colors() palette if it needs its own color.
    Also raise H if the panel extends below the current canvas.

4.8 Adjust the legend
    LEGEND_ITEMS is a table of (shape, x offset, text key); render_legend()
    draws one sample plus its label per row, with the label placed at
    LEGEND_LABEL_DX[shape] to the right of the sample. Shapes: "step", "file",
    "produced", "arrow", "color". The whole legend is centered via
    LEGEND_WIDTH (680): keep LEGEND_WIDTH about equal to the right edge of the
    last item if you add or widen items.

=============================================================================
5. COORDINATE SYSTEM AND GEOMETRY RULES
=============================================================================
* SVG units; x increases to the right, y increases downward; (0, 0) is the
  top-left of the canvas.
* Each panel is drawn inside ``<g transform="translate(origin_x origin_y)">``
  so every coordinate inside layout() / render_edges() is PANEL-LOCAL
  (relative to the panel's top-left corner), not canvas-global.
* A Box's (x, y) is its top-left corner. Anchors are side midpoints. Integer
  division (//) keeps anchors on whole units; edge_path() asserts that an
  arrow's final point lies exactly on the destination box's outline (see
  section 8 for the two arrows that are exempt).
* Standard grid (panels B, C, D, E): node center x = first_center +
  col * column_pitch; node top y = first_row_top + row * (row_height +
  row_gutter). Panel D adds 2 * row_gutter for rows > 0 to leave room for the
  split-arrow bus.
* Panel A does not use column_pitch: each column has its own width taken from
  A_COLUMN_WIDTHS, separated by A_COLUMN_GAPS, starting at PANEL_PADDING.

=============================================================================
6. EDGE GROUPS (shared "bus" connectors)
=============================================================================
A group draws several arrows as one tidy tree: stubs from each endpoint to a
horizontal bus line, then arrowhead(s) into the shared endpoint(s). Groups are
registered in GROUP_DRAWERS ({name: function}); they are drawn in that dict's
order, before the ungrouped edges. Inventing a new group name means writing a
drawer function and adding it to GROUP_DRAWERS; an edge whose group is not
registered raises ValueError. Two building blocks do the shared work:
_fan_out_bus() (one source -> many targets) and _merge_bus() (many ports ->
one bus).

  ""             (default) independent orthogonal arrow.
  "rna-input"    ONE source fans out to several targets (panel D: the FASTA
                 splits into rG4detector and ScanFold2). Bus sits half way
                 between source bottom and the highest target top.
  "handoff"      MANY sources merge into ONE arrow (panel E: the five feature
                 files converge into the dashed analysis frame, whose top
                 center receives the arrowhead; it is NOT the "Cluster
                 features" box). Bus is BUS_DROP (100) units below the sources.
  "analysis-out" ONE source fans out to several targets (panel E: benchmark
                 table -> entropy and performance steps). Bus is FANOUT_GAP
                 (32) units above the first target.
  "final-in"     MANY sources merge into ONE target (panel E: the five analysis
                 steps converge into the final publication-files box). Every
                 source that lies inside the dashed frame shares ONE port at
                 the frame's bottom center; the bus is BUS_DROP (100) units
                 below the first source. The arrow is drawn from the TARGET's
                 top center straight up to the bus and stops at the top of the
                 back sheet of the stack (STACK_OFFSETS[0] above the box).

In "handoff" and "final-in" the merged arrow is drawn at one x (the frame's or
the target's center) even if that x sits far from the middle of the bus row; the
bus itself spans only the source ports.
Edges within a group must all share the same source (fan-out groups) or the
same destination (merge groups); the first edge in the group defines it.

=============================================================================
7. HARD-CODED VALUES YOU WILL TRIP OVER
=============================================================================
These are deliberate "illustrator" positions and are NOT derived from the grid:
  * A_TOPS: a-output is forced to top = 26 and a-model-features to top = 154
    (the two right-hand nodes stack vertically); panel A's other nodes use the
    grid top. Panel A's height is fixed at 265 in panel_size().
  * E_DY (-45) shifts panel E's content vertically; it is applied to the row
    tops, the panel height and E_FRAME. Negative = move up.
  * _e_position(), panel "E": row 0 centers are 132 + 251 * col at top
    120 + E_DY; row 2 (analysis steps) centers are E_ANALYSIS_CENTERS =
    (317, 602, 887, 1233, 1528) at top 392 + E_DY; the final output (any other
    row) is centered at x = 944, top = 624 + E_DY.
  * panel_size(): A -> height 265; E -> (1695, 721 + E_DY). Edit these if you
    add rows or move nodes outside those bounds.
  * E_FRAME = (167, 362 + E_DY, 870, 132): the dashed pink rounded rectangle
    around the analysis steps. render_panel() also draws a short solid line at
    the frame's mid-height from the entropy box's left edge to the frame's
    right edge. The "handoff" bus ends at the frame's top center and the
    "final-in" bus leaves from the frame's bottom center, so move E_FRAME and
    the E analysis nodes together.
  * render_node(), node id "e-output": drawn as a stack of offset sheets
    (STACK_OFFSETS = (12, 6)) to signal "a collection of files"; the final-in
    arrow ends STACK_OFFSETS[0] above the box.
  * E_X (97) pins panel E's x in panel_origin().
  * LEGEND_ITEMS: all x offsets are literal numbers, and LEGEND_WIDTH is a
    literal 680 (not derived from the items).
  * BUS_DROP (100) and FANOUT_GAP (32) assume the current E row spacing.
If you move panel-E nodes, expect to revisit all of the above.

=============================================================================
8. VALIDATION CHECKLIST
=============================================================================
After any change:
  1. Run the script; it will raise AssertionError if an arrow endpoint no
     longer lies on its destination box. This check covers every ungrouped
     arrow and the fan-out arrows ("rna-input", "analysis-out"). The two merge
     arrows ("handoff" ends on the frame, "final-in" ends on the stack's back
     sheet) are NOT checked, so look at them yourself.
  2. Open the SVG in a browser and check that text fits inside every box,
     nothing overlaps, and arrowheads touch box edges.
  3. Check the accessible <title> / <desc> still describe the figure
     (svg.accessible_title, svg.desc).
  4. Regenerate with no overrides to confirm defaults still render.
  5. When refactoring, keep a golden copy of the SVG and ``diff`` it against
     the new output: rendering is deterministic.
See supplementary_pipeline_dag_generation.md for the editing guide and the
geometry-check command.

=============================================================================
9. CONCURRENT EDITING
=============================================================================
From revision-wt, acquire the advisory lock with `mkdir docs/.dag_edit.lock`
before editing this script, its tests or generated SVG. If it exists, wait
and coordinate with its owner. Release only your own lock with `rmdir` after
regeneration and validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

# ---------------------------------------------------------------------------
# CANVAS AND PALETTE
# ---------------------------------------------------------------------------
# W, H: overall SVG size in units (also used for the viewBox and background).
W, H = 1800, 1800

# Empty margin (right and bottom) used when sizing panels, and the left offset
# of panel A's first column.
PANEL_PADDING = 25

# Each entry is (fill, border) as hex colors. See "4.2 Change colors" above
# for which nodes use which entry.
COLORS = {
    "rev": ("#f0eaff", "#8768c5"),  # purple -> workflow A, model benchmark
    "te": ("#e8f3ff", "#3787c5"),  # blue   -> workflow B, repetitive elements
    "nbd": ("#eaf8ef", "#45a36b"),  # green  -> workflow C, non-B DNA
    "rna": ("#fff2df", "#d58b2b"),  # orange -> workflow D, RNA structure / rG4
    "analysis": ("#f8eaf2", "#bc5c87"),  # pink   -> analysis steps in panel E
    "output": ("#e8f4f2", "#36958c"),  # teal   -> final outputs in panel E
}


# ---------------------------------------------------------------------------
# TEXT CATALOG
# ---------------------------------------------------------------------------
# All displayed copy, separate from geometry. Keys are stable identifiers:
# NodeSpec.text ("<prefix>" + ".title" / ".detail") and the render_* functions
# look strings up here. Use \n inside node titles/details for explicit line
# breaks. Keys are grouped by panel below; the order has no effect on the SVG.
# Values in the "detail" entries of file nodes are shown in monospace (file
# formats); step details are shown in the regular small font.
DEFAULT_TEXT = {
    # --- Document-level: title block and accessibility metadata -------------
    "svg.title": "LNClassifier workflow DAGs",  # big heading, top left
    "svg.subtitle": "Benchmark and feature generation run independently; their outputs converge for downstream analysis and publication.",
    "svg.versions": "Shared annotation and assembly for B–D: GENCODE v47 · GRCh38",
    "svg.accessible_title": "LNClassifier supplementary workflow DAGs",  # <title>, for screen readers
    "svg.desc": "Independent benchmark above three feature-generation workflows. RNA outputs include rG4 detection and peak counts, plus ScanFold2 features; cross-workflow tables converge for downstream analyses and publication figures.",  # <desc>, for screen readers
    # --- Panel A: model benchmark (also the panel-E analysis steps) ---------
    "revision.title": "A · Model benchmark",  # panel title
    "revision.input.title": "Transcript sequences\n+ annotations",
    "revision.input.detail": ".fa + .gtf",
    "revision.prepare.title": "Prepare dataset",
    "revision.prepare.detail": "v46 vs v47; CD-HIT",
    "revision.train.title": "Run training",
    "revision.train.detail": "8 classifiers, 5-fold CV",
    "revision.infer.title": "Run inference",
    "revision.infer.detail": "8 classifiers, 5-fold CV",
    "revision.process.title": "Process tables",
    "revision.process.detail": "Merge and summarize",
    "revision.model_features.title": "Model features",  # produced file, A and E
    "revision.model_features.detail": ".tsv",
    # Panel-E analysis steps (they live under "revision." because they analyze
    # the benchmark results):
    "revision.clustering.title": "Cluster features",
    "revision.clustering.detail": "feature correlations",
    "revision.shap.title": "Train RF + SHAP",
    "revision.shap.detail": "5-fold CV",
    "revision.statistics.title": "Compute univariate\nstatistics",
    "revision.statistics.detail": "group comparisons",
    "revision.entropy.title": "Compute uncertainty\nand group",
    "revision.entropy.detail": "uncertainty groups",
    "revision.other_figures.title": "Compute performance\nand consensus",
    "revision.other_figures.detail": ".pdf",
    # Final publication-files stack in panel E:
    "revision.output.title": "Publication figures, tables and supplementary analyses",
    "revision.output.detail": ".pdf + .png + .csv + .tsv",
    # --- Panel B: repetitive element (REP) features -------------------------
    "te.title": "B · Repetitive element features",
    "te.input.title": "Sequences + annotations\n(DNA + RNA)",
    "te.input.detail": ".fa + .gtf",
    "te.prepare.title": "Validate FASTA",
    "te.prepare.detail": "IDs, lengths, headers",
    "te.repeat.title": "Run RepeatMasker",
    "te.repeat.detail": "Annotate repeats",
    "te.merge.title": "Merge and summarize",
    "te.merge.detail": "Counts, coverage and stats",
    "te.output.title": "REP features",
    "te.output.detail": ".csv",
    # --- Panel C: non-B DNA (NBD) features ----------------------------------
    "nbd.title": "C · Non-B DNA features",
    "nbd.input.title": "Assembly + annotations\n(DNA only)",
    "nbd.input.detail": ".fa + .gtf",
    "nbd.prepare.title": "Build annotation BEDs",
    "nbd.prepare.detail": "transcript + exon BEDs",
    "nbd.motifs.title": "Run G4Discovery\nand pqsfinder",
    "nbd.motifs.detail": "Annotate 9 non-B DNA motifs",
    "nbd.extended.title": "Intersect and summarize",
    "nbd.extended.detail": "Counts, coverage and stats",
    "nbd.output.title": "NBD features",
    "nbd.output.detail": ".csv",
    # --- Panel D: RNA structure and RNA G4 features -------------------------
    "rna.title": "D · RNA structure and RNA G4 features",
    "rna.input.title": "Transcript FASTA\n(RNA only)",
    "rna.input.detail": ".fa",
    "rna.detect.title": "Run rG4detector",
    "rna.detect.detail": "rG4 per-nucleotide scores",
    "rna.merge.title": "Merge and summarize",
    "rna.merge.detail": "rG4 peak counts and stats",
    "rna.peaks.title": "rG4detector features",
    "rna.peaks.detail": ".csv",
    "rna.scanfold.title": "Run ScanFold2",
    "rna.scanfold.detail": "RNA base-pair prediction",
    "rna.scanfold_merge.title": "Merge and summarize",
    "rna.scanfold_merge.detail": "Summarize base pairs",
    "rna.scanfold_output.title": "ScanFold2 features",
    "rna.scanfold_output.detail": ".tsv",
    # --- Panel E: feature integration ("handoff") ---------------------------
    "handoff.title": "E · Feature integration, uncertainty analysis and publication files",
    # Top-row input files (copies of the B/C/D/A outputs):
    "handoff.te.title": "REP features",
    "handoff.te.detail": ".csv",
    "handoff.nbd.title": "NBD features",
    "handoff.nbd.detail": ".csv",
    "handoff.rg4.title": "rG4detector features",
    "handoff.rg4.detail": ".csv",
    "handoff.scanfold.title": "ScanFold2 features",
    "handoff.scanfold.detail": ".tsv",
    "handoff.revision.title": "Benchmark table",  # also used by node a-output in panel A
    "handoff.revision.detail": ".tsv",
    # --- Legend -------------------------------------------------------------
    "handoff.legend.step": "Step",
    "handoff.legend.file": "Input file",
    "handoff.legend.produced": "Output file",
    "handoff.dependencies": "Dependency flow",
    "handoff.legend.color": "Color = workflow",
}


# ---------------------------------------------------------------------------
# STYLES
# ---------------------------------------------------------------------------
# Shared CSS embedded in the SVG <style> element. Label line heights (20 for
# titles, 18 for details) are set separately (LABEL_LINE / DETAIL_LINE), not here.
STYLES = """
text{font-family:Inter,Arial,sans-serif;fill:#172033}
.title{font-size:28px;font-weight:700}
.subtitle{font-size:15px;fill:#526176}
.panel-title{font-size:20px;font-weight:700}
.node{stroke-width:2}
.label{font-size:15px;font-weight:600}
.small{font-size:14px;font-weight:400}
.file-format{font-family:ui-monospace,Consolas,monospace;font-size:14px;font-weight:400}
.edge{stroke:#64748b;stroke-width:2.5;fill:none;marker-end:url(#arrow)}
.edge-dash{stroke:#64748b;stroke-width:2.5;stroke-dasharray:8 7;fill:none;marker-end:url(#arrow)}
.legend{font-size:14px;fill:#526176}
"""


# ---------------------------------------------------------------------------
# DATA MODEL (dataclasses describing the workflow; no drawing logic here)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelSpec:
    """One rounded background rectangle grouping several nodes.

    id         Single letter "A".."E"; node ids start with it (any case).
    title_key  Key in DEFAULT_TEXT for the panel heading (drawn at panel-local
               x=25, y=37).
    fill       Hex background color of the panel.
    """

    id: str
    title_key: str
    fill: str


@dataclass(frozen=True)
class NodeSpec:
    """One box in the diagram. See "4.5 Add / remove / move a node".

    id        Unique id used by EdgeSpec; its first letter names the panel.
    text      DEFAULT_TEXT prefix: "<text>.title" (bold, may contain \\n) and
              "<text>.detail" (lighter second line(s)).
    kind      "step" | "file" | "produced_file" (shape, see render_node()).
    row, col  Grid slot (integers, 0-based) inside the panel.
    workflow  Color family "A".."E" (see workflow_colors()); empty = the
              node's own panel.
    width     Optional width override in units.
    """

    id: str
    text: str
    kind: str
    row: int
    col: int
    workflow: str = ""
    width: int | None = None

    @property
    def panel(self) -> str:
        """Panel letter, derived from the id ("b-merge" -> "B")."""
        return self.id[0].upper()

    @property
    def family(self) -> str:
        """Color family: the explicit workflow, else the panel letter."""
        return self.workflow or self.panel

    @property
    def label_key(self) -> str:
        return f"{self.text}.title"

    @property
    def sublabel_key(self) -> str:
        return f"{self.text}.detail"


@dataclass(frozen=True)
class EdgeSpec:
    """One arrow. See "4.6 Add / remove an edge" and section 6 (groups).

    src, dst      Node ids; both must be in the same panel.
    style         "solid" or "dashed" (only used for ungrouped edges).
    group         "" (independent arrow) or a bus-group name.
    src_anchor,   Side of the box the arrow leaves / enters:
    dst_anchor    "top" | "bottom" | "left" | "right".
    """

    src: str
    dst: str
    style: str = "solid"
    group: str = ""
    src_anchor: str = "bottom"
    dst_anchor: str = "top"


@dataclass(frozen=True)
class Box:
    """A node's computed rectangle in PANEL-LOCAL coordinates (top-left origin)."""

    x: int
    y: int
    width: int
    height: int
    node: NodeSpec

    def anchor(self, side: str) -> tuple[int, int]:
        """Return the midpoint of one side as an (x, y) point."""
        return {
            "top": (self.x + self.width // 2, self.y),
            "bottom": (self.x + self.width // 2, self.y + self.height),
            "left": (self.x, self.y + self.height // 2),
            "right": (self.x + self.width, self.y + self.height // 2),
        }[side]


@dataclass(frozen=True)
class GridLayout:
    """Grid parameters for one panel (used by layout()).

    first_row_top y of the top edge of row 0.
    row_height    Height of every node box.
    row_gutter    Vertical gap between consecutive rows.
    first_center  x of the center of column 0 (unused by panels A and E).
    column_pitch  Distance between column centers (unused by A and E).
    """

    first_row_top: int
    row_height: int
    row_gutter: int
    first_center: int = 0
    column_pitch: int = 0


# ---------------------------------------------------------------------------
# EDGE BUILDERS
# ---------------------------------------------------------------------------
def chain(*ids: str, **kwargs) -> tuple[EdgeSpec, ...]:
    """Edges a->b, b->c, ... through ``ids``; kwargs go to every EdgeSpec."""
    return tuple(EdgeSpec(a, b, **kwargs) for a, b in zip(ids, ids[1:]))


def fan_in(srcs: tuple[str, ...], dst: str, group: str) -> tuple[EdgeSpec, ...]:
    """Many sources -> one target, all in one bus ``group``."""
    return tuple(EdgeSpec(src, dst, group=group) for src in srcs)


def fan_out(src: str, dsts: tuple[str, ...], group: str) -> tuple[EdgeSpec, ...]:
    """One source -> many targets, all in one bus ``group``."""
    return tuple(EdgeSpec(src, dst, group=group) for dst in dsts)


# ---------------------------------------------------------------------------
# WORKFLOW DECLARATION: PANELS, NODES, EDGES
# ---------------------------------------------------------------------------
# Panels are drawn in this order (later panels paint on top of earlier ones).
PANELS = (
    PanelSpec("A", "revision.title", "#fcfbff"),
    PanelSpec("B", "te.title", "#fbfdff"),
    PanelSpec("C", "nbd.title", "#fbfffc"),
    PanelSpec("D", "rna.title", "#fffdf9"),
    PanelSpec("E", "handoff.title", "#f8fafc"),
)
PANEL_BY_ID = {panel.id: panel for panel in PANELS}

# Node declarations contain semantic fields and grid positions only:
#   NodeSpec(id, text prefix, kind, row, col, workflow="", width=None)
# Grid-position notes per panel:
#   A: one row, columns 0-5 left to right; the two last nodes have manual y.
#   B, C: a single column (col 0), rows 0-4 top to bottom.
#   D: input at col 1; ScanFold2 chain in col 0; rG4detector chain in col 2.
#   E: row 0 = input files; row 2 = analysis steps; row 4 = final output
#      (positions overridden in layout(); see section 7 of the module doc).
NODES = (
    # ---- Panel A: model benchmark (left -> right) ----
    NodeSpec("a-input", "revision.input", "file", 0, 0),
    NodeSpec("a-prepare", "revision.prepare", "step", 0, 1),
    NodeSpec("a-train", "revision.train", "step", 0, 2),
    NodeSpec("a-infer", "revision.infer", "step", 0, 3),
    NodeSpec("a-process", "revision.process", "step", 0, 4),
    NodeSpec("a-output", "handoff.revision", "produced_file", 0, 5),
    NodeSpec("a-model-features", "revision.model_features", "produced_file", 1, 5),
    # ---- Panel B: repetitive elements (top -> bottom) ----
    NodeSpec("b-input", "te.input", "file", 0, 0),
    NodeSpec("b-prepare", "te.prepare", "step", 1, 0),
    NodeSpec("b-repeat", "te.repeat", "step", 2, 0),
    NodeSpec("b-merge", "te.merge", "step", 3, 0),
    NodeSpec("b-output", "te.output", "produced_file", 4, 0),
    # ---- Panel C: non-B DNA (top -> bottom) ----
    NodeSpec("c-input", "nbd.input", "file", 0, 0),
    NodeSpec("c-prepare", "nbd.prepare", "step", 1, 0),
    NodeSpec("c-motifs", "nbd.motifs", "step", 2, 0),
    NodeSpec("c-extended", "nbd.extended", "step", 3, 0),
    NodeSpec("c-output", "nbd.output", "produced_file", 4, 0),
    # ---- Panel D: RNA structure / rG4 (input in the middle, two branches) ----
    NodeSpec("d-input", "rna.input", "file", 0, 1),
    NodeSpec("d-detect", "rna.detect", "step", 1, 2),
    NodeSpec("d-scanfold", "rna.scanfold", "step", 1, 0),
    NodeSpec("d-merge", "rna.merge", "step", 2, 2),
    NodeSpec("d-scanfold-merge", "rna.scanfold_merge", "step", 2, 0),
    NodeSpec("d-scanfold-output", "rna.scanfold_output", "produced_file", 3, 0),
    NodeSpec("d-peaks-output", "rna.peaks", "produced_file", 3, 2),
    # ---- Panel E, row 0: copies of the produced files from A-D ----
    # (the workflow keeps the source panel's color; width 192 is narrower than
    # the default so six files fit in one row.)
    NodeSpec("e-te", "handoff.te", "produced_file", 0, 0, "B", 192),
    NodeSpec("e-nbd", "handoff.nbd", "produced_file", 0, 1, "C", 192),
    NodeSpec("e-scanfold", "handoff.scanfold", "produced_file", 0, 2, "D", 192),
    NodeSpec("e-rg4", "handoff.rg4", "produced_file", 0, 3, "D", 192),
    NodeSpec(
        "e-model-features", "revision.model_features", "produced_file", 0, 4, "A", 192
    ),
    NodeSpec("e-benchmark", "handoff.revision", "produced_file", 0, 5, "A", 192),
    # ---- Panel E, row 2: analysis steps (pink) ----
    NodeSpec("e-clustering", "revision.clustering", "step", 2, 0),
    NodeSpec("e-shap", "revision.shap", "step", 2, 1),
    NodeSpec("e-statistics", "revision.statistics", "step", 2, 2),
    NodeSpec("e-entropy", "revision.entropy", "step", 2, 3),
    NodeSpec("e-other-figures", "revision.other_figures", "step", 2, 4),
    # ---- Panel E, row 4: final publication files (drawn as a stack of sheets) ----
    NodeSpec("e-output", "revision.output", "produced_file", 4, 0, width=580),
)

# Panel A's chain runs left to right, so it leaves/enters on the sides.
A_FLOW = {"src_anchor": "right", "dst_anchor": "left"}

# Edges are drawn by the panel that owns their SOURCE node. Order matters only
# for the ungrouped edges, which are painted in this order after the groups.
EDGES = (
    # ---- Panel A: left-to-right chain; the last step fans out to two files ----
    *chain(
        "a-input", "a-prepare", "a-train", "a-infer", "a-process", "a-output", **A_FLOW
    ),
    EdgeSpec("a-process", "a-model-features", **A_FLOW),
    # ---- Panels B and C: vertical chains ----
    *chain("b-input", "b-prepare", "b-repeat", "b-merge", "b-output"),
    *chain("c-input", "c-prepare", "c-motifs", "c-extended", "c-output"),
    # ---- Panel D: the input splits (shared "rna-input" bus) into two chains ----
    *fan_out("d-input", ("d-detect", "d-scanfold"), "rna-input"),
    EdgeSpec("d-detect", "d-merge"),
    *chain("d-scanfold", "d-scanfold-merge", "d-scanfold-output"),
    EdgeSpec("d-merge", "d-peaks-output"),
    # ---- Panel E ----
    # Merge bus "handoff": five feature files -> the dashed analysis frame.
    *fan_in(
        ("e-te", "e-nbd", "e-rg4", "e-scanfold", "e-model-features"),
        "e-clustering",
        "handoff",
    ),
    # Fan-out bus "analysis-out": benchmark table -> entropy + performance steps.
    *fan_out("e-benchmark", ("e-entropy", "e-other-figures"), "analysis-out"),
    # Merge bus "final-in": every analysis step -> publication files.
    *fan_in(
        ("e-clustering", "e-shap", "e-statistics", "e-entropy", "e-other-figures"),
        "e-output",
        "final-in",
    ),
)


# ---------------------------------------------------------------------------
# LAYOUT CONSTANTS
# ---------------------------------------------------------------------------
# Default node width by kind (used when NodeSpec.width is None; panel A uses
# A_COLUMN_WIDTHS instead).
NODE_WIDTHS = {"step": 240, "file": 280, "produced_file": 260}

# Panel A only: width of each of the six columns, sized for the longest label
# or sublabel in that column. Widen a column if you lengthen its text.
A_COLUMN_WIDTHS = (310, 250, 156, 166, 208, 188)
A_COLUMN_GAPS = (22, 22, 22, 22, 64)  # gap after column i, i.e. before column i+1

# Panel A only: hand-set tops for the two stacked right-hand files.
A_TOPS = {"a-output": 26, "a-model-features": 154}

# Per-panel grid (see GridLayout). Panel A uses A_COLUMN_WIDTHS for x and E
# overrides both axes in _e_position(), so their centers/pitch stay at 0.
GRID = {
    "A": GridLayout(first_row_top=90, row_height=72, row_gutter=0),
    "B": GridLayout(70, 72, 24, first_center=165, column_pitch=240),
    "C": GridLayout(70, 72, 24, first_center=165, column_pitch=240),
    "D": GridLayout(70, 72, 24, first_center=165, column_pitch=165),
    "E": GridLayout(120, 72, 64, first_center=155, column_pitch=95),
}

# Canvas y of each panel's top edge. (Panel x positions are computed; see
# _middle_row_positions() and panel_origin().)
PANEL_Y = {"A": 120, "B": 400, "C": 400, "D": 400, "E": 1000}

# Panel E: x of its left edge on the canvas, vertical shift (negative = move
# up), hand-placed row geometry, and the frame around the analysis steps.
E_X = 97
E_DY = -45
E_INPUT_TOP, E_ANALYSIS_TOP, E_OUTPUT_TOP = 120, 392, 624
E_ANALYSIS_CENTERS = (317, 602, 887, 1233, 1528)  # uneven spacing on purpose
E_FRAME = (167, 362 + E_DY, 870, 132)  # x, y, width, height

# Bus geometry (see section 6) and the offset of the stacked-sheet look.
BUS_DROP = 100  # merge buses sit this far below their first source
FANOUT_GAP = 32  # "analysis-out" bus sits this far above its first target
STACK_OFFSETS = (12, 6)  # offsets of the two back sheets behind "e-output"

# Line heights used to center text inside nodes, and the dog-ear size.
LABEL_LINE, DETAIL_LINE, FOLD = 20, 18, 18

# Legend: (shape, x offset inside the legend group, DEFAULT_TEXT key), the
# label's x distance from the sample's x, and the width reserved for centering.
LEGEND_FILL, LEGEND_STROKE = COLORS["te"]
LEGEND_ITEMS = (
    ("step", 35, "handoff.legend.step"),
    ("file", 145, "handoff.legend.file"),
    ("produced", 278, "handoff.legend.produced"),
    ("arrow", 400, "handoff.dependencies"),
    ("color", 630, "handoff.legend.color"),
)
LEGEND_LABEL_DX = {"step": 40, "file": 46, "produced": 46, "arrow": 60, "color": 18}
LEGEND_WIDTH = 680


# ---------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# ---------------------------------------------------------------------------
def txt(x: float, y: float, value: str, cls: str, anchor: str = "start") -> str:
    """Return one SVG <text> element.

    x, y    Baseline position (y is the text baseline, not the top).
    value   Plain text; escaped automatically (& < > become entities).
    cls     CSS class from STYLES (e.g. "label", "legend").
    anchor  SVG text-anchor: "start" | "middle" | "end".
    """
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def panel_nodes(panel_id: str) -> tuple[NodeSpec, ...]:
    """All NodeSpecs drawn in one panel, in declaration order."""
    return tuple(node for node in NODES if node.panel == panel_id)


def _e_position(node: NodeSpec) -> tuple[int, int]:
    """(center_x, top) of a panel-E node: hand-placed, not grid-derived."""
    if node.row == 0:  # input-file row: six files at a 251-unit pitch
        return 132 + 251 * node.col, E_INPUT_TOP + E_DY
    if node.row == 2:  # analysis-step row: explicit x centers
        return E_ANALYSIS_CENTERS[node.col], E_ANALYSIS_TOP + E_DY
    return 944, E_OUTPUT_TOP + E_DY  # final output box


def layout(panel: PanelSpec, specs: tuple[NodeSpec, ...]) -> dict[str, Box]:
    """Convert (row, col) grid slots to pixel Boxes for one panel.

    Returns {node_id: Box} in PANEL-LOCAL coordinates. This is where the
    per-panel special cases live (panel A column widths and manual tops, the
    extra spacing in panel D, and the hand-placed positions in panel E).
    """
    grid = GRID[panel.id]
    boxes = {}
    for node in specs:
        width = node.width if node.width is not None else NODE_WIDTHS[node.kind]
        center_x = grid.first_center + node.col * grid.column_pitch
        top = grid.first_row_top + node.row * (grid.row_height + grid.row_gutter)
        if panel.id == "A":
            # Variable column widths: pack columns left to right with a gap.
            width = A_COLUMN_WIDTHS[node.col]
            center_x = (
                PANEL_PADDING
                + sum(A_COLUMN_WIDTHS[: node.col])
                + sum(A_COLUMN_GAPS[: node.col])
                + width // 2
            )
            top = A_TOPS.get(node.id, top)
        elif panel.id == "D" and node.row > 0:
            # Leave space for the split arrowheads below the horizontal bus.
            top += 2 * grid.row_gutter
        elif panel.id == "E":
            center_x, top = _e_position(node)
        boxes[node.id] = Box(center_x - width // 2, top, width, grid.row_height, node)
    return boxes


def panel_size(panel: PanelSpec, boxes: dict[str, Box]) -> tuple[int, int]:
    """Return (width, height) of a panel in units.

    B, C, D: tightest box around the nodes plus PANEL_PADDING.
    A and E: fixed sizes (edit the numbers here if their content changes).
    """
    right = max(box.x + box.width for box in boxes.values())
    bottom = max(box.y + box.height for box in boxes.values())
    if panel.id == "A":
        return right + PANEL_PADDING, 265
    if panel.id == "E":
        return 1695, 721 + E_DY
    return right + PANEL_PADDING, bottom + PANEL_PADDING


def _middle_row_positions(gutter: int = 40) -> dict[str, int]:
    """Center panels B, C, D as a group, spaced by a fixed gutter, using rendered widths.

    Returns {panel_id: canvas x of the panel's left edge}. ``gutter`` is the
    horizontal space between neighbouring panels; raise it to spread them out.
    """
    widths = {}
    for panel_id in ("B", "C", "D"):
        panel = PANEL_BY_ID[panel_id]
        widths[panel_id], _ = panel_size(panel, layout(panel, panel_nodes(panel_id)))
    total_width = widths["B"] + widths["C"] + widths["D"] + 2 * gutter
    b_x = (W - total_width) // 2
    c_x = b_x + widths["B"] + gutter
    d_x = c_x + widths["C"] + gutter
    return {"B": b_x, "C": c_x, "D": d_x}


# Computed once at import time (uses layout(), so it must come after it).
PANEL_X = _middle_row_positions()


def legend_baseline() -> int:
    """y of the legend text baseline: 38 units below panel E's bottom edge."""
    panel = PANEL_BY_ID["E"]
    _, height = panel_size(panel, layout(panel, panel_nodes("E")))
    return PANEL_Y["E"] + height + 38


def panel_origin(panel: PanelSpec, width: int) -> tuple[int, int]:
    """Canvas (x, y) of a panel's top-left corner.

    E is pinned at E_X; B, C, D use PANEL_X; A (not in PANEL_X) is centered.
    """
    x = E_X if panel.id == "E" else PANEL_X.get(panel.id, (W - width) // 2)
    return x, PANEL_Y[panel.id]


# ---------------------------------------------------------------------------
# EDGE DRAWING
# ---------------------------------------------------------------------------
def edge_path(
    points: tuple[tuple[int, int], ...],
    style: str,
    arrow: bool = True,
    destination: Box | None = None,
) -> str:
    """Return an SVG <path> through ``points`` (a polyline of (x, y) corners).

    style        "solid" -> class edge, "dashed" -> class edge-dash.
    arrow        True: arrowhead at the last point (via the CSS marker-end).
                 False: plain line, used for bus stubs and bus bars.
    destination  If given (with arrow), asserts the last point lies on that
                 box's outline so arrowheads never float in space.
    """
    if arrow and destination is not None:
        assert_endpoint_on_box(points[-1], destination)
    path = "M" + " L".join(f"{x} {y}" for x, y in points)
    css = {"solid": "edge", "dashed": "edge-dash"}[style]
    marker = "" if arrow else ' style="marker-end:none"'
    return f'<path d="{path}" class="{css}"{marker}/>'


def assert_endpoint_on_box(point: tuple[int, int], box: Box) -> None:
    """Raise AssertionError unless ``point`` is on the outline of ``box``.

    This is the built-in geometry check: if you move a node and an arrow no
    longer meets it, the script fails loudly instead of drawing a gap.
    """
    x, y = point
    on_vertical_side = (
        x in (box.x, box.x + box.width) and box.y <= y <= box.y + box.height
    )
    on_horizontal_side = (
        y in (box.y, box.y + box.height) and box.x <= x <= box.x + box.width
    )
    assert (
        on_vertical_side or on_horizontal_side
    ), f"edge endpoint {point} does not lie on destination {box.node.id}"


def orthogonal_route(
    source: Box, target: Box, src_anchor: str, dst_anchor: str
) -> tuple[tuple[int, int], ...]:
    """Choose a horizontal/vertical-only route for an ungrouped edge.

    Cases, in order:
      right -> right     Detour through a corridor just right of both boxes
                         (bypasses shared buses).
      already aligned    Straight line (same x or same y).
      left -> top        One elbow: horizontal to the target's x, then down.
      left/right start   Horizontal, vertical at the mid x, horizontal.
      otherwise          Vertical, horizontal at the mid y, vertical.
    """
    start, end = source.anchor(src_anchor), target.anchor(dst_anchor)
    if src_anchor == dst_anchor == "right":
        # Side branches bypass the shared buses through an open corridor.
        channel_x = max(start[0], end[0]) + GRID[source.node.panel].row_gutter // 4
        return start, (channel_x, start[1]), (channel_x, end[1]), end
    if start[0] == end[0] or start[1] == end[1]:
        return start, end
    if src_anchor == "left" and dst_anchor == "top":
        return start, (end[0], start[1]), end
    if src_anchor in {"left", "right"}:
        channel_x = (start[0] + end[0]) // 2
        return start, (channel_x, start[1]), (channel_x, end[1]), end
    bus_y = (start[1] + end[1]) // 2
    return start, (start[0], bus_y), (end[0], bus_y), end


def _fan_out_bus(source: Box, targets: list[Box], bus_y: int) -> list[str]:
    """ONE source -> MANY targets: stem down, bar across, arrows down.

    Each arrow ends on its target's top edge (checked by edge_path()).
    """
    stem = source.anchor("bottom")
    centers = [box.anchor("top")[0] for box in targets]
    paths = [
        edge_path((stem, (stem[0], bus_y)), "solid", False),
        edge_path(((min(centers), bus_y), (max(centers), bus_y)), "solid", False),
    ]
    paths.extend(
        edge_path(((x, bus_y), box.anchor("top")), "solid", destination=box)
        for x, box in zip(centers, targets)
    )
    return paths


def _merge_bus(ports: list[tuple[int, int]], bus_y: int) -> list[str]:
    """MANY ports -> one horizontal bus: a stub from each port, then the bar.

    The caller adds the single arrow that leaves the bus.
    """
    xs = [port[0] for port in ports]
    paths = [edge_path((port, (port[0], bus_y)), "solid", False) for port in ports]
    paths.append(edge_path(((min(xs), bus_y), (max(xs), bus_y)), "solid", False))
    return paths


def _inside_frame(box: Box) -> bool:
    """True if ``box`` lies entirely inside the dashed analysis frame."""
    fx, fy, fw, fh = E_FRAME
    return (
        fx <= box.x
        and box.x + box.width <= fx + fw
        and fy <= box.y
        and box.y + box.height <= fy + fh
    )


def _bus_rna_input(edges: list[EdgeSpec], boxes: dict[str, Box]) -> list[str]:
    """Bus sits half way between the source bottom and the highest target top."""
    source = boxes[edges[0].src]
    targets = [boxes[edge.dst] for edge in edges]
    bus_y = (source.anchor("bottom")[1] + min(box.y for box in targets)) // 2
    return _fan_out_bus(source, targets, bus_y)


def _bus_analysis_out(edges: list[EdgeSpec], boxes: dict[str, Box]) -> list[str]:
    """Bus sits FANOUT_GAP units above the first target."""
    source = boxes[edges[0].src]
    targets = [boxes[edge.dst] for edge in edges]
    return _fan_out_bus(source, targets, targets[0].y - FANOUT_GAP)


def _bus_handoff(edges: list[EdgeSpec], boxes: dict[str, Box]) -> list[str]:
    """Feature files merge into ONE arrow that enters the top of the frame."""
    ports = [boxes[edge.src].anchor("bottom") for edge in edges]
    bus_y = ports[0][1] + BUS_DROP
    fx, fy, fw, _ = E_FRAME
    frame_top = (fx + fw // 2, fy)
    return _merge_bus(ports, bus_y) + [
        edge_path(((frame_top[0], bus_y), frame_top), "solid")
    ]


def _bus_final_in(edges: list[EdgeSpec], boxes: dict[str, Box]) -> list[str]:
    """Analysis steps merge into the publication-files stack.

    Steps inside the dashed frame share ONE port on its bottom edge. The
    arrow stops at the top of the stack's back sheet.
    """
    sources = [boxes[edge.src] for edge in edges]
    target = boxes[edges[0].dst]
    bus_y = sources[0].anchor("bottom")[1] + BUS_DROP
    fx, fy, fw, fh = E_FRAME
    ports = []
    for box in sources:
        port = (fx + fw // 2, fy + fh) if _inside_frame(box) else box.anchor("bottom")
        if port not in ports:
            ports.append(port)
    center_x, top_y = target.anchor("top")
    stack_top = (center_x, top_y - STACK_OFFSETS[0])
    return _merge_bus(ports, bus_y) + [
        edge_path(((center_x, bus_y), stack_top), "solid")
    ]


# Group name -> drawer. Drawing order follows this dict (see section 6).
GROUP_DRAWERS = {
    "handoff": _bus_handoff,
    "rna-input": _bus_rna_input,
    "analysis-out": _bus_analysis_out,
    "final-in": _bus_final_in,
}


def render_edges(panel_edges: tuple[EdgeSpec, ...], boxes: dict[str, Box]) -> list[str]:
    """Return SVG path strings for every edge whose source is in this panel.

    Two passes: (1) grouped edges are drawn as shared buses, one group at a
    time via GROUP_DRAWERS (see module doc, section 6); (2) all remaining
    edges are drawn as independent orthogonal arrows.
    """
    unknown = {edge.group for edge in panel_edges} - {""} - set(GROUP_DRAWERS)
    if unknown:
        raise ValueError(
            f"unregistered edge group(s): {sorted(unknown)}; see GROUP_DRAWERS"
        )
    # Sanity check: every anchor named by an edge must lie on its own box.
    for edge in panel_edges:
        assert_endpoint_on_box(boxes[edge.src].anchor(edge.src_anchor), boxes[edge.src])
        assert_endpoint_on_box(boxes[edge.dst].anchor(edge.dst_anchor), boxes[edge.dst])
    rendered = []
    for group, draw in GROUP_DRAWERS.items():
        group_edges = [edge for edge in panel_edges if edge.group == group]
        if group_edges:
            rendered.extend(draw(group_edges, boxes))
    # Pass 2: every edge that did not belong to a group above.
    for edge in panel_edges:
        if edge.group:
            continue
        source, target = boxes[edge.src], boxes[edge.dst]
        points = orthogonal_route(source, target, edge.src_anchor, edge.dst_anchor)
        rendered.append(edge_path(points, edge.style, destination=target))
    return rendered


# ---------------------------------------------------------------------------
# NODE, PANEL AND LEGEND DRAWING
# ---------------------------------------------------------------------------
def workflow_colors(node: NodeSpec) -> tuple[str, str]:
    """Return (fill, border) for a node, chosen by its color family.

    Exception: analysis STEPS in panel E always use the "analysis" pink pair,
    regardless of workflow.
    """
    palette = {
        "A": COLORS["rev"],
        "B": COLORS["te"],
        "C": COLORS["nbd"],
        "D": COLORS["rna"],
        "E": COLORS["output"],
    }
    if node.panel == "E" and node.kind == "step":
        return COLORS["analysis"]
    return palette[node.family]


def render_node(box: Box, text: dict[str, str]) -> str:
    """Return SVG for one node: its shape followed by its centered text.

    Shapes by ``kind``:
      step           Rounded rectangle (rx=10) with the workflow fill color.
      file           White page with the top-right corner cut and an OUTLINED
                     fold line (input file).
      produced_file  Same page with a FILLED triangular fold (output file).
    Text: title lines (class "label", LABEL_LINE units per line) then detail
    lines (DETAIL_LINE units per line; monospace "file-format" class for files,
    "small" for steps). The block is centered vertically in the box. Long lines
    are NOT wrapped or shrunk; use \\n in the text or widen the box.
    """
    workflow_fill, stroke = workflow_colors(box.node)
    is_file = box.node.kind != "step"
    fill = "#ffffff" if is_file else workflow_fill
    if is_file:
        right = box.x + box.width
        body = (
            f'<path d="M {box.x} {box.y} H {right - FOLD} '
            f"L {right} {box.y + FOLD} V {box.y + box.height} "
            f'H {box.x} Z" class="node" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2.5"/>'
        )
        if box.node.kind == "produced_file":
            # Filled dog-ear marks a file produced by the pipeline.
            ear = (
                f'<path d="M {right - FOLD} {box.y} '
                f"L {right} {box.y + FOLD} "
                f'L {right - FOLD} {box.y + FOLD} Z" '
                f'fill="{stroke}" stroke="{stroke}" stroke-width="1"/>'
            )
        else:
            ear = (
                f'<path d="M {right - FOLD} {box.y} '
                f'V {box.y + FOLD} H {right}" fill="none" '
                f'stroke="{stroke}" stroke-width="2.5"/>'
            )
        rect = body + ear
    else:
        rect = (
            f'<rect x="{box.x}" y="{box.y}" width="{box.width}" height="{box.height}" '
            f'rx="10" class="node" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="2.5"/>'
        )
    label = text[box.node.label_key]
    detail = text[box.node.sublabel_key]
    # Each entry: (text, css class, line height in units).
    lines = [(line, "label", LABEL_LINE) for line in label.splitlines()]
    detail_class = "file-format" if is_file else "small"
    lines += [(line, detail_class, DETAIL_LINE) for line in detail.splitlines()]
    # Start y so that the whole text block is vertically centered in the box.
    y = box.y + (box.height - sum(height for _, _, height in lines)) / 2
    texts = []
    for value, cls, height in lines:
        texts.append(txt(box.x + box.width / 2, y + height - 4, value, cls, "middle"))
        y += height
    if box.node.id == "e-output":
        # Offset sheets communicate a collection of publication files.
        sheets = "".join(
            f'<g transform="translate({offset} {-offset})">{rect}</g>'
            for offset in STACK_OFFSETS
        )
        rect = sheets + rect
    return rect + "".join(texts)


def render_panel(panel: PanelSpec, text: dict[str, str]) -> str:
    """Return the SVG <g> for one panel: background, title, edges, nodes.

    Everything inside is in panel-local coordinates; the enclosing translate()
    moves it to its canvas position (panel_origin()). Drawing order (back to
    front): background, title, panel-E dashed group, edges, nodes.
    """
    specs = panel_nodes(panel.id)
    ids = {node.id for node in specs}
    edges = tuple(edge for edge in EDGES if edge.src in ids)
    boxes = layout(panel, specs)
    width, height = panel_size(panel, boxes)
    origin_x, origin_y = panel_origin(panel, width)
    background = f'<rect width="{width}" height="{height}" rx="18" fill="{panel.fill}" stroke="#cbd5e1"/>'
    groups = ""
    if panel.id == "E":
        # Dashed frame around the analysis steps (see module doc section 7)
        # plus a short solid line at the frame's mid-height that starts at the
        # entropy box and runs to the frame's right edge.
        fx, fy, fw, fh = E_FRAME
        mid_y = fy + fh // 2
        groups = (
            f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="14" '
            'fill="none" stroke="#bc5c87" stroke-width="2.5" stroke-dasharray="8 6"/>'
            + edge_path(((boxes["e-entropy"].x, mid_y), (fx + fw, mid_y)), "solid")
        )
    content = (
        f'<g transform="translate({origin_x} {origin_y})">'
        + background
        + txt(25, 37, text[panel.title_key], "panel-title")
        + groups
        + "".join(render_edges(edges, boxes))
        + "".join(render_node(box, text) for box in boxes.values())
    )
    return content + "</g>"


def _legend_sample(shape: str, x: int, y: int) -> str:
    """Return the small sample graphic for one legend item (blue palette)."""
    if shape == "step":
        return (
            f'<rect x="{x}" y="{y - 13}" width="30" height="20" rx="5" '
            f'fill="{LEGEND_FILL}" stroke="{LEGEND_STROKE}" stroke-width="2"/>'
        )
    if shape in ("file", "produced"):
        sample = (
            f'<path d="M {x} {y - 13} H {x + 26} L {x + 36} {y - 3} V {y + 7} H {x} Z" '
            f'fill="#fff" stroke="{LEGEND_STROKE}" stroke-width="2"/>'
        )
        if shape == "produced":
            sample += (
                f'<path d="M {x + 26} {y - 13} L {x + 36} {y - 3} L {x + 26} {y - 3} Z" '
                f'fill="{LEGEND_STROKE}"/>'
            )
        return sample
    if shape == "arrow":
        return f'<path d="M {x} {y - 3} H {x + 50}" class="edge"/>'
    return (
        f'<circle cx="{x}" cy="{y - 3}" r="8" fill="{LEGEND_FILL}" '
        f'stroke="{LEGEND_STROKE}" stroke-width="2"/>'
    )


def render_legend(text: dict[str, str], y: int) -> str:
    """Return the legend row at baseline ``y`` (canvas coordinates).

    One sample + label per LEGEND_ITEMS row, with literal x offsets relative to
    the legend group (which svg() centers using LEGEND_WIDTH). The samples use
    the blue (workflow B) colors.
    """
    return "".join(
        _legend_sample(shape, x, y)
        + txt(x + LEGEND_LABEL_DX[shape], y + 2, text[key], "legend")
        for shape, x, key in LEGEND_ITEMS
    )


# ---------------------------------------------------------------------------
# ASSEMBLY AND COMMAND LINE
# ---------------------------------------------------------------------------
def resolve_text(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Validate partial text overrides and return a fresh, complete catalog.

    None -> a copy of DEFAULT_TEXT. Otherwise every key must already exist in
    DEFAULT_TEXT and every value must be a string (ValueError otherwise);
    the result is DEFAULT_TEXT updated with the overrides. DEFAULT_TEXT itself
    is never modified.
    """
    if overrides is None:
        return DEFAULT_TEXT.copy()
    if not isinstance(overrides, dict):
        raise ValueError("text overrides must be a JSON object of key/string pairs")
    for key, value in overrides.items():
        if key not in DEFAULT_TEXT:
            raise ValueError(
                f"unknown text key: {key!r}; use --dump-text to list valid keys"
            )
        if not isinstance(value, str):
            raise ValueError(f"text value for {key!r} must be a string")
    return {**DEFAULT_TEXT, **overrides}


def svg(text: dict[str, str] | None = None) -> str:
    """Render an SVG with optional partial text overrides, without mutating defaults.

    Assembles: <svg> root, accessible <title>/<desc>, arrow marker + CSS in
    <defs>, white background, title block (y=58/86/108), the five panels in
    PANELS order, then the centered legend. Reads and writes no files.
    (The odd indentation inside the f-string is cosmetic and only affects the
    whitespace of the generated file.)
    """
    text = resolve_text(text)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <title>{escape(text['svg.accessible_title'])}</title>
  <desc>{escape(text['svg.desc'])}</desc>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#64748b"/></marker>
    <style>{STYLES}</style>
  </defs>
  <rect width="{W}" height="{H}" fill="#ffffff"/>
  {txt(70, 58, text['svg.title'], "title")}
  {txt(70, 86, text['svg.subtitle'], "subtitle")}
    {txt(70, 108, text['svg.versions'], "subtitle")}
    {''.join(render_panel(panel, text) for panel in PANELS)}
        <g transform="translate({(W - LEGEND_WIDTH) // 2} 0)">{render_legend(text, legend_baseline())}</g>
</svg>
"""


def main() -> None:
    """Command-line entry point (see "1. Quick start" in the module doc)."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip().split("\n\n")[0],
        epilog=(
            "Full guide: the module docstring of this script (sections 1-9) and "
            "supplementary_pipeline_dag_generation.md."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("supplementary_pipeline_dag.svg"),
    )
    parser.add_argument(
        "--text",
        type=Path,
        metavar="JSON",
        help="UTF-8 JSON object of text overrides (partial or complete)",
    )
    parser.add_argument(
        "--dump-text",
        type=Path,
        metavar="JSON",
        help="write the complete text catalog and exit without rendering SVG",
    )
    args = parser.parse_args()
    try:
        text = resolve_text(
            json.loads(args.text.read_text(encoding="utf-8")) if args.text else None
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dump_text:
        args.dump_text.parent.mkdir(parents=True, exist_ok=True)
        args.dump_text.write_text(
            json.dumps(text, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.dump_text}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg(text), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
