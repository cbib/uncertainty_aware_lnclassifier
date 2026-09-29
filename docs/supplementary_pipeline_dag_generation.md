# Supplementary workflow DAGs: specification and tutorial

## Overview layout

The overview follows the manually edited Illustrator layout. Panel A places
benchmark tables and model features above and below the final processing step.
Panels B–D begin at y=430; panel E begins at y=1057. These positions preserve
the edited spacing, using normalized panel coordinates instead of Illustrator's
nested transforms and negative offsets.

Panel E places six file references in one row. Five feature files feed the
dashed group containing clustering, RF + SHAP, and univariate statistics.
The benchmark table feeds uncertainty grouping and performance/consensus at
right; uncertainty also connects to the feature-analysis group. All five
analyses converge on a stack of publication files. The simplified legend sits
below panel E. Connector shafts missing from the Illustrator export are
restored, and the generator retains its original typography and accessibility
metadata.

Regenerate the overview after editing its builder:

```bash
python revision-wt/docs/build_supplementary_pipeline_dag.py
python -m unittest discover -s revision-wt/docs -p 'test_supplementary_pipeline_dag.py'
```

This layout supersedes any earlier description below that places feature
generation before the benchmark itself: only downstream feature analysis
requires both sources of tables.

This document specifies how to regenerate the workflow DAGs and the
supplementary SVG overview:

[`supplementary_pipeline_dag.svg`](supplementary_pipeline_dag.svg)

The project contains independent Snakemake workflows. They are not combined
into one executable DAG. The intended execution order is:

```text
te_pipeline       ┐
nonb-pipeline     ├─> feature files ─> revision-wt
secondrna         ┘
```

The upstream workflows must be run before the downstream `revision-wt`
analysis. Their outputs are currently supplied to `revision-wt` through its
configuration files.

## 1. What the supplementary figure contains

The final figure has five panels:

| Panel | Content |
|---|---|
| A | `revision-wt` benchmark |
| B | TE annotation, RepeatMasker and TE-feature generation |
| C | Non-B DNA annotation, motif generation, intersections and statistics |
| D | Default rG4 detection, batch merging, summaries, peak statistics and visualizations, plus a separate ScanFold2 branch |
| E | Required execution order and cross-workflow handoffs |

Panels A–D are abstractions of the generated Snakemake DAGs. The full DOT
files and SnakeVision SVGs are the technical, rule-level record.

## 2. Environment

Use the `snakenew` Conda environment. In this workspace the environment's
activation wrapper does not reliably expose its `bin` directory, so the
commands below use absolute paths.

```bash
export SNAKEMAKE=/home/dgarcia/miniforge3/envs/snakenew/bin/snakemake
export SNAKEVISION=/home/dgarcia/miniforge3/envs/snakenew/bin/snakevision
export DAG_TMP=/tmp/lnc-snakemake-cache
export DAG_TEMP=/tmp/lnc-tmp

mkdir -p "$DAG_TMP" "$DAG_TEMP"

"$SNAKEMAKE" --version
"$SNAKEVISION" --version
```

Expected versions at the time this document was written:

```text
Snakemake 9.22.0
```

The temporary directories are needed because Snakemake creates a source cache
while parsing workflows. They must be writable.

## 3. Generate the DOT job DAGs

Run these commands from the repository root. `--dag dot` emits the resolved
job DAG for the selected target. It expands configured datasets, chunks,
chromosomes, folds and model branches.

```bash
ROOT=/mnt/cbib/LNClassifier/paper
OUT=/tmp/lnclassifier-dags
mkdir -p "$OUT"
```

### TE pipeline

```bash
cd "$ROOT/te_pipeline"

XDG_CACHE_HOME="$DAG_TMP" TMPDIR="$DAG_TEMP" \
  "$SNAKEMAKE" \
    --snakefile Snakefile \
    --configfile config/config_47.yaml \
    --cores 1 \
    --dag dot all \
    > "$OUT/te-pipeline-dag.dot"
```

The v47 configuration currently expands four RepeatMasker chunks.

### Non-B DNA pipeline

```bash
cd "$ROOT/nonb-pipeline"

XDG_CACHE_HOME="$DAG_TMP" TMPDIR="$DAG_TEMP" \
  "$SNAKEMAKE" \
    --snakefile Snakefile \
    --configfile config/config.yaml \
    --cores 1 \
    --dag dot all \
    > "$OUT/nonb-pipeline-dag.dot"
```

This graph includes the configured motif set and chromosome-level G4Discovery
jobs. If Snakemake attempts to clone `snakemake-wrappers`, network access is
required once to populate the source cache.

### RNA structure pipeline

```bash
cd "$ROOT/secondrna"

XDG_CACHE_HOME="$DAG_TMP" TMPDIR="$DAG_TEMP" \
  "$SNAKEMAKE" \
    --snakefile Snakefile \
    --configfile config/config.default.yaml \
    --cores 1 \
    --dag dot all \
    > "$OUT/secondrna-dag.dot"
```

The default `all` target includes rG4 detector-mode outputs and the ScanFold2
completion target. Prediction-mode output is not part of `all`; request it
explicitly with the `rg4detector_predictions` target. ScanFold2 is a separate
branch from the rG4 tables; it does not feed the rG4 summaries, peak statistics
or visualizations.

### Final `revision-wt` pipeline

```bash
cd "$ROOT/revision-wt"

XDG_CACHE_HOME="$DAG_TMP" TMPDIR="$DAG_TEMP" \
  "$SNAKEMAKE" \
    --snakefile Snakefile \
    --configfile config/config.yaml \
    --cores 1 \
    --dag dot all \
    > "$OUT/revision-wt-dag.dot"
```

This is the authoritative final pipeline DAG. It is large because it expands
the CV folds and classifier branches. The current full graph is approximately
1,500 DOT lines.

### Post-benchmark feature analysis and figures

The benchmark workflow ends by producing the merged benchmark tables. Feature
statistics and publication figures are a separate downstream workflow, with
`workflow/rules/figures.smk` as its entry point and `all_figures` as the target.
That target includes entropy grouping, clustering, statistical tests, SHAP
aggregation, and the configured performance, entropy, UpSet, t-SNE and SHAP
figures. Its inputs include benchmark tables and feature-analysis outputs
produced by earlier stages.

Generate the downstream rule graph from `revision-wt`:

```bash
cd "$ROOT/revision-wt"

XDG_CACHE_HOME="$DAG_TMP" TMPDIR="$DAG_TEMP" \
  "$SNAKEMAKE" \
    --snakefile workflow/rules/figures.smk \
    --dry-run --cores 1 \
    --rulegraph dot all_figures \
    | sed -n '/^digraph snakemake_dag/,$p' \
    > docs/snakevision/post-benchmark-analysis-dag.dot
```

This rule-level graph is intentionally separate from the root `Snakefile` DAG: the figure
workflow consumes the benchmark tables but does not redefine the CV training
and table-aggregation rules. It shows each rule once, without expanding
wildcard-specific jobs such as the five SHAP folds.

## 4. Generate SnakeVision SVGs

SnakeVision reads a Snakemake DOT graph from standard input. Skipping the
terminal `all` node reduces visual clutter because it otherwise receives many
incoming edges.

```bash
for name in te-pipeline nonb-pipeline secondrna revision-wt; do
  "$SNAKEVISION" \
    --skip-rules all \
    --output "$OUT/${name}-dag.svg" \
    < "$OUT/${name}-dag.dot"
done

"$SNAKEVISION" \
  "$ROOT/revision-wt/docs/snakevision/post-benchmark-analysis-dag.dot" \
  --skip-rules all \
  --output "$ROOT/revision-wt/docs/snakevision/post-benchmark-analysis-dag.svg"
```

The post-benchmark graph is passed by filename because that is the reliable
input mode for the installed Snakevision version.

Useful style overrides include:

```bash
"$SNAKEVISION" \
  --skip-rules all \
  --style scale=10.0 node_radius=6.0 edge_stroke_width=2.0 \
  --output "$OUT/revision-wt-dag.svg" \
  < "$OUT/revision-wt-dag.dot"
```

SnakeVision supports interactive JavaScript and animation, but the default
static SVG is preferred for publication and long-term reproducibility.

## 5. Optional Graphviz rendering

Graphviz is not required by SnakeVision. If `dot` is installed, it can create
a conventional PDF or SVG directly from each DOT file:

```bash
command -v dot

dot -Tpdf "$OUT/revision-wt-dag.dot" \
  -o "$OUT/revision-wt-dag.graphviz.pdf"

dot -Tsvg "$OUT/revision-wt-dag.dot" \
  -o "$OUT/revision-wt-dag.graphviz.svg"
```

Graphviz output is useful for debugging connectivity. SnakeVision output is
usually more suitable for inspection because it uses curved edges and a more
compact layout.

## 6. Constructing the publication overview SVG

The publication-facing SVG is intentionally not a direct concatenation of
the four full DAGs. The full graphs contain too many repeated jobs for a
readable supplementary overview. The overview is a manually curated,
stage-level composition based on the generated DAGs.

The reproducible generator is:

```text
revision-wt/docs/build_supplementary_pipeline_dag.py
```

Run it from `revision-wt`:

```bash
cd "$ROOT/revision-wt"
python docs/build_supplementary_pipeline_dag.py \
  --output docs/supplementary_pipeline_dag.svg
```

The script has no third-party dependencies. It defines the colors, panels,
stage nodes and edge types directly, so the SVG can be regenerated from a
clean checkout without depending on a previously generated SVG.

### Customizing the text without editing Python

All visible text and the accessible SVG title/description are listed in
`DEFAULT_TEXT`. Export this catalog to get an editable UTF-8 JSON file:

```bash
python docs/build_supplementary_pipeline_dag.py --dump-text /tmp/dag-text.json
python docs/build_supplementary_pipeline_dag.py --text /tmp/dag-text.json -o /tmp/custom-dag.svg
```

Edit the JSON **values**, preserving the keys. You can keep the full catalog
or only the entries you want to override, for example:

```json
{
  "svg.title": "My workflow overview",
  "te.repeat.title": "Run RepeatMasker\nAnnotate repeats",
  "te.repeat.detail": "",
  "revision.entropy.title": "Uncertainty",
  "svg.accessible_title": "My supplementary workflow diagram"
}
```

Omitted entries retain their defaults. An empty string hides that text.
Unknown keys, non-string values, invalid JSON and unreadable files produce
command-line errors before an SVG is written. JSON does not support comments
or trailing commas. Use `--help` for the script's embedded editing guide.
Combining `--text` and `--dump-text` exports the merged catalog without
rendering an SVG; `--output` is unused in that mode.

| Key pattern | Controls |
|---|---|
| `svg.title`, `svg.subtitle` | Visible page heading and subtitle |
| `svg.accessible_title`, `svg.desc` | SVG metadata for accessibility |
| `<panel>.title`, `<panel>.note` | Panel heading and explanatory note |
| `<panel>.<node>.title`, `<panel>.<node>.detail` | Box labels and secondary text, where present |
| `handoff.dependencies`, `handoff.optional`, `handoff.legend.*` | Arrow legend text |
| `handoff.source.A` through `handoff.source.D` | Source-panel references above the repeated inputs in E |

Panel prefixes are `te`, `nbd`, `rna`, `revision` and `handoff`. Node keys
are stable identifiers, independent of displayed labels: renaming “Entropy”
does not alter its connections. Plain Unicode text, including `&` and `<`, is
escaped automatically; markup is displayed literally.

`\n` in JSON becomes a line break in node labels/details and panel notes.
Other text is single-line. Text does not auto-wrap or shrink. Node title and
detail lines occupy 20 and 18 SVG units respectively and the complete block
is vertically centered. Increase box height for extra lines and width for
longer lines; panel notes need sufficient space above the first row of boxes.
An empty detail removes its line from the centered block. Inspect customized
SVGs in a browser before publication. The optional `pango-view` check measures
default node text at the renderer's font sizes; customized text still needs inspection.

From Python, use partial overrides directly (with `docs` on the import path):

```python
from build_supplementary_pipeline_dag import svg

custom_svg = svg({"svg.title": "My workflow overview"})
```

`svg()` returns a string and performs no file I/O. Each call merges a fresh
catalog; it does not change defaults for subsequent renders. Individual panel
renderers receive a complete catalog, available from `resolve_text({...})`.

### Editing the overview layout

The declarative specifications `PANELS`, `NODES` and `EDGES` define the
diagram. Each `NodeSpec` has a panel, workflow color, kind and grid `(row,
col)`; it contains no pixel coordinates. `GridLayout` turns that grid into
boxes, and `render_edges()` derives all edge endpoints from their anchors.
Edit `DEFAULT_TEXT` or use JSON overrides for labels. Use `NodeSpec.width`
only for a necessary node-specific width; adjust a panel's `GridLayout` for
shared row, column and width changes. `PANEL_X`, `PANEL_Y`, `W` and `H` set
panel and canvas placement. `COLORS` and `STYLES` control shared appearance.
Labels support explicit `\n` line breaks; increase box height before adding
lines.

Edges declare source and destination ids plus optional anchors. `top`,
`bottom`, `left` and `right` anchors land at box midpoints; `orthogonal_route()`
creates straight routes or routes with up to two bends. Named edge groups in `render_edges()`
draw shared fan-in/fan-out buses. Edges render before boxes, so line ends do
not obscure labels. `dashed` is reserved for optional dependencies; the
current figure has no optional workflow branch, so the optional-branch legend
entry is hidden. Its text key remains available for compatibility. The legend
also explains the pink analysis steps and teal files in E.

Panel E uses an integer grid with five input files, three centered analysis
steps, and four evenly spaced figure files. All three rows share the center
axis of the merged tables and final output. The entropy figure's incoming
stem lands off-center within its fan-in bus to accommodate the four-file row;
the bus still spans the outermost sources. Side-anchor routes carry the
t-SNE, SHAP and benchmark branches clear of that bus and of node borders.

The dependency and format choices are grounded in these repository sources:

- `workflow/rules/figures.smk`: the four figure groups declare PDF outputs;
  performance figures consume benchmark class tables.
- `workflow/scripts/plot_upset_figure.py`: `load_tables()` supplies the binary
  benchmark table, although `upset_figure` does not declare it as a rule input.
  The overview therefore shows the benchmark-to-performance/UpSet dependency
  at the data-flow level.
- `workflow/rules/shap.smk`: clustered SHAP consumes feature-cluster files and
  aggregates SHAP results before plotting. The clustering-to-SHAP arrow
  summarizes that chain; it does not enumerate every SHAP input.
- `config/feature_analysis_config.yaml`: the ScanFold2 handoff is
  `scanfold_final_partners_stats.tsv`, matching the `.tsv` label in D and E.

This remains a stage-level overview, not an exhaustive rule dependency graph.
The merged tables are produced files, and execution details such as folds are
omitted from node copy. Annotation/assembly context remains in the global caption.

Regenerate the SVG and run the dependency-free geometry check:

```bash
python docs/build_supplementary_pipeline_dag.py
python -m unittest discover -s docs -p 'test_supplementary_pipeline_dag.py'
```

The check validates XML, bounds, row spacing, dependencies, source references,
conditional legends, perpendicular arrow endpoints, and routes crossing box
interiors or running along borders. With `pango-view` installed, it also checks
node text width and height. Visually inspect the SVG after editing. `before.png`,
`after.png` and `before_after_comparison.svg`/`.png` record the visual comparison;
the comparison preserves each image's aspect ratio.

When updating it:

1. Regenerate all four DOT files.
2. Inspect the SnakeVision SVGs for changed rule names, branches or target
   expansions.
3. Update the panel data and layout helpers in
   `build_supplementary_pipeline_dag.py` if the workflow structure changed.
4. Regenerate the SVG with the script.
5. Keep upstream workflows as independent panels.
6. Keep the heavy cross-workflow arrows only in Panel E.
7. Validate the resulting SVG.

```bash
xmllint --noout \
  "$ROOT/revision-wt/docs/supplementary_pipeline_dag.svg"
```

The manual grouping should preserve these distinctions:

- solid arrows: dependencies, including feature-file handoffs into Panel E;
- dashed arrows: optional branches when present;
- grouped nodes: repeated folds, tools, chromosomes or chunks.

## 7. Checking the generated graphs

Basic checks:

```bash
for f in "$OUT"/*.dot; do
  echo "=== $f"
  wc -l "$f"
  grep -c 'label =' "$f"
done
```

Check representative rule names:

```bash
grep 'label =' "$OUT/te-pipeline-dag.dot"
grep 'label =' "$OUT/nonb-pipeline-dag.dot"
grep 'label =' "$OUT/secondrna-dag.dot"
grep 'label =' "$OUT/revision-wt-dag.dot" | head -40
```

Check that the expected final workflow stages occur in the `revision-wt`
graph:

```bash
grep -E 'training|testing|process|embedding|entropy|cluster|shap|figure' \
  "$OUT/revision-wt-dag.dot"
```

## 8. Troubleshooting

### `snakemake: command not found`

Use the absolute environment binary:

```bash
/home/dgarcia/miniforge3/envs/snakenew/bin/snakemake --version
```

### Read-only source-cache error

Set writable cache and temporary directories:

```bash
export XDG_CACHE_HOME=/tmp/lnc-snakemake-cache
export TMPDIR=/tmp/lnc-tmp
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
```

### Wrapper clone or network error

The workflow may need the pinned `snakemake-wrappers` repository while
parsing rules. Run the Non-B command with network access once, then rerun the
same command after the cache has been populated.

### Missing input files

`--dag` constructs the graph but still evaluates configuration, checkpoints,
and input functions. It does not execute jobs. Missing biological input data
or inconsistent configuration can therefore prevent DAG construction.

Use the configured toy target or a test configuration when validating syntax
without the full reference dataset.

### Graph is too dense

Use SnakeVision's rule-skipping option:

```bash
"$SNAKEVISION" \
  --skip-rules all some_large_aggregation_rule \
  --output "$OUT/revision-wt-dag-trimmed.svg" \
  < "$OUT/revision-wt-dag.dot"
```

For the supplementary overview, group repeated jobs manually instead of
removing biologically meaningful branches.

## 9. Reproducibility record

For a release, preserve alongside the final SVG:

- the four DOT files;
- the four SnakeVision SVGs;
- the exact config files used;
- Snakemake and SnakeVision versions;
- the date of generation;
- the commit or archive identifier of the workflow source.

The DOT files are the most important machine-readable record. The curated
overview SVG is the communication layer built from those graphs.
