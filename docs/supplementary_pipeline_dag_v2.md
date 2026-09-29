# Dependency-explicit comparison proposal

Open `supplementary_pipeline_dag_comparison.html` for side-by-side views, or
`supplementary_pipeline_dag_v2_comparison.png` for a static comparison of E.
The original generator and SVG remain unchanged.

V2 reuses panels A–D. E uses repeated, named input references instead of long
crossing arrows. The square input-list enclosures are collections of separate
artifacts, not invented merged files. The directed dependency graph remains
explicit in `STAGES`: each output name can be referenced by a later stage.

The proposed abstraction groups uncertainty metrics with group assignment,
statistical testing with entropy plotting, embeddings with t-SNE plotting,
and RF training/SHAP computation with aggregation and plotting. The t-SNE
subtitle distinguishes plot annotation from embedding computation. Performance
and UpSet share a benchmark-only branch. The output enclosure is a visual
collection, not a processing rule or produced file.

Tradeoff: E is taller and more explicit, with dependencies read through repeated
names rather than traced through one connected drawing. Publication artifacts
shown are representative; this is not an exhaustive file inventory. UpSet's
benchmark dependency reflects its script's data loading even though its rule
does not declare those inputs. Workflow rules were not edited.

Regenerate from `revision-wt` (acquire `docs/.dag_edit.lock` first):

```sh
python docs/build_supplementary_pipeline_dag_v2.py --comparison docs/supplementary_pipeline_dag_v2_comparison.svg
rsvg-convert docs/supplementary_pipeline_dag_v2.svg -o docs/supplementary_pipeline_dag_v2.png
rsvg-convert docs/supplementary_pipeline_dag.svg -o docs/supplementary_pipeline_dag_v1.png
rsvg-convert docs/supplementary_pipeline_dag_v2_comparison.svg -o docs/supplementary_pipeline_dag_v2_comparison.png
python -m unittest discover -s docs -p 'test_supplementary_pipeline_dag_v2.py'
```

The new builder uses the standard library and the existing renderer. PNG export
uses the installed `rsvg-convert`. The comparison HTML works locally with no
network or third-party dependencies.
