"""Run: python -m unittest discover -s docs -p 'test_supplementary_pipeline_dag_v2.py'."""

import unittest
from xml.etree import ElementTree as ET

import build_supplementary_pipeline_dag as base
import build_supplementary_pipeline_dag_v2 as proposal


class ProposalTest(unittest.TestCase):
    def test_scientific_dependencies(self):
        stages = {stage.id: stage for stage in proposal.STAGES}
        self.assertEqual(stages["uncertainty"].inputs, ("benchmark",))
        self.assertEqual(stages["performance"].inputs, ("benchmark",))
        self.assertTrue(
            {"uncertainty", "clusters", "supplementary", "benchmark"}
            <= set(stages["entropy_figures"].inputs)
        )
        self.assertTrue(
            {"clusters", "splits", "supplementary", "benchmark"}
            <= set(stages["shap"].inputs)
        )
        self.assertIn("compute_embeddings", stages["tsne"].rules)
        self.assertIn("uncertainty", stages["tsne"].inputs)
        self.assertEqual(
            stages["shap"].rules, ("shap_fold", "shap_aggregate", "shap_figures")
        )
        available = {"features", "benchmark", "benchmark_index", "splits"}
        for stage in proposal.STAGES:
            self.assertTrue(set(stage.inputs) <= available, stage.id)
            available.add(stage.id)

    def test_preserved_upstream_and_valid_document(self):
        before = base.svg()
        after = proposal.svg()
        root = ET.fromstring(after)
        for panel in base.PANELS[:-1]:
            self.assertIn(base.render_panel(panel, base.DEFAULT_TEXT), after)
        self.assertEqual(base.svg(), before)
        self.assertEqual(root.attrib["height"], str(proposal.HEIGHT))
        self.assertNotIn(
            base.DEFAULT_TEXT["revision.output.title"], "".join(root.itertext())
        )
        self.assertEqual(after.count('id="arrow"'), 1)

    def test_bounds_spacing_and_endpoints(self):
        boxes = proposal.boxes()
        for stage in proposal.STAGES:
            row = [boxes[f"{stage.id}-{role}"] for role in ("inputs", "step", "output")]
            self.assertEqual(len({box.anchor("left")[1] for box in row}), 1)
            for source, target in zip(row, row[1:]):
                self.assertLess(source.x + source.width, target.x)
                base.assert_endpoint_on_box(source.anchor("right"), source)
                base.assert_endpoint_on_box(target.anchor("left"), target)
            for box in row:
                self.assertLessEqual(
                    box.x + box.width, proposal.PANEL_WIDTH - proposal.PADDING
                )
                self.assertLessEqual(
                    box.y + box.height, proposal.PANEL_HEIGHT - proposal.PADDING
                )
        ordered = list(boxes.values())
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                self.assertFalse(
                    first.x < second.x + second.width
                    and second.x < first.x + first.width
                    and first.y < second.y + second.height
                    and second.y < first.y + first.height
                )


if __name__ == "__main__":
    unittest.main()
