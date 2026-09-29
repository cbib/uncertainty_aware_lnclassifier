"""Run: python -m unittest discover -s docs -p 'test_supplementary_pipeline_dag_v3.py'."""

import unittest
from xml.etree import ElementTree as ET

import build_supplementary_pipeline_dag_v3 as dag


class CompactDiagramTest(unittest.TestCase):
    def test_preserved_dependencies_and_upstream_panels(self):
        document = dag.svg()
        ET.fromstring(document)
        self.assertEqual(dag.STAGES, dag.v2.STAGES)
        self.assertLess(dag.INPUT_WIDTH, dag.v2.WIDTHS[0] / 2)
        for panel in dag.base.PANELS[:-1]:
            self.assertIn(dag.base.render_panel(panel, dag.base.DEFAULT_TEXT), document)
        grid = dag.boxes()
        for stage in dag.STAGES:
            inputs = [grid[f"{stage.id}-input-{ref}"] for ref in stage.inputs]
            self.assertTrue(all(box.node.kind == "produced_file" for box in inputs))
            paths = dag.routes(stage, grid)
            self.assertEqual(
                [path[0][0] for path in paths[:-2]],
                [box.anchor("right") for box in inputs],
            )
            self.assertEqual(paths[-2][0][-1], grid[f"{stage.id}-step"].anchor("left"))
            self.assertEqual(
                paths[-1][0][-1], grid[f"{stage.id}-output"].anchor("left")
            )

    def test_bounds_overlaps_and_routes(self):
        grid = dag.boxes()
        boxes = list(grid.values())
        for index, box in enumerate(boxes):
            self.assertGreaterEqual(box.x, dag.PADDING)
            self.assertLessEqual(box.x + box.width, dag.PANEL_WIDTH - dag.PADDING)
            self.assertLessEqual(box.y + box.height, dag.PANEL_HEIGHT - dag.PADDING)
            for other in boxes[index + 1 :]:
                self.assertFalse(
                    box.x < other.x + other.width
                    and other.x < box.x + box.width
                    and box.y < other.y + other.height
                    and other.y < box.y + box.height
                )
        for stage in dag.STAGES:
            for points, arrow, destination in dag.routes(stage, grid):
                if arrow:
                    dag.base.assert_endpoint_on_box(points[-1], destination)
                for start, end in zip(points, points[1:]):
                    self.assertTrue(start[0] == end[0] or start[1] == end[1])
                    for box in boxes:
                        if start[1] == end[1]:
                            crosses = box.y <= start[1] <= box.y + box.height and max(
                                min(start[0], end[0]), box.x
                            ) < min(max(start[0], end[0]), box.x + box.width)
                        else:
                            crosses = box.x <= start[0] <= box.x + box.width and max(
                                min(start[1], end[1]), box.y
                            ) < min(max(start[1], end[1]), box.y + box.height)
                        self.assertFalse(crosses, (stage.id, box.node.id, start, end))


if __name__ == "__main__":
    unittest.main()
