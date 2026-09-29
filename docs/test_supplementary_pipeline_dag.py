"""Run with: python -m unittest discover -s docs -p 'test_supplementary_pipeline_dag.py'."""

import re
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import build_supplementary_pipeline_dag as dag


class DiagramTest(unittest.TestCase):
    def test_panel_order_and_layout(self):
        for panel in dag.PANELS:
            panel_nodes = tuple(node for node in dag.NODES if node.panel == panel.id)
            boxes = dag.layout(panel, panel_nodes)
            width, height = dag.panel_size(panel, boxes)
            x, y = dag.panel_origin(panel, width)
            self.assertTrue(0 <= x <= x + width <= dag.W)
            self.assertTrue(0 <= y <= y + height <= dag.H)
            for box in boxes.values():
                self.assertGreaterEqual(box.x, dag.GRID[panel.id].padding)
                self.assertLessEqual(
                    box.x + box.width, width - dag.GRID[panel.id].padding
                )
                self.assertGreaterEqual(box.y, 0)
                self.assertLessEqual(
                    box.y + box.height, height - dag.GRID[panel.id].padding
                )
            for row in {node.row for node in panel_nodes} if panel.id != "A" else set():
                centers = {
                    boxes[node.id].y + boxes[node.id].height / 2
                    for node in panel_nodes
                    if node.row == row
                }
                self.assertEqual(len(centers), 1)
            for col in {node.col for node in panel_nodes} if panel.id != "E" else set():
                centers = {
                    boxes[node.id].x + boxes[node.id].width / 2
                    for node in panel_nodes
                    if node.col == col
                }
                self.assertEqual(len(centers), 1)
            box_list = list(boxes.values())
            for index, first in enumerate(box_list):
                for second in box_list[index + 1 :]:
                    overlaps = (
                        first.x < second.x + second.width
                        and second.x < first.x + first.width
                        and first.y < second.y + second.height
                        and second.y < first.y + first.height
                    )
                    self.assertFalse(
                        overlaps, f"{first.node.id} overlaps {second.node.id}"
                    )
        panel_a = next(panel for panel in dag.PANELS if panel.id == "A")
        boxes_a = dag.layout(
            panel_a, tuple(node for node in dag.NODES if node.panel == "A")
        )
        revision_flow = (
            "a-input",
            "a-prepare",
            "a-train",
            "a-infer",
            "a-process",
            "a-output",
        )
        self.assertGreater(boxes_a["a-input"].width, boxes_a["a-train"].width)
        self.assertGreater(boxes_a["a-prepare"].width, boxes_a["a-output"].width)
        revision_gaps = [
            boxes_a[target].x - (boxes_a[source].x + boxes_a[source].width)
            for source, target in zip(revision_flow, revision_flow[1:])
        ]
        self.assertEqual(len(set(revision_gaps)), 1)
        self.assertGreater(revision_gaps[0], 0)
        self.assertLess(
            dag.panel_size(
                panel_a,
                dag.layout(panel_a, tuple(n for n in dag.NODES if n.panel == "A")),
            )[1],
            300,
        )
        panel_d = next(panel for panel in dag.PANELS if panel.id == "D")
        _, panel_d_height = dag.panel_size(
            panel_d, dag.layout(panel_d, tuple(n for n in dag.NODES if n.panel == "D"))
        )
        self.assertLessEqual(dag.PANEL_Y["D"] + panel_d_height, dag.PANEL_Y["E"])
        panel_e = next(panel for panel in dag.PANELS if panel.id == "E")
        boxes = dag.layout(
            panel_e, tuple(node for node in dag.NODES if node.panel == "E")
        )
        files = [box for box in boxes.values() if box.node.row == 0]
        self.assertEqual(len(files), 6)
        self.assertEqual(len({box.y for box in files}), 1)
        self.assertLess(boxes_a["a-output"].y, boxes_a["a-process"].y)
        self.assertGreaterEqual(
            boxes_a["a-model-features"].y, boxes_a["a-process"].y + 72
        )
        for name in ("e-clustering", "e-shap", "e-statistics"):
            box = boxes[name]
            self.assertTrue(167 < box.x < box.x + box.width < 1037)
            self.assertTrue(362 < box.y < box.y + box.height < 494)
        self.assertGreater(boxes["e-entropy"].x, 1037)
        self.assertGreater(boxes["e-output"].y, boxes["e-entropy"].y + 72)
        panel_b = next(panel for panel in dag.PANELS if panel.id == "B")
        boxes_b = dag.layout(
            panel_b, tuple(node for node in dag.NODES if node.panel == "B")
        )
        self.assertEqual(len({box.anchor("top")[0] for box in boxes_b.values()}), 1)
        te_flow = ("b-input", "b-prepare", "b-repeat", "b-merge", "b-output")
        self.assertEqual(
            [boxes_b[name].y for name in te_flow],
            sorted(boxes_b[name].y for name in te_flow),
        )
        te_gaps = [
            boxes_b[target].y - (boxes_b[source].y + boxes_b[source].height)
            for source, target in zip(te_flow, te_flow[1:])
        ]
        self.assertEqual(len(set(te_gaps)), 1)
        te_edges = {(edge.src, edge.dst): edge for edge in dag.EDGES}
        for source, target in zip(te_flow, te_flow[1:]):
            edge = te_edges[source, target]
            route = dag.orthogonal_route(
                boxes_b[source], boxes_b[target], edge.src_anchor, edge.dst_anchor
            )
            self.assertEqual(len(route), 2)
            self.assertEqual(route[0][0], route[1][0])

        panel_c = next(panel for panel in dag.PANELS if panel.id == "C")
        boxes_c = dag.layout(
            panel_c, tuple(node for node in dag.NODES if node.panel == "C")
        )
        nonb_flow = ("c-input", "c-prepare", "c-motifs", "c-extended", "c-output")
        self.assertEqual(len({boxes_c[name].anchor("top")[0] for name in nonb_flow}), 1)
        self.assertEqual(
            [boxes_c[name].y for name in nonb_flow],
            sorted(boxes_c[name].y for name in nonb_flow),
        )
        nonb_gaps = [
            boxes_c[target].y - (boxes_c[source].y + boxes_c[source].height)
            for source, target in zip(nonb_flow, nonb_flow[1:])
        ]
        self.assertEqual(len(set(nonb_gaps)), 1)
        nonb_edges = {(edge.src, edge.dst): edge for edge in dag.EDGES}
        for source, target in zip(nonb_flow, nonb_flow[1:]):
            edge = nonb_edges[source, target]
            route = dag.orthogonal_route(
                boxes_c[source], boxes_c[target], edge.src_anchor, edge.dst_anchor
            )
            self.assertEqual(len(route), 2)
            self.assertEqual(route[0][0], route[1][0])

    def test_specs_are_coordinate_free_and_edges_resolve(self):
        node_ids = {node.id for node in dag.NODES}
        self.assertEqual(len(node_ids), len(dag.NODES))
        self.assertTrue(
            all(node.kind in {"step", "file", "produced_file"} for node in dag.NODES)
        )
        self.assertTrue(all(node.workflow in "ABCDE" for node in dag.NODES))
        self.assertTrue(
            all(edge.src in node_ids and edge.dst in node_ids for edge in dag.EDGES)
        )
        for node in dag.NODES:
            self.assertIsInstance(node.row, int)
            self.assertIsInstance(node.col, int)

    def test_corrected_dependencies(self):
        dependencies = {(edge.src, edge.dst, edge.style) for edge in dag.EDGES}
        self.assertNotIn("c-basic", {node.id for node in dag.NODES})
        self.assertNotIn(("c-motifs", "c-basic", "solid"), dependencies)
        self.assertIn(("d-input", "d-scanfold", "solid"), dependencies)
        self.assertIn(("d-scanfold", "d-scanfold-merge", "solid"), dependencies)
        self.assertIn(("d-scanfold-merge", "d-scanfold-output", "solid"), dependencies)
        self.assertNotIn(("d-scanfold", "d-rg4", "solid"), dependencies)
        self.assertIn(("d-merge", "d-peaks-output", "solid"), dependencies)
        self.assertNotIn(("d-rg4", "d-summary", "solid"), dependencies)
        self.assertNotIn("d-rg4", {node.id for node in dag.NODES})
        self.assertNotIn("d-count", {node.id for node in dag.NODES})
        self.assertNotIn("d-summary", {node.id for node in dag.NODES})
        self.assertNotIn("d-summary-output", {node.id for node in dag.NODES})
        self.assertIn(("a-process", "a-output", "solid"), dependencies)
        self.assertIn(("a-process", "a-model-features", "solid"), dependencies)
        self.assertIn(("e-benchmark", "e-entropy", "solid"), dependencies)
        self.assertIn(("e-benchmark", "e-other-figures", "solid"), dependencies)
        for name in ("e-te", "e-nbd", "e-rg4", "e-scanfold", "e-model-features"):
            self.assertIn((name, "e-clustering", "solid"), dependencies)
        for name in (
            "e-clustering",
            "e-shap",
            "e-statistics",
            "e-entropy",
            "e-other-figures",
        ):
            self.assertIn((name, "e-output", "solid"), dependencies)

    def test_rna_panel_matches_default_targets(self):
        rna_nodes = {node.id for node in dag.NODES if node.panel == "D"}
        self.assertEqual(
            rna_nodes,
            {
                "d-input",
                "d-detect",
                "d-scanfold",
                "d-merge",
                "d-scanfold-merge",
                "d-scanfold-output",
                "d-peaks-output",
            },
        )
        d_edges = tuple(edge for edge in dag.EDGES if edge.src.startswith("d-"))
        dependencies = {(edge.src, edge.dst) for edge in d_edges}
        self.assertEqual(
            {(edge.src, edge.dst) for edge in d_edges if edge.group == "rna-input"},
            {("d-input", "d-detect"), ("d-input", "d-scanfold")},
        )
        self.assertIn(("d-detect", "d-merge"), dependencies)
        self.assertIn(("d-merge", "d-peaks-output"), dependencies)
        self.assertIn(("d-scanfold", "d-scanfold-merge"), dependencies)
        self.assertIn(("d-scanfold-merge", "d-scanfold-output"), dependencies)
        self.assertEqual(
            dag.DEFAULT_TEXT.keys() & {"rna.summary.title", "rna.summary.detail"},
            set(),
        )
        panel_d = next(panel for panel in dag.PANELS if panel.id == "D")
        boxes = dag.layout(
            panel_d, tuple(node for node in dag.NODES if node.panel == "D")
        )
        scanfold_path = ("d-scanfold", "d-scanfold-merge", "d-scanfold-output")
        gap = dag.GRID["B"].row_gutter
        self.assertEqual(
            boxes["d-scanfold"].y - boxes["d-input"].anchor("bottom")[1], 3 * gap
        )
        rg4_path = ("d-detect", "d-merge", "d-peaks-output")
        for path in (scanfold_path, rg4_path):
            for source, target in zip(path, path[1:]):
                self.assertEqual(
                    boxes[target].y - boxes[source].anchor("bottom")[1], gap
                )
            self.assertEqual(
                len({boxes[node_id].anchor("top")[0] for node_id in path}), 1
            )
            self.assertEqual(
                [boxes[node_id].y for node_id in path],
                sorted(boxes[node_id].y for node_id in path),
            )

    def test_legend_is_complete_and_aligned(self):
        root = ET.fromstring(dag.svg())
        baselines = {
            node.attrib["y"]
            for node in root.iter()
            if node.attrib.get("class") == "legend"
        }
        self.assertEqual(baselines, {str(dag.legend_baseline() + 2)})
        panel_e = next(panel for panel in dag.PANELS if panel.id == "E")
        boxes = dag.layout(
            panel_e, tuple(node for node in dag.NODES if node.panel == "E")
        )
        panel_width, _ = dag.panel_size(panel_e, boxes)
        self.assertLess(1140, panel_width)
        _, panel_height = dag.panel_size(panel_e, boxes)
        panel_bottom = dag.PANEL_Y["E"] + panel_height
        self.assertEqual(dag.legend_baseline() - panel_bottom, 38)
        self.assertGreaterEqual(dag.H, dag.legend_baseline() + 10)
        legend_groups = [
            child
            for child in root
            if child.tag.endswith("}g")
            and any(item.attrib.get("class") == "legend" for item in child.iter())
        ]
        self.assertEqual(len(legend_groups), 1)
        self.assertEqual(
            legend_groups[0].attrib["transform"],
            f"translate({(dag.W - dag.LEGEND_WIDTH) // 2} 0)",
        )

    def test_legend_only_shows_used_edges(self):
        legend = dag.render_legend(dag.DEFAULT_TEXT, dag.legend_baseline())
        self.assertNotIn("Optional branch", legend)
        labels = [
            node.text for node in ET.fromstring("<g>" + legend + "</g>").iter("text")
        ]
        self.assertEqual(
            labels,
            [
                "Step",
                "Input file",
                "Output file",
                "Dependency flow",
                "Color = workflow",
            ],
        )

    @unittest.skipUnless(shutil.which("pango-view"), "pango-view is optional")
    def test_node_text_fits_measured_font_width(self):
        text = dag.DEFAULT_TEXT
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "measure.png"
            for panel in dag.PANELS:
                nodes = tuple(node for node in dag.NODES if node.panel == panel.id)
                boxes = dag.layout(panel, nodes)
                for node in nodes:
                    box = boxes[node.id]
                    lines = [
                        (line, "Inter Bold 15")
                        for line in text[node.label_key].splitlines()
                    ]
                    line_height = 20 * len(lines)
                    if node.sublabel_key:
                        font = "monospace 14" if node.kind != "step" else "Inter 14"
                        line_height += 18 * len(text[node.sublabel_key].splitlines())
                        lines.extend(
                            (line, font)
                            for line in text[node.sublabel_key].splitlines()
                        )
                    self.assertLessEqual(line_height, box.height - 12, node.id)
                    for value, font in lines:
                        command = [
                            "pango-view",
                            "--pixels",
                            "--margin=0",
                            f"--font={font}",
                            f"--text={value}",
                            "--no-display",
                            f"--output={output}",
                        ]
                        for attempt in range(2):
                            result = subprocess.run(
                                command,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                            )
                            if result.returncode == 0:
                                break
                        else:
                            self.fail(
                                f"pango-view failed twice for {node.id}: "
                                f"{result.stderr.decode(errors='replace')}"
                            )
                        image = output.read_bytes()
                        measured_width = struct.unpack(">I", image[16:20])[0]
                        self.assertLessEqual(
                            measured_width,
                            box.width - 12,
                            f"{node.id} text too wide: {value!r} ({measured_width}px/{box.width}px)",
                        )

    def test_text_customization(self):
        before = dag.svg()
        overrides = {
            key: f"Custom {i} & <text>" for i, key in enumerate(dag.DEFAULT_TEXT)
        }
        root = ET.fromstring(dag.svg(overrides))
        rendered = list(root.itertext())
        # Retired text keys remain accepted for existing JSON overrides.
        active_keys = {node.label_key for node in dag.NODES} | {
            node.sublabel_key for node in dag.NODES
        }
        active_keys |= {panel.title_key for panel in dag.PANELS} | {
            panel.note_key for panel in dag.PANELS
        }
        for key in active_keys - {None}:
            self.assertIn(overrides[key], rendered)
        self.assertEqual(dag.svg(), before)
        custom = ET.fromstring(
            dag.svg({"te.repeat.title": "First\nSecond", "te.repeat.detail": ""})
        )
        self.assertIn("First", list(custom.itertext()))
        self.assertIn("Second", list(custom.itertext()))
        for invalid in ([], {"typo": "label"}, {"svg.title": 123}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                dag.svg(invalid)

    def test_routes_and_bounds(self):
        document = dag.svg()
        self.assertEqual(ET.fromstring(document).tag, "{http://www.w3.org/2000/svg}svg")
        self.assertEqual(document, dag.svg())
        self.assertEqual(len(dag.PANELS), 5)
        self.assertNotIn('class="edge-dash"', document)
        self.assertEqual(document.count('id="arrow"'), 1)
        for panel in dag.PANELS:
            panel_nodes = tuple(node for node in dag.NODES if node.panel == panel.id)
            boxes = dag.layout(panel, panel_nodes)
            panel_edges = tuple(edge for edge in dag.EDGES if edge.src in boxes)
            width, height = dag.panel_size(panel, boxes)
            for edge in panel_edges:
                start = boxes[edge.src].anchor(edge.src_anchor)
                end = boxes[edge.dst].anchor(edge.dst_anchor)
                dag.assert_endpoint_on_box(end, boxes[edge.dst])
                if edge.group:
                    continue
                points = dag.orthogonal_route(
                    boxes[edge.src], boxes[edge.dst], edge.src_anchor, edge.dst_anchor
                )
                self.assertEqual(points[0], start)
                self.assertEqual(points[-1], end)
                self.assertTrue(
                    all(
                        first[0] == second[0] or first[1] == second[1]
                        for first, second in zip(points, points[1:])
                    )
                )
                self.assertLessEqual(len(points) - 2, 2, f"{edge.src} -> {edge.dst}")
                # Arrival and departure must be perpendicular to their box edges.
                for anchor, first, second in (
                    (edge.src_anchor, points[0], points[1]),
                    (edge.dst_anchor, points[-1], points[-2]),
                ):
                    axis = 0 if anchor in {"top", "bottom"} else 1
                    self.assertEqual(first[axis], second[axis], edge)
            for path in dag.render_edges(panel_edges, boxes):
                points = [
                    tuple(map(int, pair.split()))
                    for pair in re.findall(r"[ML](\d+ \d+)", path)
                ]
                for first, second in zip(points, points[1:]):
                    self.assertTrue(
                        first[0] == second[0] or first[1] == second[1],
                        f"non-orthogonal route: {path}",
                    )
                    # Reject both interior crossings and lines running along borders.
                    for box in boxes.values():
                        if first[1] == second[1]:
                            overlap = box.y <= first[1] <= box.y + box.height and max(
                                min(first[0], second[0]), box.x
                            ) < min(max(first[0], second[0]), box.x + box.width)
                        else:
                            overlap = box.x <= first[0] <= box.x + box.width and max(
                                min(first[1], second[1]), box.y
                            ) < min(max(first[1], second[1]), box.y + box.height)
                        self.assertFalse(overlap, f"{path} crosses {box.node.id}")
                    for x, y in (first, second):
                        self.assertTrue(0 <= x <= width and 0 <= y <= height)
                if "marker-end:none" not in path:
                    self.assertTrue(
                        any(
                            points[-1] == box.anchor(anchor)
                            for box in boxes.values()
                            for anchor in ("top", "bottom", "left", "right")
                        ),
                        path,
                    )


if __name__ == "__main__":
    unittest.main()
