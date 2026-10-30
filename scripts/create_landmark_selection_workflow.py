"""Generate the landmark-first ComfyUI canvas without bundled place choices."""

from __future__ import annotations

import json
from pathlib import Path

from create_geoblock_workflow import node


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/landmark_selection_comfy.json"


def main() -> None:
    nodes = [
        node(1, "GeoBlockSearchLandmarks", "1 · Search for your landmark",
             (80, 160), (380, 175), [],
             [("landmark matches", "GEOBLOCK_CANDIDATES"), ("search report", "STRING")], [""]),
        node(2, "GeoBlockChooseLandmark", "2 · Choose a result and coverage",
             (570, 160), (390, 200),
             [("landmark_matches", "GEOBLOCK_CANDIDATES")],
             [("chosen place", "GEOBLOCK_PLACE"), ("selection report", "STRING")], [1, 1.0]),
        node(3, "GeoBlockShowReport", "Review search matches", (570, -100), (420, 160),
             [("report", "STRING")], [], []),
        node(4, "GeoBlockShowReport", "Review chosen landmark", (1080, 160), (430, 170),
             [("report", "STRING")], [], []),
    ]
    links = [[1, 1, 0, 2, 0, "GEOBLOCK_CANDIDATES"],
             [2, 1, 1, 3, 0, "STRING"],
             [3, 2, 1, 4, 0, "STRING"]]
    by_id = {item["id"]: item for item in nodes}
    for link_id, source, source_slot, target, target_slot, _ in links:
        output = by_id[source]["outputs"][source_slot]
        output["links"] = (output["links"] or []) + [link_id]
        by_id[target]["inputs"][target_slot]["link"] = link_id
    graph = {
        "id": "geoblock-landmark-selection", "revision": 0,
        "last_node_id": 4, "last_link_id": 3,
        "nodes": nodes, "links": links, "groups": [],
        "definitions": {"subgraphs": []}, "config": {},
        "extra": {"ds": {"scale": 0.9, "offset": [20, 40]}}, "version": 0.4,
    }
    DESTINATION.write_text(json.dumps(graph, indent=2) + "\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
