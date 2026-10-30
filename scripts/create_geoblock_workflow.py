"""Write a loadable ComfyUI canvas for the GeoBlock preparation prototype."""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/geoblock_prototype_comfy.json"
WOODSIDE_DESTINATION = ROOT / "examples/woodside_library_comfy.json"


def node(node_id, kind, title, pos, size, inputs, outputs, widgets):
    return {
        "id": node_id, "type": kind, "title": title, "pos": list(pos), "size": list(size),
        "flags": {}, "order": node_id - 1, "mode": 0,
        "inputs": [{"name": name, "type": data_type, "link": None} for name, data_type in inputs],
        "outputs": [{"name": name, "type": data_type, "links": None} for name, data_type in outputs],
        "properties": {"Node name for S&R": kind}, "widgets_values": widgets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("worms_head", "woodside_library"), default="worms_head")
    args = parser.parse_args()
    woodside = args.fixture == "woodside_library"
    place_widgets = (["Woodside Library, 3140 Woodside Road, Woodside, CA", "Search place name",
                      37.4291056, -122.2580225, 1.2] if woodside else
                     ["Worm's Head, Rhossili, Wales", "Search place name", 51.5643, -4.3241, 2.5])
    nodes = [
        node(12, "GeoBlockDiscoverSources", "0 · Browse geometry sources", (80, -230), (430, 180), [],
             [("source report", "STRING")], ["All", ""]),
        node(13, "GeoBlockShowReport", "Source catalogue and eligibility", (570, -230), (430, 180),
             [("report", "STRING")], [], []),
        node(1, "GeoBlockResolvePlace", "1 · Choose place and coverage", (80, 160), (355, 285), [],
             [("place", "GEOBLOCK_PLACE"), ("location report", "STRING")],
             place_widgets),
        node(2, "GeoBlockTerrainDemo", "2 · Verify measured 3D scene", (515, 160), (315, 160),
             [("place", "GEOBLOCK_PLACE")],
             [("terrain scene", "GEOBLOCK_SCENE"), ("scene preview", "IMAGE"), ("terrain report", "STRING")], []),
        node(3, "GeoBlockPlanShot", "3 · Plan camera and duration", (940, 160), (340, 195),
             [("terrain_scene", "GEOBLOCK_SCENE")],
             [("shot", "GEOBLOCK_SHOT"), ("H3 frames", "INT"), ("shot report", "STRING")],
             ["Close measured-building orbit", 5.2] if woodside else ["Wide turn, slower return", 7.3]),
        node(4, "GeoBlockBuildReferences", "4 · Prepare H3 references", (1390, 160), (380, 245),
             [("terrain_scene", "GEOBLOCK_SCENE"), ("shot", "GEOBLOCK_SHOT")],
             [("control video", "VIDEO"), ("place photo 1", "IMAGE"), ("place photo 2", "IMAGE"),
              ("H3 prompt", "STRING"), ("H3 frames", "INT"), ("reference report", "STRING")],
             ["Clear-line comic" if woodside else "Photographic",
              "Not applicable" if woodside else "Breaking waves"]),
        node(5, "GeoBlockShowReport", "Review source and attribution", (1830, 160), (360, 180),
             [("report", "STRING")], [], []),
        node(6, "PreviewImage", "Terrain scene preview", (515, 465), (315, 245),
             [("images", "IMAGE")], [], []),
        node(7, "PreviewImage", "Place image 1", (1370, 550), (310, 245),
             [("images", "IMAGE")], [], []),
        node(8, "PreviewImage", "Place image 2", (1740, 550), (310, 245),
             [("images", "IMAGE")], [], []),
    ]
    if woodside:
        nodes.extend([
            node(9, "GeoBlockClearLineStyle", "Style references for H3 images 3–4",
                 (1390, 860), (370, 145), [("terrain_scene", "GEOBLOCK_SCENE")],
                 [("style image 1", "IMAGE"), ("style image 2", "IMAGE"),
                  ("style report", "STRING")], []),
            node(10, "PreviewImage", "Clear-line style 1", (1390, 1080), (310, 245),
                 [("images", "IMAGE")], [], []),
            node(11, "PreviewImage", "Clear-line style 2", (1740, 1080), (310, 245),
                 [("images", "IMAGE")], [], []),
        ])
    index = {item["id"]: item for item in nodes}
    links = []

    def connect(source, source_slot, target, target_slot):
        link_id = len(links) + 1
        output = index[source]["outputs"][source_slot]
        input_ = index[target]["inputs"][target_slot]
        if output["type"] != input_["type"]:
            raise ValueError(f"Incompatible link {source}:{source_slot} → {target}:{target_slot}")
        output["links"] = (output["links"] or []) + [link_id]
        input_["link"] = link_id
        links.append([link_id, source, source_slot, target, target_slot, output["type"]])

    connect(1, 0, 2, 0)
    connect(12, 0, 13, 0)
    connect(2, 0, 3, 0)
    connect(2, 0, 4, 0)
    connect(3, 0, 4, 1)
    connect(2, 1, 6, 0)
    connect(4, 1, 7, 0)
    connect(4, 2, 8, 0)
    connect(4, 5, 5, 0)
    if woodside:
        connect(2, 0, 9, 0)
        connect(9, 0, 10, 0)
        connect(9, 1, 11, 0)
    graph = {
        "id": "geoblock-woodside" if woodside else "geoblock-prototype", "revision": 0,
        "last_node_id": max(index), "last_link_id": len(links),
        "nodes": nodes, "links": links, "groups": [], "definitions": {"subgraphs": []},
        "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [20, 60]}}, "version": 0.4,
    }
    destination = WOODSIDE_DESTINATION if woodside else DESTINATION
    destination.write_text(json.dumps(graph, indent=2) + "\n")
    print(destination)
    api = {
        "12": {"class_type": "GeoBlockDiscoverSources", "inputs": {
            "category": "All", "place_or_region": ""}},
        "13": {"class_type": "GeoBlockShowReport", "inputs": {"report": ["12", 0]}},
        "1": {"class_type": "GeoBlockResolvePlace", "inputs": dict(zip(
            ("place", "location_mode", "latitude", "longitude", "coverage_radius_km"), place_widgets))},
        "2": {"class_type": "GeoBlockTerrainDemo", "inputs": {"place": ["1", 0]}},
        "3": {"class_type": "GeoBlockPlanShot", "inputs": {
            "terrain_scene": ["2", 0],
            "camera_route": "Close measured-building orbit" if woodside else "Wide turn, slower return",
            "duration_seconds": 5.2 if woodside else 7.3}},
        "4": {"class_type": "GeoBlockBuildReferences", "inputs": {
            "terrain_scene": ["2", 0], "shot": ["3", 0],
            "visual_style": "Clear-line comic" if woodside else "Photographic",
            "sea_motion": "Not applicable" if woodside else "Breaking waves"}},
        "5": {"class_type": "GeoBlockShowReport", "inputs": {"report": ["4", 5]}},
        "6": {"class_type": "PreviewImage", "inputs": {"images": ["2", 1]}},
        "7": {"class_type": "PreviewImage", "inputs": {"images": ["4", 1]}},
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["4", 2]}},
    }
    if woodside:
        api.update({
            "9": {"class_type": "GeoBlockClearLineStyle", "inputs": {"terrain_scene": ["2", 0]}},
            "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
            "11": {"class_type": "PreviewImage", "inputs": {"images": ["9", 1]}},
        })
    api_destination = (ROOT / "outputs/woodside_library_3dep/geoblock_api.json" if woodside else
                       ROOT / "outputs/geoblock_prototype_api.json")
    api_destination.parent.mkdir(parents=True, exist_ok=True)
    api_destination.write_text(json.dumps(api, indent=2) + "\n")
    print(api_destination)


if __name__ == "__main__":
    main()
