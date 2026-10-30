"""Interior location photos are dropped before the grounding brief, and indices map back."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from touchdown import studio_brief


def test_interior_reference_is_dropped_and_indices_map_back(monkeypatch, tmp_path):
    setup = {"destination": {"title": "Castle Howard"}, "wearing": "", "subject": {"name": "me.jpg"},
             "references": [{"name": "a.jpg"}, {"name": "gallery.jpg"}, {"name": "c.jpg"}]}
    seen = {}

    def chat(url, images, prompt, schema, output):
        if schema is studio_brief.IndoorCheck:
            return {"indoors": images[0][0].name == "gallery.jpg", "evidence": ""}
        seen["images"] = [p.name for p, _ in images]
        return {"selected_references": [0, 1]}

    monkeypatch.setattr(studio_brief, "chat", chat)
    result = studio_brief.grounding(setup, tmp_path, "http://x", tmp_path / "grounding.json")
    assert seen["images"] == ["a.jpg", "c.jpg", "me.jpg"]
    assert result["selected_references"] == [0, 2] and result["indoor_references"] == [1]
