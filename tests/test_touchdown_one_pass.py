"""Regression for a downstream candidate conditioned on a changed upstream seed."""
from __future__ import annotations

from pathlib import Path

import pytest

from touchdown.build import Build
from touchdown.pace import frame_indices
from touchdown.terrain_route import choose_landing, choose_view, large_pavement_mask, pavement_corridor, search_views
from touchdown.terrain_quality import assess_endpoint_layout
from touchdown.video import H, W, push_in


def manifest(path: Path, opener_seed: int) -> Path:
    path.write_text(f'''[shot]
name = "cache-regression"
place = "twin-peaks"
person = "test person"
scene = "test scene"
region = "test region"

[stills]
seed = 1
[opener]
seeds = [{opener_seed}]
[cloud]
seeds = [4]
[dive]
[swoop]
seeds = [7]
[body]
[end]
seeds = [3]
[sound]
engine = "procedural"
''')
    return path


def test_upstream_seed_change_invalidates_conditioned_candidate(tmp_path: Path) -> None:
    shot = tmp_path / "shot.toml"
    rendered: list[str] = []

    def run(seed: int) -> str:
        build = Build(manifest(shot, seed), tmp_path, None, False, unattended=True)

        def stage(name: str):
            def write(folder: Path):
                def render(candidate: int, sub: Path):
                    rendered.append(name)
                    source = sub / "source.mp4"
                    source.write_text(f"{name}={candidate}; opener={seed}")
                    return source

                picked, facts = build.candidates(name, folder, render)
                output = folder / f"{name}.mp4"
                output.write_bytes(picked.read_bytes())
                return output, facts
            return write

        build.run("opener", (), stage("opener"))
        return build.run("cloud", ("opener",), stage("cloud")).read_text()

    assert "opener=1" in run(1)
    assert "opener=1" in run(1)
    assert rendered == ["opener", "cloud"]
    assert "opener=2" in run(2)
    assert rendered == ["opener", "cloud", "opener", "cloud"]


def test_unattended_rejects_pick_or_multiple_seeds(tmp_path: Path) -> None:
    shot = manifest(tmp_path / "shot.toml", 1)
    shot.write_text(shot.read_text().replace("seeds = [4]", "seeds = [4, 5]"))
    with pytest.raises(ValueError, match="exactly one seed"):
        Build(shot, tmp_path, None, False, unattended=True)


def test_comfy_python_keeps_venv_symlink_when_made_absolute(tmp_path: Path, monkeypatch) -> None:
    import sys

    shot = manifest(tmp_path / "shot.toml", 1)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)
    build = Build(shot, tmp_path, None, False, comfy_python="venv/bin/python", unattended=True)
    assert build.comfy_python == str(venv_python)
    assert Path(build.comfy_python).resolve() != Path(build.comfy_python)


def test_late_pacing_preserves_order_and_segment_boundaries() -> None:
    bounds = {"opener": [0, 22], "cloud": [22, 52], "dive": [52, 107],
              "swoop": [107, 198], "end": [198, 271]}
    chosen, facts = frame_indices(271, bounds, peak=2.4, dive_start=0.25, swoop_end=0.55)
    assert len(chosen) == 239
    assert chosen[0] == 0 and chosen[-1] == 270
    assert all(b > a for a, b in zip(chosen, chosen[1:]))
    assert facts["segments"] == {"opener": [0, 22], "cloud": [22, 52], "dive": [52, 93],
                                  "swoop": [93, 166], "end": [166, 239]}
    assert max(b - a for a, b in zip(chosen, chosen[1:])) <= 3


def test_cached_nested_stage_output_is_reused(tmp_path: Path) -> None:
    shot = manifest(tmp_path / "shot.toml", 1)
    shot.write_text(shot.read_text() + "\n[terrain]\nenabled = true\n")
    build = Build(shot, tmp_path, None, False, unattended=True)
    called = []

    def nested(folder: Path):
        called.append(True)
        output = folder / "scene" / "guide.mp4"
        output.parent.mkdir()
        output.write_bytes(b"guide")
        return output, {}

    assert build.run("terrain", (), nested).read_bytes() == b"guide"
    assert build.run("terrain", (), nested).read_bytes() == b"guide"
    assert len(called) == 1


def test_landing_scorer_rejects_steep_patch(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    terrain = tmp_path / "dem"
    terrain.mkdir()
    heights = np.zeros((401, 401), np.float32)
    # The catalogue centre is steep. An offset patch is level and paved.
    heights[195:206, 195:206] = np.arange(11, dtype=np.float32)[:, None]
    np.save(terrain / "height.npy", heights)
    image = np.full((2048, 2048, 3), (160, 160, 160), np.uint8)
    path = tmp_path / "naip.png"
    cv2.imwrite(str(path), image)
    chosen = choose_landing(terrain, path, radius_m=40)
    assert chosen["relief_m"] <= 1.2


def test_landing_keeps_a_level_overlook_with_a_nearby_drop(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    terrain = tmp_path / 'dem'
    terrain.mkdir()
    heights = np.zeros((401, 401), np.float32)
    heights[:194] = -40  # a drop roughly 14 m ahead, beyond the standing patch
    np.save(terrain / 'height.npy', heights)
    image = tmp_path / 'naip.png'
    cv2.imwrite(str(image), np.full((2048, 2048, 3), 160, np.uint8))
    chosen = choose_landing(terrain, image, radius_m=30)
    assert (chosen['x'], chosen['y']) == (0, 0)
    assert chosen['relief_m'] == 0


def test_view_search_resolves_missing_heading_toward_measured_drop(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    terrain = tmp_path / "dem"
    terrain.mkdir()
    np.save(terrain / "height.npy", np.zeros((401, 401), np.float32))
    # Only the northern distant view has a drop; the final approach is level.
    far = np.zeros((681, 681), np.float32)
    far[:320] = -800
    np.save(terrain / "height_far.npy", far)
    image = tmp_path / "naip.png"
    cv2.imwrite(str(image), np.full((2048, 2048, 3), 160, np.uint8))
    view = choose_view(terrain, image, radius_m=30)
    assert view["heading"] == pytest.approx(0, abs=15)
    assert view["distant_drop_m"] == 800
    assert view["views_checked"] > 1
    assert search_views(terrain, image, radius_m=30, limit=4)[0] == view


def test_photo_anchor_search_allows_drop_beyond_level_standing_patch(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    terrain = tmp_path / 'dem'
    terrain.mkdir()
    heights = np.zeros((401, 401), np.float32)
    heights[:194] = -20  # drop ahead, outside the small standing patch
    np.save(terrain / 'height.npy', heights)
    far = np.zeros((681, 681), np.float32)
    far[:320] = -800
    np.save(terrain / 'height_far.npy', far)
    image = tmp_path / 'naip.png'
    cv2.imwrite(str(image), np.full((2048, 2048, 3), 160, np.uint8))
    anchors = [{'x': 0, 'y': 0, 'heading_deg': 0, 'page_id': 7}]
    views = search_views(terrain, image, radius_m=30, photo_views=anchors, limit=4)
    assert all(v['photo_anchor']['camera_distance_m'] <= 10 for v in views)
    assert views[0]['photo_anchor']['page_id'] == 7


def test_pavement_corridor_rejects_a_sustained_developed_surface() -> None:
    import numpy as np

    hsv = np.zeros((2048, 2048, 3), np.uint8)
    hsv[:, :, 1] = 105  # natural ground outside the approach
    gray = np.full((2048, 2048), 120, np.uint8)
    # A 50 m-wide paved span from roughly 50 to 160 m behind the landing.
    hsv[1135:1390, 967:1081, 1] = 50
    gray[1135:1390, 967:1081] = 130
    blocked = pavement_corridor(hsv, gray, 0, 0, 0)
    clear = pavement_corridor(hsv, gray, 0, 0, 180)
    assert blocked["pavement_longest_m"] >= 80
    assert not blocked["pavement_clear"]
    assert clear["pavement_clear"]


def test_full_corridor_catches_parking_off_the_centerline() -> None:
    import numpy as np

    hsv = np.zeros((2048, 2048, 3), np.uint8)
    hsv[:, :, 1] = 105
    gray = np.full((2048, 2048), 120, np.uint8)
    # A large lot 300 m behind and 45 m beside the flight line.
    hsv[1640:1790, 1090:1200, 1] = 50
    gray[1640:1790, 1090:1200] = 130
    narrow = pavement_corridor(hsv, gray, 0, 0, 0)
    full = pavement_corridor(hsv, gray, 0, 0, 0, large_pavement_mask(hsv, gray))
    assert narrow["pavement_clear"]
    assert not full["pavement_clear"]
    assert max(full["pavement_swath_fractions"]) > 0.05


def test_endpoint_layout_rejects_pavement_to_cliff_substitution(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    guide = tmp_path / "guide.png"
    road = tmp_path / "road.png"
    cliff = tmp_path / "cliff.png"
    cv2.imwrite(str(guide), np.full((544, 960, 3), (170, 170, 170), np.uint8))
    cv2.imwrite(str(road), np.full((544, 960, 3), (150, 150, 150), np.uint8))
    cv2.imwrite(str(cliff), np.full((544, 960, 3), (70, 110, 170), np.uint8))
    assert not assess_endpoint_layout(guide, road)["warnings"]
    assert assess_endpoint_layout(guide, cliff)["warnings"]
    island = tmp_path / "island.png"
    painted = np.full((544, 960, 3), (150, 150, 150), np.uint8)
    cv2.rectangle(painted, (170, 350), (790, 470), (70, 110, 170), thickness=-1)
    cv2.imwrite(str(island), painted)
    assert "large new foreground surface" in " ".join(assess_endpoint_layout(guide, island)["warnings"])


def test_endpoint_layout_rejects_moved_landmarks(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    guide_image = np.full((544, 960, 3), 170, np.uint8)
    for _ in range(100):
        x, y = int(rng.integers(15, 900)), int(rng.integers(125, 300))
        shade = int(rng.integers(20, 240))
        cv2.circle(guide_image, (x, y), int(rng.integers(2, 7)), (shade,) * 3, -1)
    shifted_image = cv2.warpAffine(guide_image, np.float32([[1, 0, 45], [0, 1, 0]]),
                                   (960, 544), borderValue=(170, 170, 170))
    guide = tmp_path / "guide.png"
    shifted = tmp_path / "shifted.png"
    cv2.imwrite(str(guide), guide_image)
    cv2.imwrite(str(shifted), shifted_image)
    result = assess_endpoint_layout(guide, shifted)
    assert result["landmark_median_shift_px"] == pytest.approx(45, abs=1)
    assert "landmarks moved" in " ".join(result["warnings"])


def test_endpoint_layout_rejects_vanished_landmarks(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    guide_image = np.full((544, 960, 3), 170, np.uint8)
    for _ in range(100):
        x, y = int(rng.integers(15, 900)), int(rng.integers(125, 300))
        shade = int(rng.integers(20, 240))
        cv2.circle(guide_image, (x, y), int(rng.integers(2, 7)), (shade,) * 3, -1)
    guide = tmp_path / "guide.png"
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(guide), guide_image)
    cv2.imwrite(str(blank), np.full((544, 960, 3), 170, np.uint8))
    result = assess_endpoint_layout(guide, blank)
    assert result["landmark_median_shift_px"] is None
    assert "landmarks moved or vanished" in " ".join(result["warnings"])


def test_end_push_preserves_pinned_prefix_without_a_camera_step() -> None:
    import numpy as np

    image = np.zeros((H, W, 3), np.uint8)
    image[H // 2 - 2:H // 2 + 3, W // 2 - 2:W // 2 + 3] = 255
    for i in range(9):
        assert np.array_equal(push_in(image, i, 73, 9, 2.8, 0.5, 0.46), image)
    assert not np.array_equal(push_in(image, 72, 73, 9, 2.8, 0.5, 0.46), image)


def test_flux3_end_prompts_do_not_ask_to_preserve_model_shapes() -> None:
    # Jamie 18 Oct: "preserve the shapes" made FLUX 3 copy the terrain model's low-poly facets (Monument Valley v12)
    import tomllib

    for manifest in (Path(__file__).parents[1] / "shots" / "flux3").glob("*.toml"):
        prompt = tomllib.loads(manifest.read_text()).get("stills", {}).get("end_prompt", "")
        assert "preserve the shape" not in prompt.lower(), manifest.name


def _structure_build(flux3: list[str], **stills) -> Build:
    build = Build.__new__(Build)
    build.m = {"stills": {"flux3": flux3, **stills}}
    return build


def test_flux3_end_still_defaults_to_structure_conditioning() -> None:
    assert _structure_build(["end", "orbit"]).end_conditioning() == "structure"
    assert _structure_build(["orbit"]).end_conditioning() == "render"
    assert _structure_build(["end"], end_conditioning="render").end_conditioning() == "render"
    assert "photograph" in Build.PHOTOREAL


def test_structure_gate_uses_the_skyline_and_redraws_a_pasted_diagram(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    rng = np.random.default_rng(3)
    guide_image = np.zeros((544, 960, 3), np.uint8)
    guide_image[:] = (60, 90, 150)                       # land
    xs = np.arange(960)
    skyline = (200 + 40 * np.sin(xs / 90)).astype(int)
    for x in xs:
        guide_image[:skyline[x], x] = (230, 190, 160)    # sky: blue well above red, bright
    for _ in range(300):                                 # faceted texture the photo will not share
        x, y = int(rng.integers(0, 960)), int(rng.integers(260, 544))
        cv2.circle(guide_image, (x, y), int(rng.integers(2, 6)), (int(rng.integers(0, 255)),) * 3, -1)
    photo_image = guide_image.copy()
    photo_image[260:] = (70, 110, 170)
    guide, photo, diagram = tmp_path / "guide.png", tmp_path / "photo.png", tmp_path / "diagram.png"
    cv2.imwrite(str(guide), guide_image)
    cv2.imwrite(str(photo), photo_image)
    build = _structure_build(["end"])
    build.structure_map(guide, diagram)

    same_skyline = build.end_layout(guide, photo)
    assert not any("landmark" in w for w in same_skyline["warnings"]), same_skyline["warnings"]
    assert same_skyline["photo_geometry"]["skyline_residual_px"] <= 12

    pasted = build.end_layout(guide, diagram)
    assert any("structure map" in w for w in pasted["warnings"])
    assert all(Build.redrawable(w) for w in pasted["warnings"] if "structure map" in w or "landmarks moved" in w)
