"""Measured-terrain Touchdown camera guide, run inside Blender.

Inputs are two height arrays centred on the landing (401x401 at 2 m and
681x681 at 50 m), their shared elevation datum in metadata.json, and the
NAIP pyramid already fetched by ``touchdown.dive``. The output is a 64-frame
cloud-to-500 m dive and an 81-frame 500 m-to-eye-level LTX layout guide. Their
boundary frames use precisely the same camera pose and terrain geometry.

    blender -b -t 4 -noaudio --python scripts/touchdown/terrain_blender.py -- \
      --terrain outputs/touchdown/.../dem --imagery outputs/touchdown/imagery/... \
      --output outputs/touchdown/.../terrain --heading 75.1

The target offset defaults to the supplied landing coordinate. It can be set
explicitly when testing a particular paved area, e.g. ``--target-x 6 --target-y 8``.
The arrays are currently USGS 3DEP in a UTM coordinate system; the caller is
responsible for fetching and resampling them. This is an experimental guide
renderer, not yet part of the unattended build.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def arguments() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terrain", type=Path, required=True)
    p.add_argument("--imagery", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--heading", type=float, required=True, help="clockwise degrees from north")
    p.add_argument("--target-x", type=float, default=0, help="metres east of landing")
    p.add_argument("--target-y", type=float, default=0, help="metres north of landing")
    p.add_argument("--preview-plan", type=Path, help="JSON list of candidate views; render only their final poses")
    p.add_argument("--platform-layout", type=Path)
    p.add_argument("--fit-platform", action="store_true")
    p.add_argument("--clay", action="store_true")
    p.add_argument("--buildings", type=Path, help="Overture buildings manifest (geoblock_core.buildings) to extrude")
    p.add_argument("--structures", type=Path, help="osm_structures.py JSON: OSM buildings, parts and bridge decks to extrude")
    p.add_argument("--glide", help="approach glide factor (1 = ~45 deg, smaller is steeper) or 'auto' to clear buildings")
    p.add_argument("--person-proxy", action="store_true")
    p.add_argument("--person-at", help="x,y metres: stand the 1.7 m proxy here instead of at the target "
                   "(the photo end still's figure, back-projected onto the DEM)")
    p.add_argument("--person-height", type=float, default=1.7)
    p.add_argument("--person-z", type=float, help="the proxy's base height (map z), when it stands on ground the "
                   "photo end still added rather than on the DEM")
    p.add_argument("--person-rgb", help="r,g,b 0-1: the proxy's coat colour (the end still's figure)")
    p.add_argument("--aim-z", type=float, help="the height (map z) the guide's last frames aim at, instead of 1.4 m "
                   "above the ground at the target (the photo end's figure standing above the smoothed DEM)")
    p.add_argument("--resolution", default="960x544", help="render size WxH (1920x1088: the swoop's LTX upscaler pass "
                   "re-adds the guide at full size; a 960 guide left the land soft, Monument Valley v11)")
    p.add_argument("--min-clearance", type=float, default=0.0,
                   help="keep the guide camera at least this high over the DEM until its last 12 frames (10 m DEM "
                        "fallback: the smoothed ground stays out of frame)")
    p.add_argument("--retreat-entry", type=Path, help="Experimental computed descent ending at inferred retreat view")
    p.add_argument("--end-fit", type=Path, help="JSON {pitch_deg, yaw_deg, zoom, ease_frames}: re-render only the last "
                   "guide frames with the end camera turned/zoomed to the FLUX end still (no dive, no boundary check)")
    return p.parse_args(args)


def material(name: str, path: Path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(path))
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 1
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def main() -> None:
    a = arguments()
    near = np.load(a.terrain / "height.npy")
    far = np.load(a.terrain / "height_far.npy")
    meta = json.loads((a.terrain / "metadata.json").read_text())
    if near.shape != (401, 401) or far.shape != (681, 681) or meta["step"] != 2:
        raise ValueError("Expected 401x401 2 m near DEM and 681x681 50 m far DEM")
    for width in (900, 2700, 8100, 24300, 72900):
        if not (a.imagery / f"naip_{width:05d}.png").is_file():
            raise FileNotFoundError(a.imagery / f"naip_{width:05d}.png")
    a.output.mkdir(parents=True, exist_ok=True)
    datum = float(meta["landing"])
    near = near - datum
    far = far - datum
    if not np.isfinite(near).all() or not np.isfinite(far).all():
        raise ValueError("Terrain DEM contains nodata")

    # Make coarse and fine DEMs meet at the fine patch edge. Keep the coarse
    # surface 2 m underneath the fine one inside the patch to fill tiny gaps.
    ii, jj = np.mgrid[:681, :681]
    x = (jj - 340) * 50.0
    y = (340 - ii) * 50.0
    col = np.clip(x, -400, 400) / 2 + 200
    row = 200 - np.clip(y, -400, 400) / 2
    near_at_far = near[np.rint(row).astype(int), np.rint(col).astype(int)]
    square_radius = np.maximum(np.abs(x), np.abs(y))
    weight = np.clip((500 - square_radius) / 100, 0, 1)
    far = far * (1 - weight) + near_at_far * weight - np.where(square_radius <= 400, 2.0, 0.0)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    def grid(name: str, heights: np.ndarray, step: float):
        n = len(heights)
        vertices = [((j - (n - 1) / 2) * step, ((n - 1) / 2 - i) * step, float(heights[i, j]))
                    for i in range(n) for j in range(n)]
        faces = [(i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1)
                 for i in range(n - 1) for j in range(n - 1)]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj, mesh.uv_layers.new(name="map")

    detail_file = a.terrain / 'height_detail.npy'
    detail = np.load(detail_file) - datum if detail_file.exists() else near
    detail_step = 1 if detail_file.exists() else 2
    near_obj, near_uv = grid("USGS 3DEP landing", detail, detail_step)
    for polygon in near_obj.data.polygons:
        polygon.use_smooth = True
    (a.output / 'terrain_resolution_audit.json').write_text(json.dumps({
        'render_step_m': detail_step, 'search_step_m': meta['step'],
        'mesh_shape': list(detail.shape), 'texture_projection': 'vertical aerial projection',
        'limitations': ['No cliff-face texture observations', 'No overhangs in height-field DEM']}, indent=2) + '\n')
    near_obj.data.materials.append(material("NAIP 900m landing", a.imagery / "naip_00900.png"))
    for polygon in near_obj.data.polygons:
        for li in polygon.loop_indices:
            v = near_obj.data.vertices[near_obj.data.loops[li].vertex_index].co
            near_uv.data[li].uv = (v.x / 900 + 0.5, v.y / 900 + 0.5)

    far_obj, far_uv = grid("USGS 3DEP valley", far, 50)
    levels = (900, 2700, 8100, 24300, 72900)
    for width in levels:
        far_obj.data.materials.append(material(f"NAIP {width}m", a.imagery / f"naip_{width:05d}.png"))
    for polygon in far_obj.data.polygons:
        vertices = [far_obj.data.vertices[far_obj.data.loops[li].vertex_index].co
                    for li in polygon.loop_indices]
        center = sum(vertices, Vector((0, 0, 0))) / len(vertices)
        radius = max(abs(center.x), abs(center.y))
        index = next((k for k, width in enumerate(levels) if radius < width * 0.46), len(levels) - 1)
        polygon.material_index = index
        width = levels[index]
        for li, v in zip(polygon.loop_indices, vertices):
            far_uv.data[li].uv = (v.x / width + 0.5, v.y / width + 0.5)

    if a.buildings:
        # Overture footprints extruded as plain concrete-grey prisms standing on the rendered DEM (Jamie 17 Oct:
        # the user picks the spot, cities included, so structures must be in the measured view as real volumes,
        # not smeared across the ground as aerial texture). Each sits on the lowest terrain under its outline.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from geoblock_core.buildings import extrude
        info = json.loads(a.buildings.read_text())
        with np.load(info["data"]) as data:
            xy, offsets = data["xy"].astype("float64"), data["offsets"]
            base, height = data["base_m"].astype("float64"), data["height_m"].astype("float64")
        rings = np.split(xy, offsets[1:-1]) if len(offsets) > 1 else []
        kept, bottoms, tops = [], [], []
        for ring, b, h in zip(rings, base, height):
            zs = []
            for x, y in ring:
                for obj in (near_obj, far_obj):
                    hit, location, _, _ = obj.ray_cast(Vector((x, y, 5000)), Vector((0, 0, -1)))
                    if hit:
                        zs.append(location.z)
                        break
            if not zs:
                continue
            ground = min(zs) - 0.5  # bury slightly so slopes never show a gap under the wall
            kept.append(ring)
            bottoms.append(ground + b)
            tops.append(ground + h)
        if kept:
            verts, loops, starts, totals = extrude(kept, np.array(bottoms), np.array(tops))
            mesh = bpy.data.meshes.new("Overture buildings")
            mesh.vertices.add(len(verts))
            mesh.vertices.foreach_set("co", verts.astype("float32").ravel())
            mesh.loops.add(len(loops))
            mesh.loops.foreach_set("vertex_index", loops.astype("int32"))
            mesh.polygons.add(len(totals))
            mesh.polygons.foreach_set("loop_start", starts.astype("int32"))
            mesh.polygons.foreach_set("loop_total", totals.astype("int32"))
            mesh.update(calc_edges=True)
            # Roofs take the same aerial photographs as the terrain (vertical projection, the NAIP square that
            # covers them), so the computed dive keeps looking photographed instead of turning into a clay city;
            # walls get a per-building facade tone with a window grid (Jamie 17 Oct, option 1: shrink the look gap
            # the LTX hops have to cross). Faces come out of extrude() as one roof then the walls, per building.
            facade = bpy.data.materials.new("Overture facades")
            facade.use_nodes = True
            nt = facade.node_tree
            bsdf = nt.nodes.get("Principled BSDF")
            bsdf.inputs["Roughness"].default_value = 0.7
            coord = nt.nodes.new("ShaderNodeTexCoord")
            sep = nt.nodes.new("ShaderNodeSeparateXYZ")
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            comb = nt.nodes.new("ShaderNodeCombineXYZ")
            brick = nt.nodes.new("ShaderNodeTexBrick")
            tint = nt.nodes.new("ShaderNodeAttribute")
            tint.attribute_name = "facade_tint"
            glass = nt.nodes.new("ShaderNodeMixRGB")
            glass.blend_type = "MULTIPLY"
            glass.inputs[0].default_value = 1.0
            glass.inputs["Color2"].default_value = (0.20, 0.23, 0.27, 1)
            nt.links.new(coord.outputs["Object"], sep.inputs["Vector"])
            nt.links.new(sep.outputs["X"], add.inputs[0])
            nt.links.new(sep.outputs["Y"], add.inputs[1])
            nt.links.new(add.outputs["Value"], comb.inputs["X"])
            nt.links.new(sep.outputs["Z"], comb.inputs["Y"])
            nt.links.new(comb.outputs["Vector"], brick.inputs["Vector"])
            brick.inputs["Scale"].default_value = 0.3
            brick.inputs["Mortar Size"].default_value = 0.3
            brick.offset = 0.0
            brick.inputs["Brick Width"].default_value = 1.0
            brick.inputs["Row Height"].default_value = 1.1
            nt.links.new(tint.outputs["Color"], glass.inputs["Color1"])
            nt.links.new(glass.outputs["Color"], brick.inputs["Color1"])
            nt.links.new(glass.outputs["Color"], brick.inputs["Color2"])
            nt.links.new(tint.outputs["Color"], brick.inputs["Mortar"])
            dim = nt.nodes.new("ShaderNodeMixRGB")  # the bright sky light lifts walls to pastel: darken them
            dim.blend_type = "MULTIPLY"
            dim.inputs[0].default_value = 1.0
            dim.inputs["Color2"].default_value = (0.5, 0.5, 0.5, 1)
            nt.links.new(brick.outputs["Color"], dim.inputs["Color1"])
            nt.links.new(dim.outputs["Color"], bsdf.inputs["Base Color"])
            bsdf.inputs["Specular IOR Level"].default_value = 0.2
            mesh.materials.append(facade)
            roof_levels = (900, 2700, 8100)
            for width in roof_levels:
                mesh.materials.append(material(f"NAIP {width}m roofs", a.imagery / f"naip_{width:05d}.png"))
            uv = mesh.uv_layers.new(name="map")
            tints = mesh.attributes.new("facade_tint", "FLOAT_COLOR", "FACE")
            palette = [(0.40, 0.37, 0.33), (0.50, 0.47, 0.41), (0.34, 0.22, 0.18), (0.26, 0.29, 0.32),
                       (0.37, 0.37, 0.36), (0.44, 0.41, 0.36), (0.22, 0.24, 0.27)]  # stone, limestone, brick, glass, concrete
            rng = np.random.default_rng(0)
            face = 0
            for b, ring in enumerate(kept):
                centre = ring.mean(axis=0)
                radius = max(abs(centre[0]), abs(centre[1]))
                level = next((k for k, w in enumerate(roof_levels) if radius < w * 0.46), len(roof_levels) - 1)
                width = roof_levels[level]
                tone = palette[int(rng.integers(len(palette)))]
                tone = tuple(float(np.clip(c * rng.uniform(0.85, 1.1), 0, 1)) for c in tone)
                roof = mesh.polygons[face]
                roof.material_index = 1 + level
                for li in roof.loop_indices:
                    v = mesh.vertices[mesh.loops[li].vertex_index].co
                    uv.data[li].uv = (v.x / width + 0.5, v.y / width + 0.5)
                for k in range(face, face + 1 + len(ring)):
                    tints.data[k].color = (*tone, 1.0)
                face += 1 + len(ring)
            mesh.validate()
            blocks = bpy.data.objects.new("Overture buildings", mesh)
            bpy.context.collection.objects.link(blocks)
        (a.output / "buildings_audit.json").write_text(json.dumps({
            "manifest": str(a.buildings), "count_in_manifest": int(len(rings)), "placed": len(kept),
            "height_p50_m": float(np.median(height)) if len(height) else None,
            "attribution": info.get("attribution"), "license": info.get("license")}, indent=2) + "\n")

    if a.structures:
        # OSM structures with mapped heights (Battery Spencer 18 Oct: the Golden Gate's towers, 225 m, are stacked
        # building:parts; the bare-earth map had no bridge, and the swoop grew the tower out of nothing). Plain
        # prisms on the lowest ground under each outline, coloured from the OSM colour tag or the aerial photograph
        # under them; the flat imagery below stays as it is.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from geoblock_core.buildings import extrude
        info = json.loads(a.structures.read_text())
        kept, bottoms, tops, colours = [], [], [], []
        for item in info["structures"]:
            ring = np.array(item["ring"], dtype="float64")
            zs = []
            for x, y in ring:
                for obj in (near_obj, far_obj):
                    hit, location, _, _ = obj.ray_cast(Vector((x, y, 5000)), Vector((0, 0, -1)))
                    if hit:
                        zs.append(location.z)
                        break
            if not zs:
                continue
            ground = min(zs) - 0.5
            kept.append(ring)
            bottoms.append(ground + item["min"])
            tops.append(ground + item["max"])
            colours.append(tuple(item["rgb"]))
        if kept:
            verts, loops, starts, totals = extrude(kept, np.array(bottoms), np.array(tops))
            mesh = bpy.data.meshes.new("OSM structures")
            mesh.vertices.add(len(verts))
            mesh.vertices.foreach_set("co", verts.astype("float32").ravel())
            mesh.loops.add(len(loops))
            mesh.loops.foreach_set("vertex_index", loops.astype("int32"))
            mesh.polygons.add(len(totals))
            mesh.polygons.foreach_set("loop_start", starts.astype("int32"))
            mesh.polygons.foreach_set("loop_total", totals.astype("int32"))
            mesh.update(calc_edges=True)
            plain = bpy.data.materials.new("OSM structures")
            plain.use_nodes = True
            nt = plain.node_tree
            bsdf = nt.nodes.get("Principled BSDF")
            bsdf.inputs["Roughness"].default_value = 0.6
            tint = nt.nodes.new("ShaderNodeAttribute")
            tint.attribute_name = "osm_tint"
            nt.links.new(tint.outputs["Color"], bsdf.inputs["Base Color"])
            mesh.materials.append(plain)
            tints = mesh.attributes.new("osm_tint", "FLOAT_COLOR", "FACE")
            face = 0
            for ring, rgb in zip(kept, colours):
                linear = tuple(float(c) ** 2.2 for c in rgb)  # sRGB tag/photo colour to Blender's linear
                for k in range(face, face + 1 + len(ring)):
                    tints.data[k].color = (*linear, 1.0)
                face += 1 + len(ring)
            mesh.validate()
            structures = bpy.data.objects.new("OSM structures", mesh)
            bpy.context.collection.objects.link(structures)
        (a.output / "structures_audit.json").write_text(json.dumps({
            "file": str(a.structures), "count": info.get("count"), "placed": len(kept),
            "tallest_m": info.get("tallest_m"), "source": info.get("source")}, indent=2) + "\n")

    path_objects = []
    if a.platform_layout:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from platform_layout import strip_mesh, fit_walking_surface
        layout = json.loads(a.platform_layout.read_text())
        path_material = bpy.data.materials.new('Shared pedestrian surface')
        path_material.diffuse_color = (0.48, 0.46, 0.42, 1)
        path_material.use_nodes = True
        shader = path_material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = (0.48, 0.46, 0.42, 1)
        shader.inputs['Roughness'].default_value = 1
        surface_audit = []
        walking_samples = []
        strips = []
        all_vertices, all_faces = [], []
        for path in layout['paths']:
            if path['tags'].get('surface') in ('rock', 'dirt', 'ground', 'unpaved', 'gravel', 'sand'):
                continue  # keep those surfaces in the measured aerial texture
            vertices, faces = strip_mesh(path['points'], path['width_m'], near)
            # Match the rendered DEM's triangulation, not a different bilinear
            # height surface that can sink through it at eye level.
            draped = []
            for x,y,z in vertices:
                hit, location, _, _ = near_obj.ray_cast(Vector((x,y,2000)), Vector((0,0,-1)))
                draped.append((x,y,float(location.z)+0.08 if hit else z))
            offset = len(all_vertices)
            all_vertices.extend(draped)
            all_faces.extend(tuple(i + offset for i in face) for face in faces)
            strips.append((path, offset, len(draped)))
        if a.fit_platform:
            fitted, kept_faces, audit = fit_walking_surface(all_vertices, all_faces, (a.target_x, a.target_y))
        else:
            fitted, kept_faces, audit = all_vertices, all_faces, {'assessed': False, 'reason': 'platform fitting disabled'}
        surface_audit.append(audit)
        walking_samples.extend(fitted[i] for i in {i for face in kept_faces for i in face})
        for path, offset, count in strips:
            faces = [tuple(i-offset for i in face) for face in kept_faces
                     if offset <= face[0] < offset + count]
            mesh = bpy.data.meshes.new(f"Pedestrian surface {path['id']}")
            mesh.from_pydata(fitted[offset:offset+count], [], faces)
            mesh.update()
            obj = bpy.data.objects.new(mesh.name, mesh)
            bpy.context.collection.objects.link(obj)
            obj.data.materials.append(path_material)
            path_objects.append(obj)
        # Small cut corrections keep the fitted pavement from sinking through
        # its underlying DEM. Only the actual strip footprint is affected.
        from mathutils.kdtree import KDTree
        if walking_samples:
            tree = KDTree(len(walking_samples))
            for index, point in enumerate(walking_samples):
                tree.insert((point[0], point[1], 0), index)
            tree.balance()
            for vertex in near_obj.data.vertices:
                _, index, distance = tree.find((vertex.co.x, vertex.co.y, 0))
                if distance < 0.45:
                    vertex.co.z = min(vertex.co.z, walking_samples[index][2] - 0.08)
        near_obj.data.update()
        (a.output / 'walking_surface_audit.json').write_text(json.dumps(surface_audit, indent=2) + '\n')

    world = bpy.data.worlds.new("daylight")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0.38, 0.55, 0.75, 1)
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(30), 0, math.radians(-30))
    sun_data.energy = 2
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = 36 / (2 * math.tan(math.radians(30)))  # 60-degree horizontal FOV
    camera_data.clip_end = 40000
    scene = bpy.context.scene
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = (int(v) for v in a.resolution.lower().split("x"))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = 24
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = 8

    def sample(px: float, py: float) -> float:
        if abs(px) <= 395 and abs(py) <= 395:
            return float(near[round(200 - py / 2), round(200 + px / 2)])
        return float(far[round(340 - py / 50), round(340 + px / 50)])

    if not 0 <= a.heading < 360:
        raise ValueError("Blender requires a resolved heading in [0, 360) degrees")
    heading = math.radians(a.heading)
    forward = Vector((math.sin(heading), math.cos(heading), 0))
    target = Vector((a.target_x, a.target_y, 0))
    # Translate the entire path with the selected landing. Previously only the
    # finish was translated, which made an offset landing induce a sideways
    # sweep even when the requested heading was constant.
    finish = target - forward * 10

    def aim_end(sample_fn) -> float:
        return a.aim_z if a.aim_z is not None else sample_fn(target.x, target.y) + 1.4

    def guide_path(k: float):
        """Camera positions and aims of the 81 guide frames for glide factor ``k`` (1: start 486 m back, ~45 deg)."""
        s = target + forward * (15 - 501 * k)
        ls = 501 - sample(s.x, s.y)
        poses = []
        for frame in range(1, 82):
            t = (frame - 1) / 80
            progress = (501 - 501 * math.exp(math.log(2 / 501) * t)) / 499
            xy = s * (1 - progress) + finish * progress
            clearance = ls * math.exp(math.log(2 / ls) * t) if ls > 2 else 2
            pos = Vector((xy.x, xy.y, sample(xy.x, xy.y) + clearance))
            aim_xy = (target + forward * 15) * (1 - progress) + target * progress
            poses.append((pos, Vector((aim_xy.x, aim_xy.y, aim_end(sample) * progress))))
        return s, ls, poses

    glide, glide_audit = 1.0, None
    blockers = bpy.data.objects.get("Overture buildings")
    if a.glide == "auto" and blockers is not None:
        # Cities (Jamie 17 Oct): a ~45 deg glide into Bryant Park flies through Midtown towers. Keep the user's
        # heading (it sets the final view) and steepen the approach until no building blocks the camera's line of
        # sight to the landing on any guide frame; the steepest tried is used if none is fully clear.
        glide_audit = []
        for k in (1.0, 0.75, 0.55, 0.4, 0.3, 0.22, 0.16):
            _, ls, poses = guide_path(k)
            blocked = []
            for frame, (pos, aim) in enumerate(poses[:-6], 1):  # the last frames are at eye level beside her
                ray = aim - pos
                hit, *_ = blockers.ray_cast(pos, ray.normalized(), distance=ray.length)
                if hit:
                    blocked.append(frame)
            glide_audit.append({"glide": k, "blocked_frames": len(blocked), "first_blocked": blocked[:5]})
            glide = k
            if not blocked and ls > 2:
                break
    elif a.glide not in (None, "auto"):
        glide = float(a.glide)
    start = target + forward * (15 - 501 * glide)
    local_start = 501 - sample(start.x, start.y)
    (a.output / "glide.json").write_text(json.dumps({"glide": glide, "start_back_m": round(501 * glide - 15, 1),
                                                       "trials": glide_audit}, indent=2) + "\n")
    if local_start <= 2:
        raise ValueError("500 m handoff camera intersects the terrain")

    if a.clay:
        clay = bpy.data.materials.new('Untextured terrain layout')
        clay.diffuse_color = (0.55,0.55,0.55,1)
        clay.use_nodes = True
        clay.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (0.55,0.55,0.55,1)
        clay.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value = 1
        for obj in (near_obj,far_obj):
            obj.data.materials.clear();obj.data.materials.append(clay)
            for polygon in obj.data.polygons: polygon.material_index=0
    if a.person_proxy or a.person_at:
        # An explicitly stationary human proxy (1.7 m unless --person-height) at the chosen target, or where the photo
        # end still's figure stands (--person-at; Battery Spencer v12: without it the swoop kept the figure at the old
        # landing, 16 px tall, and grew it 28x in the last 11 frames to meet the end still)
        px, py = (float(v) for v in a.person_at.split(",")) if a.person_at else (target.x, target.y)
        k = a.person_height / 1.7
        def plain(name, rgb):
            # EEVEE renders the node tree, not diffuse_color (the proxy came out white, v13 first try)
            mat = bpy.data.materials.new(name)
            mat.diffuse_color = (*rgb, 1)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (*rgb, 1)
            bsdf.inputs["Roughness"].default_value = 0.8
            return mat
        proxy_material = plain('Human proxy', (0.03, 0.03, 0.035))
        coat_material = proxy_material
        if a.person_rgb:
            coat_material = plain('Proxy coat', tuple(float(c) ** 2.2 for c in a.person_rgb.split(",")))
        base_z=a.person_z if a.person_z is not None else sample(px,py)+0.1
        def cylinder(name,offset,depth,radius,mat):
            bpy.ops.mesh.primitive_cylinder_add(vertices=12,radius=radius*k,depth=depth*k,
                location=(px+offset[0]*k,py+offset[1]*k,base_z+offset[2]*k))
            obj=bpy.context.object;obj.name=name;obj.data.materials.append(mat)
        cylinder('Planted left leg',(-.1,0,.42),.84,.07,proxy_material)
        cylinder('Planted right leg',(.1,0,.42),.84,.07,proxy_material)
        cylinder('Torso',(0,0,1.13),.62,.19,coat_material)
        cylinder('Left arm',(-.24,0,1.03),.62,.055,coat_material)
        cylinder('Right arm',(.24,0,1.03),.62,.055,coat_material)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.13*k,
            location=(px,py,base_z+1.57*k))
        bpy.context.object.name='Proxy head';bpy.context.object.data.materials.append(proxy_material)

    def floor_clearance(frame: int, clearance: float) -> float:
        """--min-clearance: at least that high over the DEM until the last 12 guide frames, then eased down."""
        if a.min_clearance <= 0 or clearance >= a.min_clearance:
            return clearance
        if frame <= 69:
            return a.min_clearance
        x = (frame - 69) / 12
        return a.min_clearance + (clearance - a.min_clearance) * x * x * (3 - 2 * x)

    def render_frame(directory: Path, frame: int, position: Vector, aim: Vector) -> None:
        camera.location = position
        camera.rotation_euler = (aim - position).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(directory / f"{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        if path_objects and not a.preview_plan:
            # An exact visibility mask from this same scene and camera, including
            # terrain occlusion; masks do not come from a separately painted shot.
            materials = {obj.name: list(obj.data.materials) for obj in (near_obj, far_obj, *path_objects)}
            def emission(name, color):
                mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                output = nodes.new('ShaderNodeOutputMaterial')
                emit = nodes.new('ShaderNodeEmission')
                emit.inputs['Color'].default_value = (*color, 1)
                mat.node_tree.links.new(emit.outputs[0], output.inputs['Surface'])
                return mat
            black, white = emission('Mask black', (0,0,0)), emission('Mask white', (1,1,1))
            background = world.node_tree.nodes.get('Background').inputs['Color']
            original_color = tuple(background.default_value)
            background.default_value = (0,0,0,1)
            for obj in (near_obj, far_obj, *path_objects):
                obj.data.materials.clear()
                obj.data.materials.append(white if obj in path_objects else black)
            mask_dir = directory.parent / (directory.name + '_surface_masks')
            mask_dir.mkdir(exist_ok=True)
            scene.render.filepath = str(mask_dir / f'{frame:04d}.png')
            bpy.ops.render.render(write_still=True)
            background.default_value = original_color
            for obj in (near_obj, far_obj, *path_objects):
                obj.data.materials.clear()
                for mat in materials[obj.name]:
                    obj.data.materials.append(mat)

    if a.preview_plan:
        previews = a.output / "previews"
        (previews / "contacts").mkdir(parents=True, exist_ok=True)
        (previews / "final").mkdir(exist_ok=True)
        for index, view in enumerate(json.loads(a.preview_plan.read_text()), 1):
            bearing = math.radians(view["heading"])
            direction = Vector((math.sin(bearing), math.cos(bearing), 0))
            point = Vector((view["x"], view["y"], 0))
            start_view = point + direction * (15 - 501)
            finish_view = point - direction * 10
            start_clearance = 501 - sample(start_view.x, start_view.y)
            if start_clearance <= 2:
                raise ValueError(f"Candidate {index} camera intersects the terrain")
            poses = previews / f"poses_{index:04d}"
            poses.mkdir(exist_ok=True)
            scene.render.resolution_percentage = 25
            for pose_index, guide_frame in enumerate((1, 28, 56, 81), 1):
                t = (guide_frame - 1) / 80
                global_height = 501 * math.exp(math.log(2 / 501) * t)
                progress = (501 - global_height) / 499
                xy = start_view * (1 - progress) + finish_view * progress
                clearance = start_clearance * math.exp(math.log(2 / start_clearance) * t)
                location = Vector((xy.x, xy.y, sample(xy.x, xy.y) + clearance))
                aim_xy = (point + direction * 15) * (1 - progress) + point * progress
                aim = Vector((aim_xy.x, aim_xy.y, (sample(point.x, point.y) + 1.4) * progress))
                render_frame(poses, pose_index, location, aim)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "1", "-i",
                            str(poses / "%04d.png"), "-vf", "tile=2x2", "-frames:v", "1",
                            str(previews / "contacts" / f"{index:04d}.png")], check=True)
            scene.render.resolution_percentage = 50
            render_frame(previews / "final", index, location, aim)
        print(f"Rendered {index} candidate final views and four-pose contacts to {previews}", flush=True)
        return

    if a.retreat_entry:
        entry=json.loads(a.retreat_entry.read_text())
        frames=int(entry.get('frames',112));distance=float(entry.get('computed_distance_m',entry['distance_m']))
        end_height=float(entry.get('computed_height_m',entry['height_above_target_m']))
        pitch=float(entry['pitch_deg'])
        if frames<2 or distance<=0 or end_height<=0:
            raise ValueError('Invalid inferred entry pose')
        def bilinear(px,py):
            heights,step=(near,2) if max(abs(px),abs(py))<=395 else (far,50)
            n=len(heights);row=(n-1)/2-py/step;col=(n-1)/2+px/step
            row=float(np.clip(row,0,n-1));col=float(np.clip(col,0,n-1))
            r,c=int(row),int(col);rr,cc=min(r+1,n-1),min(c+1,n-1)
            return float((1-(row-r))*((1-(col-c))*heights[r,c]+(col-c)*heights[r,cc])
                       +(row-r)*((1-(col-c))*heights[rr,c]+(col-c)*heights[rr,cc]))
        target_z=bilinear(target.x,target.y)
        poses=[];directory=a.output/'descent_frames';directory.mkdir(exist_ok=True)
        for frame in range(frames):
            t=frame/(frames-1)
            progress=.8*t+.2*t*t if entry.get('timing')=='accelerate' else 1-(1-t)**1.5
            d=7000*math.exp(math.log(distance/7000)*progress)
            height=7000*math.exp(math.log(end_height/7000)*progress)
            xy=target-forward*d
            terrain_floor=bilinear(xy.x,xy.y)
            z=max(target_z+height,terrain_floor+1.5)
            if entry.get('look_at_waypoint',False):
                angle=math.atan2(target_z+1.0-z,d)
            else:
                turn=float(np.clip((t-.65)/.35,0,1));turn=turn*turn*(3-2*turn)
                angle=math.radians(-45*(1-turn)+pitch*turn)
            location=Vector((xy.x,xy.y,z))
            direction=forward*math.cos(angle)+Vector((0,0,math.sin(angle)))
            render_frame(directory,frame+1,location,location+direction*100)
            poses.append({'frame':frame,'camera_xyz_m':list(location),
                          'pitch_deg':math.degrees(angle),'terrain_clearance_m':z-terrain_floor})
        subprocess.run(['ffmpeg','-v','error','-y','-framerate','24','-i',str(directory/'%04d.png'),
                        '-c:v','libx264','-crf','14','-pix_fmt','yuv420p',str(a.output/'descent.mp4')],check=True)
        (a.output/'descent_poses.json').write_text(json.dumps({'entry':entry,'poses':poses},indent=2))
        print(a.output/'descent.mp4',flush=True)
        return

    if a.end_fit:
        # Move the camera to the still (17 Oct): FLUX 3 draws the distant terrain at its own scale and height, so
        # the guide's last ``ease_frames`` frames turn (pitch up / yaw left, degrees) and zoom (lens ratio) to the
        # camera the end still implies, eased in so the swoop does not jerk. Same path, same poses otherwise.
        from mathutils import Quaternion
        fit = json.loads(a.end_fit.read_text())
        ease = int(fit.get("ease_frames", 24))
        out = a.output / "guide_frames"
        out.mkdir(parents=True, exist_ok=True)
        base_lens = camera_data.lens
        # render_from: with a person proxy the whole guide is re-rendered (the figure is in every frame), the end
        # camera still eased in over the last ``ease`` frames
        for frame in range(int(fit.get("render_from", 82 - ease)), 82):
            t = (frame - 1) / 80
            global_height = 501 * math.exp(math.log(2 / 501) * t)
            progress = (501 - global_height) / 499
            xy = start * (1 - progress) + finish * progress
            clearance = floor_clearance(frame, local_start * math.exp(math.log(2 / local_start) * t))
            position = Vector((xy.x, xy.y, sample(xy.x, xy.y) + clearance))
            aim_xy = (target + forward * 15) * (1 - progress) + target * progress
            aim = Vector((aim_xy.x, aim_xy.y, aim_end(sample) * progress))
            x = max(0.0, (frame - (81 - ease)) / ease)
            w = x * x * (3 - 2 * x)
            q = (aim - position).to_track_quat("-Z", "Y")
            q = Quaternion((0, 0, 1), math.radians(fit.get("yaw_deg", 0) * w)) @ q
            q = q @ Quaternion((1, 0, 0), math.radians(fit.get("pitch_deg", 0) * w))
            # lift_m / back_m: a FLUX still drawn from a higher viewpoint (Battery Spencer v9: the postcard view down
            # on the bridge): the end camera rises and steps back along the heading, the person stays at the landing.
            # The orientation stays as fitted from the skyline (not re-aimed at the person)
            right = Vector((math.cos(heading), -math.sin(heading), 0))
            position = (position + Vector((0, 0, fit.get("lift_m", 0) * w)) - forward * (fit.get("back_m", 0) * w)
                        + right * (fit.get("side_m", 0) * w))
            camera.location = position
            camera.rotation_mode = "QUATERNION"
            camera.rotation_quaternion = q
            camera_data.lens = base_lens * (1 + (fit.get("zoom", 1) - 1) * w)
            scene.render.filepath = str(out / f"{frame:04d}.png")
            bpy.ops.render.render(write_still=True)
        camera.rotation_mode = "XYZ"
        camera_data.lens = base_lens
        (a.output / "end_fit.json").write_text(json.dumps(fit, indent=2) + "\n")
        print(f"End-fitted guide frames {int(fit.get('render_from', 82 - ease))}-81 ready: {out}", flush=True)
        return

    dive_frames = a.output / "dive_frames"
    dive_frames.mkdir(exist_ok=True)
    for frame in range(1, 65):
        t = (frame - 1) / 63
        height = 7000 * math.exp(math.log(501 / 7000) * t)
        # ease from the usual ~45 deg dive to the guide's glide by the last dive frame, so the boundary still
        # matches the guide's first frame exactly when glide < 1 (cities)
        position = target + forward * (15 - height * (1 + (glide - 1) * t * t))
        position.z = height
        aim = target + forward * 15
        render_frame(dive_frames, frame, position, aim)

    guide_frames = a.output / "guide_frames"
    guide_frames.mkdir(exist_ok=True)
    guide_poses = []
    for frame in range(1, 82):
        t = (frame - 1) / 80
        global_height = 501 * math.exp(math.log(2 / 501) * t)
        progress = (501 - global_height) / 499
        xy = start * (1 - progress) + finish * progress
        clearance = floor_clearance(frame, local_start * math.exp(math.log(2 / local_start) * t))
        position = Vector((xy.x, xy.y, sample(xy.x, xy.y) + clearance))
        aim_xy = (target + forward * 15) * (1 - progress) + target * progress
        aim = Vector((aim_xy.x, aim_xy.y, aim_end(sample) * progress))
        guide_poses.append({"frame": frame,
                            "camera_xyz_m": [round(float(v), 4) for v in position],
                            "aim_xyz_m": [round(float(v), 4) for v in aim]})
        render_frame(guide_frames, frame, position, aim)

    # Check the model-free boundary before LTX alters its first frame.
    left_image = bpy.data.images.load(str(dive_frames / "0064.png"))
    right_image = bpy.data.images.load(str(guide_frames / "0001.png"))
    left = np.asarray(left_image.pixels[:], dtype=np.float32)
    right = np.asarray(right_image.pixels[:], dtype=np.float32)
    seam = float(np.abs(left - right).mean())
    bpy.data.images.remove(left_image)
    bpy.data.images.remove(right_image)
    if seam > 0.001:
        raise RuntimeError(f"Measured dive/guide boundary is {seam:.2f} MAE")
    for name, count in (("dive", 64), ("guide", 81)):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "24", "-i",
                        str(a.output / f"{name}_frames" / "%04d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", str(a.output / f"{name}.mp4")], check=True)
    (a.output / "terrain_guide.json").write_text(json.dumps({"heading": a.heading,
        "target_x": a.target_x, "target_y": a.target_y, "boundary_mae": seam,
        "horizontal_fov_deg": 60, "dive_frames": 64, "guide_frames": 81,
        "guide_poses": guide_poses}, indent=2))
    print(f"Terrain guide ready: {a.output}, boundary MAE {seam:.3f}", flush=True)


if __name__ == "__main__":
    main()
