"""The build's ComfyUI calls, runnable on their own so they can use a different Python from the build.

On the desktop the ComfyUI venv has OpenCV but not comfy_sdk, and the project venv has comfy_sdk but not OpenCV, so
``touchdown.build --comfy-python <project venv python>`` runs the joins in one and these calls in the other:

    python -m touchdown.comfy_job ltx '{"url": ..., "folder": ..., ...}'   # prints {"path": ...} as its last line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def ltx(url: str, folder: str, frames: int, pins: list[list], prompt: str, seed: int, prefix: str,
        ltx_upscale: bool = True) -> str:
    """LTX 2.5 image-to-video pinned at ``pins`` ([frame, image or .mp4 path]); returns the saved clip's path.
    ``ltx_upscale`` False skips the 2x latent upscale + refine (960x544, the app's draft quality)."""
    import fakeout_earth_zoom as f
    folder_p = Path(folder)
    args = SimpleNamespace(no_upscale=not ltx_upscale, output_dir=folder_p, url=url)
    with f._client(url) as client:
        graph: dict = {}
        loaded = []
        for k, (i, p) in enumerate(pins):
            if str(p).endswith(".mp4"):  # a short 8n+1-frame clip: a video prefix (LTXVAddGuide takes an image batch)
                node = str(60 + 2 * k)
                graph[node] = {"class_type": "LoadVideo", "inputs": {"file": client.assets.from_file(str(p))}}
                graph[str(61 + 2 * k)] = {"class_type": "GetVideoComponents", "inputs": {"video": [node, 0]}}
                loaded.append((i, [str(61 + 2 * k), 0]))
            else:
                loaded.append((i, f._load(graph, client, str(44 + k), Path(p))))
        saved = f._submit(client, graph, tuple(f._ltx(graph, args, frames, loaded, prompt, seed, prefix)), folder_p)
    return str(saved[f.LTX_VIDEO])


def _free(url: str) -> None:
    """Ask a ComfyUI server to unload its models and free VRAM (best effort)."""
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except OSError as exc:
        print(f"could not free {url}: {exc}", flush=True)


def stills(url: str, folder: str, seed: int, name: str, words: dict, end_still: str | None = None,
           end_reference: str | None = None, subject: str | None = None, end_prompt: str | None = None,
           opener_end_prompt: str | None = None, scene_reference: str | None = None,
           flux3: list[str] | None = None, free_url: str | None = None,
           cloud_prompt: str | None = None) -> dict[str, str]:
    """FLUX klein stills: orbit, cloud and (unless ``end_still`` is given) the end still; returns name -> path.
    Stills named in ``flux3`` ("end", "orbit", "cloud", "opener_end") use FLUX 3 Image Klein at 2K instead, which
    needs a ComfyUI v0.41.3+ server at ``url``. ``free_url`` (that server's own address, not its proxy) is asked
    to unload its models afterwards, so a second ComfyUI does not keep ~17 GB while the LTX stages run."""
    import fakeout_earth_zoom as f
    out: dict[str, str] = {}
    use = set(flux3 or ())
    still = lambda key: f._flux3 if key in use else f._klein
    with f._client(url) as client:
        graph: dict = {}
        wanted: list[str] = []
        if end_still:
            out["end"] = end_still
        else:
            refs = []
            if end_reference:
                refs.append(f._load(graph, client, "42", Path(end_reference)))
            if subject:
                refs.append(f._load(graph, client, "41", Path(subject)))
            if scene_reference:
                refs.append(f._load(graph, client, "43", Path(scene_reference)))
            if end_prompt:
                text = end_prompt
            elif end_reference:
                who = "the person from image 2 (same face, hair and clothes)" if subject else words["person"]
                text = f.END_STILL_PLACE.format(who=who)
            else:  # waist-up portrait
                text = (f.END_STILL_SUBJECT if subject else f.END_STILL).format(**words)
            if "end" in use:  # FLUX 3 draws the end still at 2K: keep that copy too, it is the sharp one
                f._flux3(graph, 100, text, seed, f"geoblock/touchdown_{name}_end", tuple(refs), full=True)
                wanted.append("105")
            else:
                f._klein(graph, 100, text, seed, f"geoblock/touchdown_{name}_end", tuple(refs))
            wanted.append("107")
        orbit = f.ORBIT_STILL_FLUX3 if "orbit" in use else f.ORBIT_STILL
        still("orbit")(graph, 120, orbit.format(**words), seed + 1, f"geoblock/touchdown_{name}_orbit")
        still("cloud")(graph, 140, cloud_prompt or (f.CLOUD_STILL_FLUX3 if "cloud" in use else f.CLOUD_STILL), seed + 140,
                       f"geoblock/touchdown_{name}_cloud")
        wanted += ["127", "147"]
        if opener_end_prompt:
            # This pin controls only the orbit passage. The terrain-aligned landing still can be too unlike
            # the orbit still and cause LTX to linger in space before the cloud arrives.
            still("opener_end")(graph, 160, opener_end_prompt, seed, f"geoblock/touchdown_{name}_opener_end")
            wanted.append("167")
        try:
            saved = f._submit(client, graph, tuple(wanted), Path(folder))
        finally:
            if free_url:
                _free(free_url)
    for key, node in (("end", "107"), ("orbit", "127"), ("cloud", "147"), ("opener_end", "167")):
        if node in saved:
            out[key] = str(saved[node])
    if "105" in saved:  # the end still at its full size, when FLUX 3 drew it
        out["end_full"] = str(saved["105"])
    return out


def mid_still(url: str, folder: str, seed: int, name: str, prompt: str, reference: str,
              flux3: bool = True, free_url: str | None = None) -> str:
    """One photographic still from a reference frame, to pin inside the swoop.

    The swoop is otherwise a video-to-video pass over the *shaded* Blender guide, and the model copies that
    shading: the low-poly look and the invented mid-ground. A photographic frame pinned in the middle gives it
    the other truth to work from.
    """
    import fakeout_earth_zoom as f
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    with f._client(url) as client:
        graph: dict = {}
        painter = f._flux3 if flux3 else f._klein
        ref = f._load(graph, client, "40", Path(reference))
        painter(graph, 100, prompt, seed, f"geoblock/touchdown_{name}_mid", (ref,))
        try:
            saved = f._submit(client, graph, ("107",), out)
        finally:
            if free_url:
                _free(free_url)  # the image model must not be resident when the 21 GB transformer loads
    return saved["107"]


def face_plate(url: str, folder: str, reference: str, seed: int, prompt: str) -> str:
    """Edit one detected late-approach crop into a detailed, square face reference."""
    import fakeout_earth_zoom as f
    out = Path(folder)
    with f._client(url) as client:
        graph: dict = {}
        ref = f._load(graph, client, "42", Path(reference))
        f._klein(graph, 100, prompt, seed, "geoblock/touchdown_face_plate", (ref,))
        for node in ("102", "103"):
            graph[node]["inputs"].update(width=768, height=768)
        saved = f._submit(client, graph, ("107",), out)
    return str(saved["107"])


def buildings(url: str, x: float, y: float, zone: int, radius_m: float, folder: str, cache_dir: str) -> str:
    """Overture building footprints around the landing (UTM ``x``, ``y``), for terrain_blender.py --buildings;
    here because reading Overture needs pyarrow, shapely and pyproj. Returns the manifest path."""
    from geoblock_core.buildings import fetch_buildings
    from geoblock_core.utm import projected_crs
    scene = {"projected_crs": projected_crs(zone), "origin_projected_m": [x, y], "radius_m": radius_m}
    return fetch_buildings(scene, Path(folder), cache_dir=Path(cache_dir), half_size_m=radius_m)["path"]


def sea_masks(url: str, **kw) -> dict[str, str]:
    """Overture sea masks for the NAIP squares (touchdown.sea.masks); here because it needs pyarrow."""
    from touchdown import sea
    return sea.masks(**kw)


def terrain_v2v(url: str, guide: str, first: str, last: str, output: str,
                seed: int, frames: int, prompt: str, ltx_upscale: bool = False,
                mids: list | None = None) -> str:
    """Guided LTX 2.5 approach from the measured camera video.

    ``mids`` are extra pinned frames: ``[frame index, image path]``. A photographic frame pinned in the middle
    gives the model something better than the shaded guide to copy.
    """
    from touchdown.terrain_v2v import render
    return str(render(Path(guide), Path(first), Path(last), Path(output), seed, frames, prompt, url, ltx_upscale,
                      [(int(i), Path(p)) for i, p in (mids or [])]))


def backward_edit(url: str, folder: str, guide: str, later: str, strength: float,
                  denoise: float, seed: int, prompt: str, appearance_only: bool = False) -> str:
    """Mix geometry-only and geometry-plus-later model predictions, not output images."""
    import fakeout_earth_zoom as f
    if not 0 <= strength <= 1 or not 0 < denoise <= 1:
        raise ValueError('Invalid backward ramp strength or denoise')
    with f._client(url) as client:
        graph = {}
        geometry = f._load(graph, client, '40', Path(guide))
        appearance = f._load(graph, client, '41', Path(later))
        f._klein(graph, 100, prompt, seed, 'geoblock/touchdown_backward', (geometry,appearance))
        graph['102']['inputs'].update(width=960,height=544)
        graph['103']['inputs'].update(width=960,height=544,steps=8)
        graph['180'] = {'class_type':'ImageScale','inputs':{'image':geometry,'upscale_method':'lanczos',
            'width':960,'height':544,'crop':'disabled'}}
        graph['181'] = {'class_type':'VAEEncode','inputs':{'pixels':['180',0],'vae':['3',0]}}
        graph['182'] = {'class_type':'SplitSigmasDenoise','inputs':{'sigmas':['103',0],'denoise':denoise}}
        graph['104'] = {'class_type':'DualCFGGuider','inputs':{'model':['1',0],
            'cond1':['100r1p',0], 'cond2':['100r0p',0], 'negative':['100r0n',0],
            'cfg_conds':strength,'cfg_cond2_negative':1.0,'style':'regular'}}
        if appearance_only:
            graph['190'] = {'class_type':'ReferenceLatent','inputs':{
                'conditioning':['100',0], 'latent':['100r1e',0]}}
            graph['104']['inputs']['cond1'] = ['190',0]
        graph['105']['inputs'].update(latent_image=['181',0],sigmas=['182',1])
        saved=f._submit(client,graph,('107',),Path(folder))
    return str(saved['107'])


def layout_portrait(url: str, folder: str, guide: str, identity: str, seed: int, prompt: str | None = None) -> str:
    import fakeout_earth_zoom as f
    with f._client(url) as client:
        graph={}
        layout=f._load(graph,client,'40',Path(guide))
        reference=f._load(graph,client,'41',Path(identity))
        prompt=prompt or ('Image 1 is an untextured 3D layout. Render it as a real photograph at the same camera viewpoint. '
          'Replace its central grey human proxy with the woman from image 2, preserving her exact face, '
          'mustard yellow jacket, dark green trousers and boots. Keep the exact proxy height in the frame '
          'and the exact positions of its head and planted feet. Keep platform edges, cliff silhouettes, '
          'horizon and perspective from image 1. Natural concrete and rock textures, natural daylight. '
          'Do not copy the viewpoint or extra structures from image 2. No walking or invented railings.')
        f._klein(graph,100,prompt,seed,'geoblock/touchdown_clay_endpoint',(layout,reference))
        graph['102']['inputs'].update(width=960,height=544)
        graph['103']['inputs'].update(width=960,height=544)
        saved=f._submit(client,graph,('107',),Path(folder))
    return str(saved['107'])


def closeup_upscale(url: str, folder: str, reference: str) -> str:
    """Fourfold RealESRGAN enlargement of a crop; no generative identity repaint."""
    import fakeout_earth_zoom as f
    with f._client(url) as client:
        graph = {}
        ref = f._load(graph, client, '40', Path(reference))
        graph['100'] = {'class_type': 'UpscaleModelLoader',
                        'inputs': {'model_name': 'RealESRGAN_x2plus.pth'}}
        for node, source in [('101', ref), ('102', ['101', 0])]:
            graph[node] = {'class_type': 'ImageUpscaleWithModel',
                           'inputs': {'upscale_model': ['100', 0], 'image': source}}
        graph['107'] = {'class_type': 'SaveImage', 'inputs': {
            'images': ['102', 0], 'filename_prefix': 'geoblock/touchdown_closeup_plate'}}
        saved = f._submit(client, graph, ('107',), Path(folder))
    return str(saved['107'])


def closeup_refine(url: str, folder: str, reference: str, identity: str, seed: int) -> str:
    """Refine an enlarged scene crop with a separate isolated identity reference."""
    import fakeout_earth_zoom as f
    with f._client(url) as client:
        graph = {}
        refs = tuple(f._load(graph, client, str(40+i), Path(p))
                     for i, p in enumerate([reference, identity]))
        prompt=('Refine image 1 into a sharp realistic close-up photograph. Preserve its exact framing, '
                'camera direction, connected stone walls, steps, railings and background positions. '
                'Image 2 identifies the same woman: preserve her facial identity, black hair and yellow '
                'jacket. Repair the low-resolution facial detail in image 1 using image 2; natural eyes, '
                'real skin texture, clear eyelashes and lips, no smoothing or plastic skin. '
                'Do not change her pose, expression, head size or position. Same lighting, no text.')
        f._klein(graph, 100, prompt, seed, 'geoblock/touchdown_closeup_refined', refs)
        saved = f._submit(client, graph, ('107',), Path(folder))
    return str(saved['107'])


def topographic_still(url: str, folder: str, geometry: str, photos: list[str],
                      subject: str, seed: int, geometry_first: bool = True,
                      photo_structure: bool = False) -> str:
    """Clean role-separated references: coarse layout, actual materials, isolated subject."""
    import fakeout_earth_zoom as f
    if len(photos)!=2:
        raise ValueError('This bounded still test needs a platform and a canyon photograph')
    inputs=([geometry,*photos,subject] if geometry_first else [*photos,geometry,subject])
    geo,platform,canyon=(1,2,3) if geometry_first else (3,1,2)
    prompt=(f'Create one convincing real photograph of a woman at the Mather Point pedestrian overlook. '
      f'Image {geo} is a crude untextured geometric model, supplied STRICTLY as topographic and camera '
      'reference: use only its broad landform elevations, cliff and ridge arrangement, perspective, '
      'camera direction, and the location of the standing human marker. It is NOT an appearance reference. '
      'Do not reproduce its shader, white or grey clay, flat polygons, mesh facets, crude pavement strips '
      'or artificial stepped edges. Those are modelling placeholders, not real surfaces. '
      f'Image {platform} is a real photograph documenting the actual pedestrian surface, concrete finish, '
      'stone construction, railings and vegetation. Use it for believable surface detail and materials. '
      f'Image {canyon} documents real layered rock textures, natural erosion and distant canyon atmosphere. '
      'Use these photos for photographic appearance, not their camera viewpoints or compositions. '
      'Reconstruct all visible ground as one continuous realistic physical walking surface fitting '
      'the broad topography, with convincing thickness, boundaries, small imperfections and natural '
      'transitions into rock and vegetation. Do not paste rectangular pieces of photographs into the scene. '
      'No overhead map textures, smeared terrain, aerial imagery patches or blurred blank ground. '
      'Image 4 contains ONLY the woman identity and clothing reference. Put that same woman at the '
      'geometric human marker, preserving its position and apparent height, with both boots planted '
      'and a natural contact shadow. Keep her face, black hair, yellow jacket, green trousers and boots. '
      'Show only this one person. Unified soft natural daylight, realistic eye-level photography, '
      'photographic depth and sharp surface detail, no text.')
    if photo_structure:
        prompt=(f'Create a realistic photograph of one woman at the Mather Point overlook. '
          f'Image {platform} is the authoritative reference for the actual built lookout: '
          'its narrow concrete pedestrian walkways and viewing areas, changes in level, ramps, '
          'steps, stone retaining walls, exposed natural rocks, and metal guardrails with mesh. '
          'Reconstruct this kind of real built space around the woman. Preserve the relationships '
          'between paths, walls, rocks and railings shown in the photograph, adapting their perspective '
          'to the target camera. Include the guardrail between the standing woman and the canyon edge. '
          f'Image {geo} is ONLY a coarse reference for distant canyon topography, horizon, camera '
          'direction and approximate central subject framing. Its entire foreground platform is an '
          'incorrect modelling placeholder: replace it using the real lookout photograph. '
          'Do not copy its broad slab, raised pavement strip, polygon edges, clay material or humanoid. '
          f'Image {canyon} supplies real canyon rock layers, erosion and atmospheric depth. '
          'Use the photographs for physical structure as well as materials and surface detail. '
          'Image 4 supplies only the woman identity and clothing: same face, black hair, yellow jacket, '
          'green trousers and boots. Place her centrally on a believable pedestrian viewing area, '
          'both boots planted, realistic contact shadow, approximately the framing of the geometric marker. '
          'Replace the geometric human marker with this ONE woman, do not add a second woman. '
          'Remove every visitor from the lookout reference photograph. The completed photograph '
          'contains exactly one human figure, no duplicates and no background people. '
          'Keep the geometric camera direction and horizon rather than copying the lookout photo viewpoint. '
          'Unified natural daylight, photorealistic ground-level photography. '
          'No overhead map textures, photographic patches, white clay or text.')
    with f._client(url) as client:
        graph={};refs=[f._load(graph,client,str(40+i),Path(path)) for i,path in enumerate(inputs)]
        f._klein(graph,100,prompt,seed,'geoblock/touchdown_topography_still',tuple(refs))
        saved=f._submit(client,graph,('107',),Path(folder))
    return str(saved['107'])


def klein_refs(url: str, folder: str, refs: list[str], prompt: str, seed: int, prefix: str) -> str:
    """One FLUX.2 klein still with any number of reference images (image 1, image 2, ... in ``refs`` order)."""
    import fakeout_earth_zoom as f
    with f._client(url) as client:
        graph: dict = {}
        loaded = tuple(f._load(graph, client, str(40 + i), Path(p)) for i, p in enumerate(refs))
        f._klein(graph, 100, prompt, seed, prefix, loaded)
        saved = f._submit(client, graph, ("107",), Path(folder))
    return str(saved["107"])


JOBS = {"ltx": ltx, "stills": stills, "mid_still": mid_still, "face_plate": face_plate, "klein_refs": klein_refs,
        "sea_masks": sea_masks, "buildings": buildings, "terrain_v2v": terrain_v2v, "backward_edit": backward_edit, "layout_portrait": layout_portrait, "topographic_still": topographic_still, "closeup_upscale": closeup_upscale, "closeup_refine": closeup_refine}

if __name__ == "__main__":
    result = JOBS[sys.argv[1]](**json.loads(sys.argv[2]))
    print(json.dumps({"result": result}), flush=True)
