"""Words for what a place looks like, matched to the blocking palette.

Render-to-real models repaint the blocking with what the prompt says the place is made of,
so each palette carries a description of the real ground. Keep it to materials, vegetation
and landforms: the camera, framing and weather come from elsewhere.
"""

from __future__ import annotations

import math

LOOKS = {
    "default": {
        "all": "real rock, soil and vegetation of {name}, with natural colour and texture",
        "high": "the ground near the camera is {name}'s natural rock, soil and vegetation",
        "low": "the land far below is {name}'s natural rock, soil and vegetation",
        "far": "the distant land fades naturally toward the horizon"},
    "grand-canyon": {
        "all": (
            "the South Rim plateau is natural pinyon-juniper woodland: irregular, windswept trees of many sizes, "
            "some close together, with grasses, sagebrush and pale limestone ground between them. Below the rim, "
            "eroded buttes and temples are banded in red Supai and Redwall rock, buff Coconino sandstone and cream "
            "Kaibab limestone caps, with grey-green talus slopes, sparse shrubs and deep shadowed side canyons. The "
            "flat forested North Rim runs level along the far horizon"),
        "high": (
            "the flat ground is the South Rim plateau, natural pinyon-juniper woodland: irregular, windswept trees "
            "of many sizes, some close together, with grasses, sagebrush and pale limestone ground between them"),
        "low": (
            "far below, eroded buttes and temples are banded in red Supai and Redwall rock, buff Coconino sandstone "
            "and cream Kaibab limestone caps, with grey-green talus slopes, sparse shrubs and deep shadowed side "
            "canyons"),
        "far": "the canyon's far side shows as flat-topped buttes and the level forested North Rim on the horizon",
        "trees": ("pinyon pines and junipers", "ponderosa pines"),
        "ground": "grasses, sagebrush and pale limestone ground"},
}
TALL_TREE_M = 12.0  # the proxies switch from rounded crowns to conifer cones here (trees.CONIFER_FROM_M)


OPEN_COVER = 0.05  # below this the ground reads as open, not woodland
SPARSE_COVER = 0.2  # around a viewpoint, below this: a few trees, no species or heights


def canopy_text(summary: dict, palette: str = "default", ground: str = "") -> str:
    """The measured trees, in words, for prompts: ``summary`` from ``trees.canopy_summary``.

    The keyframe painter repaints what the prompt says the place is made of, so a stock "sparse woodland"
    line would thin out the measured forest in Image 1 while LTX follows the layout: describe what was measured.
    ``ground``, the ground between the trees measured on aerial photos where there are some, replaces the
    palette's stock words.
    """
    short, tall = LOOKS.get(palette, {}).get("trees", ("trees", "tall conifers"))
    ground = ground or LOOKS.get(palette, {}).get("ground", "grass, shrubs and bare ground")
    cover = summary["cover"]
    where = f"the ground within {summary['around_m']:.0f} m of the viewpoint" if "around_m" in summary else "the flat ground"
    keep = (" Each dark green shape in Image 1 marks where one real tree stands and how big it is: paint a natural, "
            "irregular tree of that size in each place, with real branches and foliage rather than the simple shape, "
            "and do not thin them out or add trees where Image 1 has open ground")
    if cover < OPEN_COVER:  # say plainly that it is open: a tree list makes the painter plant a forest
        trees = "no trees" if not summary["count"] else f"only {summary['count']} scattered trees"
        return f"{where} is open ground, {ground}, with {trees} (measured) and no other trees anywhere near.{keep}"
    if "around_m" in summary and cover < SPARSE_COVER:
        # Around a viewpoint, species and "up to 17 m" made the painter fill every open patch with tall pines.
        return (f"{where} is mostly open ground, {ground}, with just a few small trees and bushes (measured: they "
                f"cover about {100 * cover:.0f}% of it) and no other trees.{keep}")
    density = ("is dense woodland" if cover >= 0.45 else "is woodland" if cover >= 0.2 else "has open, scattered trees")
    low, high = summary["height_p25_m"], summary["height_p75_m"]
    mix = f"mostly {short} about {low:.0f}–{high:.0f} m tall"
    if summary["tall_share"] >= 0.03:
        mix += f", with {tall} up to {summary['height_p95_m']:.0f} m among them"
    return (f"{where} {density} (measured: trees cover about {100 * cover:.0f}% of it), {mix}, "
            f"and {ground} between them.{keep}")



def frame_situation(scene: dict, route: dict, frame: int) -> dict:
    """Where the camera is at ``frame``: height above the ground below it, and whether that ground is the
    landmark's own level (on top) or well below it (out over a drop)."""
    import rasterio
    frame = max(0, min(frame, len(route["positions"]) - 1))
    x, y, z = route["positions"][frame]
    if z - scene["target_height_m"] > 20_000:  # Earth zoom: far above any measured terrain
        altitude = z
        return {"frame": frame, "ground_m": None, "above_ground_m": round(altitude, 1),
                "over": "orbit" if altitude > 300_000 else "high"}
    if route.get("kind") == "earth_zoom":
        from .zoom import cloud_deck_m
        base, top = cloud_deck_m(scene)
        if base <= z <= top + (top - base) / 2:  # just above the top, the stacked sheets already read as a whiteout
            return {"frame": frame, "ground_m": None, "above_ground_m": round(z - scene["target_height_m"], 1),
                    "over": "cloud"}
    east, north = scene["origin_projected_m"]
    with rasterio.open(scene["terrain_raster"]) as dem:
        ground = float(next(dem.sample([(east + x, north + y)]))[0])
    low, high = scene["elevation_range_m"]
    on_top = ground >= scene["target_height_m"] - 0.05 * (high - low)
    situation = {"frame": frame, "ground_m": round(ground, 1), "above_ground_m": round(z - ground, 1),
                 "over": "top" if on_top else "drop"}
    if route.get("kind") == "earth_zoom" and frame >= route.get("arrive_frame", len(route["positions"])):
        sx, sy = route.get("subject_local_m", (0.0, 0.0))  # the subject stands on the landmark
        situation["subject_distance_m"] = round(math.hypot(x - sx, y - sy), 1)
    return situation


def look_text(palette: str, name: str, part: str = "all", look: dict | None = None, canopy: str = "") -> str:
    """A curated palette's words, else the place's measured look (landcover.look_from_landcover), else generic.

    ``canopy`` (``canopy_text``, when the blocking has measured tree proxies) replaces the stock words for
    the ground near the camera.
    """
    if canopy and part == "high":
        return canopy
    if canopy and part == "all":
        return (f"{canopy}. {look_text(palette, name, 'low', look)}. {look_text(palette, name, 'far', look)}"
                .split(" Each dark green shape in Image 1")[0] + " The trees keep their places and sizes as the camera moves")
    if palette in LOOKS and palette != "default":
        return LOOKS[palette][part].format(name=name)
    return (look or LOOKS["default"])[part].format(name=name)


def scene_palette(palette: str, scene: dict | None) -> str:
    """The node's palette, or the catalog place's own one (e.g. grand-canyon) when the node is left on default."""
    if palette == "default" and scene and scene.get("palette") in LOOKS:
        return scene["palette"]
    return palette


DEFAULT_SUBJECT = "a hiker in a light jacket, cargo trousers and a cap, with a small day pack"
SHEET_SETTING = ("standing upright and relaxed, facing the camera, the whole body from head to feet in frame and "
                 "centred, in front of a plain light grey studio wall, soft even light")
# For GeoBlockPlaceSubject, which cuts the person out of the sheet: a light grey jacket keyed out along with a
# light grey wall (Twin Peaks r6k), so the wall is a colour no outfit is.
# Pasted from a sheet posed "upright, facing the camera", the hiker stood square with arms straight down, stiff
# beside the real scene (r18-r20k): the chroma sheet asks for a natural stance instead.
CHROMA_SHEET_SETTING = ("standing in a natural, relaxed stance with the weight on one leg, one hand resting on a "
                        "backpack strap, body turned a little to one side and face toward the camera, the whole "
                        "body from head to feet in frame "
                        "and centred, in front of a plain flat bright green chroma-key studio wall, soft even white "
                        "light, no green light on the person")


def default_subject(weather: str | None = None) -> str:
    """An invented person dressed for the weather."""
    from .weather import WEATHER
    return WEATHER.get(weather or "", {}).get("outfit", DEFAULT_SUBJECT)


def sheet_light(weather: str | None) -> str:
    """The sheet person's light, as the landing will light them. Lit evenly in the studio, the pasted hiker stayed
    flat against the summit's side sun (Twin Peaks r16, r18k). The Earth zoom's sun stands 100 degrees clockwise of
    the route heading (blender_earth_zoom), i.e. to the right of and a little behind the hold camera, which the
    person faces: their left side, on the right as we look at them, takes the sun."""
    from .weather import WEATHER, DEFAULT_WEATHER
    look = WEATHER.get(weather or DEFAULT_WEATHER, WEATHER[DEFAULT_WEATHER])
    if look.get("sun_energy", 0) < 2.5 or look.get("sun_elevation", 0) >= 60:
        return "soft daylight from above, as under a cloudy sky"
    return (f"hard direct sunlight from the right, about {look['sun_elevation']:.0f} degrees above the horizon "
            "and level with them: the right side of the person as we look at them is sunlit, the left side in "
            "soft shadow, with a crisp light edge along the right of the face, jacket and trousers")


def subject_sheet_prompt(description: str, photo: bool, chroma: bool = False, weather: str | None = None) -> str:
    """klein prompt for the subject sheet: the person alone, full length, on a plain background. Every eye-level
    keyframe is given this one picture, so the person matches across them, and nothing from the background of the
    user's photo (or a head-and-shoulders crop of it) leaks into the scene. ``chroma`` puts them on a green
    wall, for cutting them out."""
    setting = CHROMA_SHEET_SETTING if chroma else SHEET_SETTING
    if chroma:  # cut out and pasted into the scene: lit as the scene lights them, the wall kept even for the key
        setting = setting.replace("soft even white light", f"lit by {sheet_light(weather)}; the green wall evenly "
                                  "lit on its own, with no shadow on it")
    if photo:
        notes = f" Notes about them: {description}." if description else ""
        return ("Image 1 shows a person. Make a new full-length photograph of exactly this person: the same face, "
                "hair, skin tone and body, and the same clothes, shoes and accessories, " + setting + ". Only "
                "this one person; nothing from Image 1's background." + notes + " Sharp natural photographic "
                "detail, real skin and fabric.")
    return (f"A full-length studio photograph of one person, {setting}: {description}. Sharp natural "
            "photographic detail, real skin and fabric.")


def subject_sentence(subject: dict | None, situation: dict | None) -> str:
    """The person at the landmark, for frames where the camera has come down to them."""
    if situation and situation.get("plate") and situation.get("subject_distance_m") is not None:
        # The empty plate a person is pasted onto afterwards: klein grew a bush exactly where their feet go
        # (Twin Peaks r9k), so the spot is asked for clear.
        # "Nobody is in the picture ... open and walkable" got a stranger on the path instead (r10k).
        # Naming the spot where the person will stand drew a stranger to it, whatever the wording (r10k-r13k):
        # a man on the path, then one walking away right under the hiker. So the spot isn't named; the
        # featureless near foreground of the blocking is described as ground instead. Fixed words for it ("grass,
        # soil and small stones") turned Twin Peaks' summit road into a dirt path, and LTX, following the blocking's
        # road, morphed one into the other as it landed (r16 f96-108). So the words are the measured ground there.
        ground = situation.get("foreground") or "the ground Image 1 shows there"
        return ("An empty landscape with no people and no animals anywhere in it. The near foreground is "
                f"{ground}, seen close up in sharp sunlit detail and laid out exactly as in Image 1: a road stays "
                "a road with its edges and markings, a path stays a path. ")
    if not subject or not situation or situation.get("subject_distance_m") is None:
        return ""
    photo = "Image 3" if situation.get("reference") else "Image 2"
    if subject.get("image", subject.get("photo")):  # the user's photo, or the subject sheet
        hint = f" ({subject['description']})" if subject.get("description") and not subject.get("photo") else ""
        who = f"the person shown in {photo}{hint}, with the same face, hair, body and clothes"
    else:
        who = subject.get("description") or DEFAULT_SUBJECT
    distance = situation["subject_distance_m"]
    framing = ("exactly the size and crop of the grey stand-in figure in Image 1, cut off where it is cut off; do "
               "not shrink the person to show more of them" if distance < 6 else "whole body visible")
    if situation.get("placed"):  # GeoBlockPlaceSubject pasted them in at the stand-in's size
        return (f"The person in Image 1 already stands exactly where and as large as they must be, about "
                f"{distance:.0f} m in front of the camera at the viewpoint: keep them at exactly that position and "
                "size, cut off where they are cut off, facing the camera, natural relaxed pose, and make them a real "
                f"photograph of {who}, lit like the scene. Only this person; the rest of Image 1 stays as it is. ")
    return (f"One person stands where the grey stand-in figure is in Image 1, about {distance:.0f} m in front of "
            f"the camera at the viewpoint, facing the camera, {framing}, natural relaxed pose: {who}. Replace the "
            "stand-in with this real person; only this person, and the rest of Image 1 stays as it is. ")


# The Earth zoom's cloud keyframe once asked for cloud "brighter toward the sun": klein painted a sun with a ray,
# and LTX carried that streak across f12-44 of every place. Every keyframe of a planned shot asks for none.
NO_GLARE = "No sun disc, no lens flare, no light rays or streaks."


def keyframe_prompt(name: str, palette: str, weather_sentence: str, situation: dict | None = None,
                    subject: dict | None = None, look: dict | None = None, weather: str | None = None,
                    canopy: str = "") -> str:
    """Qwen-Image-Edit prompt that repaints one blocking frame as a photograph without moving anything.

    Describe only what the frame can contain: a whole-place description makes Qwen add a cliff edge to a
    plateau frame, or foreground rim trees to a frame out over the canyon.
    """
    person = subject_sentence(subject, situation)
    reference = situation and situation.get("reference")
    reference_sentence = (
        "Image 2 is the ground this camera reaches a moment later, seen from lower down: give the ground the same "
        "colour, rock, soil and vegetation as Image 2, at this frame's own height and framing. "
        if reference == "descent" else
        "Image 2 is the same spot a moment earlier, from further back: keep its trees, bushes, rocks, "
        f"ground, background{'' if situation.get('plate') else ' and person'} exactly the same, only seen from "
        "closer. " if reference else "")
    from .weather import WEATHER
    above = WEATHER.get(weather or "", {}).get("above", {})
    if situation is not None and situation["over"] in above:  # night or dust over the whole top of the zoom
        scene_text = above[situation["over"]].format(name=name, km=(situation.get("above_ground_m") or 0) / 1000)
    elif situation is None:
        scene_text = f"The place is {name}: {look_text(palette, name, look=look, canopy=canopy)}."
    elif situation["over"] == "orbit":
        scene_text = (f"The planet Earth seen from orbit, {situation['above_ground_m'] / 1000:,.0f} km up, looking "
                      f"straight down toward {name}: real oceans, deserts, forests and mountain ranges under real "
                      "white cloud systems, a thin blue atmosphere at the limb, black space around it.")
    elif situation["over"] == "high":
        scene_text = (f"A satellite photograph looking straight down from {situation['above_ground_m'] / 1000:.0f} "
                      f"km above the land around {name}: "
                      + (f"{look_text(palette, name, 'low', look=look)}, in natural colour"
                         if look and palette in ("default", "") else
                         "real terrain, river canyons, forest and desert in natural colour")
                      + ", with scattered real clouds.")
    elif situation["over"] == "cloud":
        scene_text = (f"The camera is inside a thick layer of white cloud high above {name}: soft white and pale "
                      "grey cloud fills the frame, evenly lit, with faint darker billows; no ground, "
                      "no horizon and no sky are visible through it.")
    elif situation["over"] == "top":
        scene_text = (f"The camera is {situation['above_ground_m']:.0f} m above flat ground at {name}: "
                      f"{look_text(palette, name, 'high', look=look, canopy=canopy)}. {look_text(palette, name, 'far', look=look)}.")
    else:
        scene_text = (f"The camera is in the air at {name}, {situation['above_ground_m']:.0f} m above the ground "
                      f"below it, with no foreground edge or trees unless Image 1 has them: "
                      f"{look_text(palette, name, 'low', look=look)}. {look_text(palette, name, 'far', look=look)}.")
    plate = bool(situation and situation.get("plate"))
    return (
        "Image 1 is the only source of truth for geometry, composition and camera. Repaint its surfaces only as a "
        "real photograph; do not reframe, recenter, zoom, or move the horizon or any ridge. Add nothing that is "
        # A plate's sentence is about the empty ground; "except the person described" there invited a walker.
        f"not in Image 1{', except the person described' if person and not plate else ''}. {scene_text} {person}"
        f"{reference_sentence}{weather_sentence}"
        f"{' ' + NO_GLARE if situation is not None else ''}"
        " Sharp natural photographic detail.")


BUILDING_CUE = " Each pale block in Image 1"  # buildings.city_text: keyframe painters only, LTX has no Image 1


def without_image_cues(text: str) -> str:
    """A look for the video prompt: the keyframe painters' Image 1 instructions about building blocks become a
    plain statement that the buildings hold still."""
    if BUILDING_CUE not in text:
        return text
    head, tail = text.split(BUILDING_CUE, 1)
    ends = [i for i in (tail.find(". "), tail.find("; ")) if i >= 0]
    rest = tail[min(ends):] if ends else ""
    return head + " The buildings keep their places and heights as the camera moves" + rest


def render_to_real_prompt(name: str, palette: str, preset_description: str, weather_sentence: str,
                          preset: str = "", look: dict | None = None, weather: str | None = None,
                          canopy: str = "") -> str:
    """3DREAL prompt: trigger word, the camera action, the place's materials, then the light."""
    if preset == "Earth zoom":  # ground-level words here put trees on the land seen from orbit
        from .weather import WEATHER
        top = WEATHER.get(weather or "", {}).get("whole_shot")
        top = top + " " if top else ""
        deck = WEATHER[weather].get("deck_word", "white clouds") if top else "white clouds"
        underfoot = f" on {look['underfoot']}" if look and look.get("underfoot") and not canopy else ""
        return (
            f"3DREAL. Make it photorealistic. {top}One continuous zoom from space: the whole Earth in orbit with real "
            f"cloud systems, then straight down as satellite views of the land sharpen, through a layer of "
            f"{deck}, then a fast forward swoop low over the ground to eye level at {name}, where "
            + without_image_cues(look_text(palette, name, 'all', look=look, canopy=canopy) if canopy else
                                 f"{look_text(palette, name, 'low', look=look)}; {look_text(palette, name, 'far', look=look)}")
            + ". The camera then pushes in slowly "
            f"on one person standing at the viewpoint{underfoot}, who shifts their weight, glances out across the view to one "
            f"side, then turns back toward the camera. {weather_sentence} Real satellite and aerial photography "
            f"becoming handheld photography, natural colour, no text. {NO_GLARE}")
    return (
        f"3DREAL. Make it photorealistic. A single smooth aerial drone shot at {name}. {preset_description} "
        f"In the real place {without_image_cues(look_text(palette, name, look=look, canopy=canopy))}. {weather_sentence} Real photographic detail, natural "
        "colour. No people, no buildings, no text.")
