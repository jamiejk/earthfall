"""Fresh upstream scene preparation for the grounded canonical recipe.

Uses the photographic scene prompts before the experimental strict phase/30,000ft
configuration. Flight settings are supplied by the frozen canonical preset.
"""
from pathlib import Path
import re
from touchdown.studio_brief import (FRAMINGS, FRAMING_LIMITS, POSES, LENS, MODEL, MODEL_DIGEST, allowed_supports, vision)

# Where the near portrait's camera comes to rest, as a positive end state (see the near prompt).
NEAR_STOPS = {
    "full": "a full-length portrait framed from just below the shoes to comfortably above the head",
    "three-quarter": "a medium-long three-quarter portrait framed from just below the knees to comfortably above the head",
    "medium": "a medium portrait framed from just below the waist to comfortably above the head",
    "close": "a head-and-shoulders portrait framed from mid-chest to comfortably above the head",
}
# What the person does after the camera stops, in time order (POSES are written for the stills' wording).
NEAR_ACTIONS = {
    "relaxed": "the subject breathes gently, blinks once and gives a small, warm smile, arms relaxed at their sides.",
    "model": "the subject settles one hand on a hip and turns their shoulders slightly toward the camera, holding a natural pose.",
    "look-back": "the subject, who has been looking toward the landmark, turns their head and shoulders back toward the camera with a small smile.",
    "wave": "the subject shifts their weight slightly, raises one hand to chest height for a brief, natural greeting wave, and lowers the arm back to their side as a warm smile forms.",
    "arms-open": "the subject opens both arms naturally out to the sides in a welcoming gesture and holds them comfortably open.",
    "twirl": "the subject makes one complete turn in place on the same spot and finishes facing the camera with a small smile.",
}

def scene_config(setup, brief, folder: Path, root: Path, seed: int):
    near_frames = 97 if setup["ending"] == "twirl" else 73 if setup["ending"] in ("wave", "arms-open") else 49
    refs = []
    for i in brief["selected_references"]:
        p = setup["references"][i]
        refs.append({**p, "local_path": str(folder / p["name"])})
    # Where the person stands comes only from the chosen support. A setting sentence that also places
    # them ("stands on a grassy patch") overrode it and failed the support check (Machu Picchu, 24 Oct).
    place = " ".join(s for s in re.split(r"(?<=[.!?])\s+", brief["setting"])
                     if not re.search(r"\b(stands?|standing|person|woman|man|figure)\b", s, re.I))
    setting = f"{setup['destination']['title']}. {place}".strip()
    support = brief["support_description"]
    appearance = brief["appearance"]
    light = brief["light"]
    subject_index = len(refs) + 1
    regional_label = ", ".join(filter(None, [setup["destination"].get("region"), setup["destination"].get("country")])) or "the reference-derived region"
    # A small captioner previously put a landmark "visible from orbit" into
    # this local field. Such a sentence must not leak into any camera stage.
    landscape = " ".join(s for s in re.split(r"(?<=[.!?])\s+", brief["surrounding_landscape"])
                         if not re.search(r"\b(orbit|planet|globe|space|continental)\b", s, re.I))
    landscape = landscape or f"Plausible connected surroundings in {regional_label} with the materials of the location photographs."
    if brief.get("local_features"):  # OpenStreetMap land cover and water around the coordinates
        # replaces Qwen's guess, which invents regions ("typical of the New Forest" for Highclere, 29 Oct)
        landscape = brief["local_features"]
    arrival = (
        f"Create ONE integrated modern real travel photograph at {setting}. Images 1–{len(refs)} are real location references. "
        f"Image {subject_index} identifies the person: preserve their face and hair. Expected appearance: {appearance}. "
        # A raised camera: the reversed portrait then starts the flight already climbing. From eye level
        # the retreat skimmed flat along the ground and never reached an aerial (Teotihuacan, 24 Oct).
        "The camera is RAISED about five metres above the ground and a few metres back, looking DOWN at the "
        "person at about 30 degrees, like a drone hovering low. A full-body environmental portrait slightly right "
        "of centre, body about half of the image height, the ground visible all around their feet. "
        "Show the complete head, BOTH ACTUAL SHOES and a clear band of ground beneath both soles. "
        f"Both feet stand firmly on {support}, continuing visibly into the same ground-level forecourt around the landmark. "
        # Lake Bled: a paved strip with water on both sides read as a platform in the middle of the lake.
        "That ground is solid land: it runs out to the bottom and at least one side edge of the frame and joins the land the landmark stands on. "
        "Keep the whole recognisable landmark clear beyond the person, preferably to the left. "
        f"Expand plausible small surroundings from the references: {landscape}. {light}. "
        # The ending can only move what the still contains: Lake Bled's clear blue sky gave the drift line
        # nothing to drift (background motion equal to Sydney's).
        "Scattered large white cumulus clouds with soft shadows in the sky above the scene. "
        "35mm live-action travel photography from that raised viewpoint, sharp face, fabric and natural materials, consistent scale and perspective. "
        "The person and the place share one exposure and light. Initially relaxed arms. No pasted cutout, no pedestal, "
        "no roof, no raised slab or terrace beneath the person, no standing on the monument, no cropped feet, "
        "no invented columns, no illustration, no CGI or 3D render, no text. Exact topography is not required.")
    # Image-to-video "delta-only" prompt (Jamie, 29 Oct): describe only what changes, in time order, in
    # whole sentences. The image already shows the place and the person, and pasting the captioner's
    # descriptions mid-sentence ("on the same patch of The village is...") garbled the one line that said
    # what moves. Rules phrased as "never ... chest-up" put weight on "chest-up" and pulled the camera in;
    # the stop is described as the positive end state instead. A negative prompt cannot carry them: the
    # distilled sampler runs at CFG 1, where the negative conditioning has no effect.
    near = ("A continuous live-action shot on a rectilinear 35 mm lens with deep focus, in the same light as the photograph. "
        "Starting from the initial high angle, the camera glides smoothly downward and slightly forward on a crane, "
        "descending to eye level while keeping a generous distance, and comes to a complete, steady stop in "
        f"{NEAR_STOPS[setup['framing']]}. "
        "Throughout the descent a fresh breeze blows steadily from left to right, rustling the surrounding grass, "
        "plants and tree leaves and gently stirring the subject's hair and clothes. "
        f"Once the camera settles, {NEAR_ACTIONS[setup['ending']]} "
        "The scene is silent apart from the wind, with no music and no speech.")
    flight = ("Continue the supplied photographic camera retreat into one uninterrupted fast aerial flight. "
        f"Preserve all supplied moving portrait frames. Their forward-played ending must perform: {POSES[setup['ending']]} "
        f"The person remains on the SAME connected {support}. Preserve {appearance}. "
        "After the supplied prefix begin a SLOW, LOW retreat BACKWARD AND UPWARD over OPEN GROUND, "
        f"revealing {setting}. Keep the landmark on the same side with a single consistent heading. "
        "Keep the ground under the person connected continuously to the wider forecourt; never substitute a roof or raised platform. "
        f"Expand the surroundings coherently: {landscape}. "
        # The delivered film runs this retreat backwards, so the frames nearest the portrait are the
        # approach the viewer sees last and judges most closely. Leaving them half a second long is what
        # made the landmark indistinct: the model spent 139 of 171 pre-portrait frames above the cloud.
        "The FIRST HALF of this clip, after the supplied portrait prefix, is a slow retreat over the landmark: "
        "the person shrinks to a small figure while the monument, its masonry and its ground remain LARGE in "
        "frame, sharply resolved and clearly legible, with no blur, smearing or softening of the stone. "
        "Only in the SECOND HALF does the person become a tiny, unresolvable speck. "
        "Then climb to TWO KILOMETRES above ground, with a true oblique 45-degree HIGH AERIAL view over several kilometres. "
        "Establish a wide regional view, not an enlarged monument. No person can be resolved there. "
        "Continue climbing for several more kilometres BEFORE entering cloud. Keep rising quickly, pass briefly INSIDE "
        "moving dense white cloud which fills the lens, emerge above the cloud tops and continue climbing to "
        f"the curved blue atmospheric limb of Earth against black space over the broad region of {regional_label}. "
        "At orbital scale the local landmark, stones, people and individual buildings are microscopic and cannot "
        "be resolved. Show only continental land, distant water where regionally appropriate, large weather systems "
        "and atmospheric haze. Never enlarge the local landmark into a planetary feature. "
        f"Same materials, land colours and sunlight at every scale: {light}. "
        "One fixed-focal-length physical camera flight. No optical zoom, no camera orbit, no hovering, no "
        "crossfade or dissolve, no morphing architecture, no surface replacement, no repeated giant landmark.")
    regional_patterns = {
        "farmland": "a vast patchwork of irregular agricultural fields, hedgerows, small woods and narrow country roads",
        "urban": "a broad connected city of many small urban blocks, streets, parks and distant roofs",
        "desert": "a broad arid landscape of branching dry drainage, sand, rock and sparse scrub",
        "coast": "a broad coastline, inland terrain and open water, with a natural irregular shore",
        "forest": "extensive forest canopy, clearings, small drainage channels and winding tracks",
        "mountains": "a broad connected mountain region, small ridges, valleys and drainage networks",
        "mixed": "a broad regional mosaic of connected terrain, vegetation, small roads and drainage",
    }
    kind = brief.get("regional_kind", "mixed")
    # Qwen calls inland English estates "coast" (Castle Howard, Blenheim, 29 Oct) and every flight then
    # drew a sea shore below the cloud. The measured sea around the coordinates decides instead.
    if kind == "coast" and "local_coastal" in brief and not brief["local_coastal"]:
        cover = brief.get("local_cover") or []
        kind = {"woodland": "forest", "farmland": "farmland", "meadows": "farmland",
                "village houses": "mixed"}.get(cover[0] if cover else "", "mixed")
    regional_pattern = regional_patterns[kind]
    palette = ", ".join(c for c in brief.get("regional_colours", ["green", "brown"])
                        if c != "blue" or brief.get("local_coastal", True)) or "green, brown"
    reliefs = {"high_mountains": "high mountain ranges with snow or bare ridges, deep shadowed valleys and branching river systems",
              "hills": "rolling hills cut by branching river valleys",
              "plateau_and_canyons": "a high rugged plateau of eroded ridges, escarpments and branching valleys, with strong relief shadows",
              "plain": "a broad low plain with meandering rivers",
              "coastal_lowland": "a low coastline with river mouths, estuaries and open sea"}
    relief = reliefs[brief.get("regional_relief", "plain")]
    # The high still shows the ~25 km around the place; its relief is measured there, the orbit's over the region.
    local_relief = reliefs[brief.get("local_relief") or brief.get("regional_relief", "plain")]
    settlement = {"large_city": "Near the centre, the city is only a small soft grey-brown smudge of urban sprawl, under a tenth "
                  "of the picture width, with faint radiating roads; no street or building can be resolved.",
                  "town": "Near the centre a small town is a faint grey speck with a thread of road; no building can be resolved.",
                  "none": ""}[brief.get("nearest_settlement", "none")]
    coast = " The sea coast runs along one side, with open blue water beyond it."
    orbit_coast = coast if brief.get("coastal") else ""
    # The high still shows ~25 km, so it gets the coast only when the sea is that close (Granada is 50 km inland).
    coast = coast if brief.get("local_coastal", brief.get("coastal")) else ""
    near_settlement = {"large_city": "Part of the view is the surrounding city's dense streets and roofs, with no single building enlarged. ",
                       "town": "A small town's roofs sit in one part of the view. ", "none": ""}[brief.get("nearest_settlement", "none")]
    # Wider views inherit terrain and palette, not a close-up composition.
    high = ("Create one convincing real high-altitude aerial photograph. "
        "The camera is TWO KILOMETRES ABOVE THE GROUND, looking DOWN diagonally at 45 degrees. Show several kilometres "
        f"of continuous landscape: {regional_pattern}, set in {local_relief}.{coast} Muted natural {palette} regional colours. {light}. "
        f"{near_settlement}{(brief.get('local_features') or '') + ' ' if brief.get('local_features') else ''}"
        "This is the first broad view BELOW the cloud layer. Nearby thin moving cloud wisps frame the upper corners, "
        "with the wide landscape clearly visible far underneath. Terrain extends to every edge; no eye-level horizon. "
        "The local landmark is microscopic, less than one percent of picture width, lost naturally in the wider setting. "
        "Individual people are invisible. Broad connected patterns, atmospheric depth, natural irregular detail. "
        "No foreground person, no portrait, no enlarged monument, no giant stone circle, no sky filling half the frame, "
        "no model village, no miniature diorama, no rendered terrain, no illustration, no labels or text.")
    orbit = (f"Create a natural photograph from low Earth orbit, FOUR HUNDRED KILOMETRES above the broad region of {regional_label}. "
        f"The land below is {relief}, in muted natural {palette} colours.{orbit_coast} {settlement} "
        
        "A vast curved Earth surface, thin blue atmospheric limb, black space above, natural broad white weather systems "
        f"over regional land and distant water where geographically plausible. {light}. "
        "Show hundreds of kilometres of regional land, gentle large-scale drainage patterns and blue atmospheric haze. "
        "No individual buildings, no monuments, no stones or stone circles, no person, no local landscape "
        "pasted onto the globe or floating above it. One photographic planet, restrained natural colour, no text, no CGI.")
    cfg = dict(place=setup["destination"]["title"], subject=str(folder / setup["subject"]["name"]), references=refs,
        regional_context=str(root / "grounding.json"),
        seed=seed, models=dict(unet="flux3-image-klein-int8-convrot.safetensors", clip="qwen3vl_4b_int8_convrot.safetensors", vae="flux2-vae.safetensors"),
        video=dict(near_frames=near_frames, near_conditioning='native', frames=217+near_frames-49, scale_anchors=True,
                   pin_high=False,
                   # Flight refinement is OFF on purpose. The portrait stage is pinned to 1280x736 and
                   # refines to 2560x1472, and delivery refuses a flight of any other aspect, so the
                   # flight base can only be 1280x736 -- which refines to 2560x1472 and runs the 24 GB
                   # card out of memory on a 241-frame take. 960x544 refines fine to 1920x1088 but is
                   # 1.765 against the portrait's 1.739 and is rejected at the join. The delivered
                   # approach is therefore a Lanczos enlargement, and sharpening it is a delivery-time
                   # detail pass, not a generation setting.
                   flight_refine=False, target_seconds=(8.0, 10.5),
                   high_frame=near_frames+48, flight_width=1280, flight_height=736, preserve_native_portrait=True,
                   # Two generations instead of one ground-to-orbit take: see touchdown/split_flight.py.
                   flight_mode="split", framing=setup["framing"]),
        words=dict(region=regional_label, pattern=regional_pattern, palette=palette, light=light,
                   landscape=landscape, support=support, appearance=appearance),
        gate=dict(checker_version=vision.VERSION, description=f"{appearance}. At {setting}, standing on {support}. Both actual shoes touch the ground.",
                  arrival_report=str(root / "gates/arrival/report.json"), model=MODEL, model_digest=MODEL_DIGEST,
                  checks=["foot_contact", "support_context", "appearance", "scene_match"],
                  allowed_supports=allowed_supports(brief["support"])),
        prompts=dict(arrival=f"{arrival} {LENS}", near=near, prefix_flight=f"{flight} {LENS}",
                     high=f"{high} {LENS}", orbit=orbit))
    return cfg
