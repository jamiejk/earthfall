"""Weather presets shared by the Blender blocking render and the video-model prompts.

The blocking colours steer render-to-real models as much as the prompt does (a grey clay
sky comes back grey), so each preset sets both the render's sky, sun and haze and a
sentence for the prompt.

Keys besides the sky colours, sun and haze (all optional except ``prompt`` and ``deck``):

- ``aloft``: the weather as seen from orbit or a satellite view; the ground sentence ("snow
  falling", "lightning striking the far rim") is wrong there.
- ``in_cloud``: the cloud's tint for keyframes inside the Earth zoom's cloud deck (none: plain white cloud).
- ``above``: scene text replacing the daytime orbit, satellite and in-cloud descriptions ({name}, {km}), for
  weather that changes the whole top of the zoom (night, a dust storm).
- ``globe_light``, ``shell_colour``, ``deck_colour``: Earth zoom blocking: how lit the globe is, and the colour of
  the clouds seen from orbit and of the deck flown through.
- ``whole_shot``, ``deck_word``: for the Earth zoom LTX prompt, when the weather holds from orbit to the ground.
- ``deck``: Earth zoom cloud-deck cover (0..1) and darkness, so the dive reads as weather.
- ``sun_colour``, ``ambient`` (sky-light multiplier), ``ambient_colour``.
- ``visibility``: scales the distance haze starts and ends at (dust storm: short).
- ``snow``: 0..1 snow on flat ground above the snow line.
- ``fog``: a layer of cloud lying in the low ground, below the landmark's own level.
- ``sky_clouds``: (cover 0..1, colour) of clouds painted on the camera-visible sky, so the video model keeps
  them instead of clearing the sky.
- ``stars``: a star field in the camera-visible sky.
- ``outfit``: what an invented subject wears in this weather.
"""

from __future__ import annotations

WEATHER = {
    "clear": {
        "horizon": (0.50, 0.66, 0.90), "zenith": (0.08, 0.22, 0.62), "sun_energy": 4.5, "sun_elevation": 28.0,
        "haze": 1.0, "deck": (0.35, 0.0),
        "prompt": "Clear blue sky, bright sun from the side, crisp shadows, slight distant haze.",
        "aloft": "Bright sunlight, a few scattered white clouds."},
    "partly cloudy": {
        "sky_clouds": (0.45, (0.95, 0.95, 0.97)),
        "horizon": (0.58, 0.68, 0.84), "zenith": (0.16, 0.30, 0.62), "sun_energy": 3.5, "sun_elevation": 35.0,
        "haze": 1.1, "deck": (0.55, 0.05),
        "prompt": "Blue sky with scattered white cumulus clouds, their shadows drifting across the "
                  "ground, bright sun between them.",
        "aloft": "Scattered white cumulus clouds casting shadows on the land."},
    "overcast": {
        "sky_clouds": (0.95, (0.66, 0.67, 0.70)),
        "in_cloud": "The cloud is a dull, even grey.",
        "horizon": (0.62, 0.64, 0.66), "zenith": (0.42, 0.45, 0.50), "sun_energy": 1.2, "sun_elevation": 45.0,
        "haze": 1.4, "deck": (0.85, 0.25),
        "prompt": "Overcast grey sky with a flat, even cloud layer, soft diffuse light and no hard "
                  "shadows.",
        "aloft": "A broad grey-white cloud layer covers much of the land."},
    "hazy": {
        "horizon": (0.74, 0.74, 0.72), "zenith": (0.38, 0.48, 0.66), "sun_energy": 3.0, "sun_elevation": 30.0,
        "haze": 1.9, "deck": (0.5, 0.05),
        "prompt": "Hazy summer air, pale washed-out sky, distant ridges fading into soft blue haze.",
        "aloft": "Soft pale haze over the land, a few clouds."},
    "golden hour": {
        "sky_clouds": (0.25, (1.0, 0.80, 0.62)),
        "in_cloud": "The cloud glows warm gold from the low sun.",
        "horizon": (0.95, 0.66, 0.38), "zenith": (0.20, 0.30, 0.58), "sun_energy": 3.0, "sun_elevation": 8.0,
        "haze": 1.2, "deck": (0.45, 0.0),
        "prompt": "Golden hour: low warm sun close to the horizon, long shadows, rock glowing orange, "
                  "warm sky fading to blue overhead.",
        "aloft": "Low warm evening sun, long shadows of clouds and ridges across the land."},
    "thunderstorm": {
        "sky_clouds": (0.9, (0.22, 0.23, 0.27)),
        "in_cloud": "The cloud is dark grey and turbulent, with dim cold light.",
        "horizon": (0.30, 0.32, 0.36), "zenith": (0.10, 0.11, 0.14), "sun_energy": 0.8, "sun_elevation": 40.0,
        "haze": 1.5, "deck": (0.95, 0.6),
        "prompt": "Violent thunderstorm: dark bruised storm clouds low over the canyon, forked "
                  "lightning bolts striking the far rim, grey rain shafts in the distance, dim cold "
                  "light with brief bright flashes.",
        "aloft": "Towering white thunderstorm clouds with dark shadowed bases over the land.",
        "outfit": "a hiker in a dark rain jacket with the hood down, wet hair, hiking trousers and boots"},
    "snow": {
        "horizon": (0.66, 0.76, 0.92), "zenith": (0.12, 0.28, 0.62), "sun_energy": 4.0, "sun_elevation": 22.0,
        "haze": 1.1, "deck": (0.5, 0.05), "snow": 1.0,
        "prompt": "Winter morning after fresh snowfall: clean white snow on the flat ground, the ledges and "
                  "the tops of the cliffs, bare steep rock faces showing through, crisp cold blue sky, bright "
                  "low sun, sparkling snow.",
        "aloft": "Fresh white snow covers the high ground, bare darker valleys, bright winter sunlight.",
        "outfit": "a hiker in a warm puffer jacket, beanie, gloves, winter trousers and snow boots"},
    "fog": {
        "horizon": (0.82, 0.83, 0.86), "zenith": (0.28, 0.44, 0.70), "sun_energy": 3.5, "sun_elevation": 12.0,
        "sun_colour": (1.0, 0.9, 0.78), "haze": 1.3, "deck": (0.6, 0.0),
        "fog": {"level": 0.6, "density": 0.95},
        "prompt": "Early morning cloud inversion: a sea of soft white fog fills the low ground below, with "
                  "only the high ridges and buttes rising out of it like islands, clear sky and low warm "
                  "sunlight above.",
        "aloft": "Soft white fog fills the valleys and low ground, high ridges rising out of it.",
        "outfit": "a hiker in a fleece jacket, a knitted beanie, hiking trousers and boots"},
    "dusk": {
        "sky_clouds": (0.3, (0.95, 0.62, 0.58)),
        "in_cloud": "The cloud is lit pink and orange by the setting sun, darker below.",
        "horizon": (0.98, 0.52, 0.36), "zenith": (0.10, 0.12, 0.34), "sun_energy": 1.6, "sun_elevation": 2.0,
        "sun_colour": (1.0, 0.55, 0.32), "ambient": 0.7, "ambient_colour": (0.42, 0.40, 0.62),
        "haze": 1.2, "deck": (0.45, 0.1),
        "prompt": "Dusk just after sunset: the sky glows orange and pink at the horizon, deepening to violet "
                  "and dark blue overhead, the land in soft cool shadow with the last warm light on the "
                  "highest ridges.",
        "aloft": "Sunset light: long dark shadows, clouds lit pink and orange from below.",
        "outfit": "a hiker in a warm jacket, beanie and hiking trousers"},
    "night": {
        "above": {
            "orbit": ("The night side of planet Earth seen from orbit, {km:,.0f} km up, looking down toward {name}: "
                      "dark land and black oceans faintly lit silver-blue by the moon, dim grey clouds, scattered tiny "
                      "orange city lights, a thin glowing line of atmosphere at the limb, stars in black space. It is "
                      "night: no daylight anywhere."),
            "high": ("A night satellite photograph from {km:.0f} km above the land around {name}: dark terrain faintly "
                     "lit by moonlight, a few tiny orange town lights, dim grey clouds. It is night: no daylight."),
            "cloud": ("The camera is inside a dark layer of cloud at night, high above {name}: dim blue-grey cloud, "
                      "faintly lit by moonlight, fills the frame; no ground, no horizon and no sky are visible.")},
        "in_cloud": "The cloud is dark blue-grey, faintly lit by moonlight.",
        "horizon": (0.07, 0.09, 0.18), "zenith": (0.010, 0.014, 0.040), "sun_energy": 0.35,
        "sun_elevation": 35.0, "sun_colour": (0.62, 0.72, 1.0), "ambient": 0.18,
        "ambient_colour": (0.16, 0.20, 0.36), "haze": 0.8, "deck": (0.35, 0.55), "stars": True,
        "whole_shot": "It is night the whole way down, with no daylight at any point: the night side of the Earth "
                      "from orbit, dark night clouds, then the dim blue land under the Milky Way.",
        "deck_word": "dark moonlit cloud",
        "globe_light": 0.12, "shell_colour": (0.30, 0.33, 0.40), "deck_colour": (0.22, 0.25, 0.34),
        # klein paints whatever the words name, negated or not: "no moon" painted a moon, "head torch switched
        # off" a lit head torch, "no snow" a snowy rim. So name only what should be there.
        "prompt": "Clear starry night: the Milky Way and countless stars across a deep blue-black sky, the land in "
                  "dim cool blue night light with deep shadows, every rock, soil and plant surface dark, muted "
                  "grey-blue and low in brightness, a long-exposure night photograph.",
        "aloft": "The night side of the Earth, dark land faintly lit by moonlight, a thin glowing atmosphere "
                 "and stars.",
        "outfit": "a hiker in a dark warm jacket and beanie, face softly lit by the dim blue night light"},
    "sandstorm": {
        "above": {
            "orbit": ("Planet Earth seen from orbit, {km:,.0f} km up, looking down toward {name}: a vast orange-brown "
                      "dust storm spreads over the desert around {name}, hiding the land under a tan haze, real white "
                      "cloud systems further away, a thin blue atmosphere at the limb, black space around it."),
            "high": ("A satellite photograph from {km:.0f} km above {name}: a thick orange-brown dust storm hides most "
                     "of the land, the terrain only faintly visible through the dust."),
            "cloud": ("The camera is inside a thick cloud of orange-brown dust high above {name}: swirling tan and rust "
                      "dust fills the frame, the sun a dim pale disc; no ground, no horizon and no sky are visible.")},
        "in_cloud": "The cloud is tinted brown and orange with dust.",
        "horizon": (0.74, 0.50, 0.30), "zenith": (0.52, 0.38, 0.26), "sun_energy": 1.6, "sun_elevation": 32.0,
        "sun_colour": (1.0, 0.72, 0.46), "ambient_colour": (0.62, 0.46, 0.32), "haze": 2.6,
        "visibility": 0.35, "deck": (0.8, 0.0), "shell_colour": (0.78, 0.60, 0.42),
        "deck_colour": (0.74, 0.54, 0.36),
        "whole_shot": "A dust storm covers the region the whole way down: tan dust over the land seen from orbit, "
                      "a dive through brown dust, then the dust-filled air at the ground.",
        "deck_word": "orange-brown dust",
        "prompt": "A desert dust storm: thick orange-brown dust fills the air, a towering wall of dust rolls "
                  "in across the land, the sun a pale dim disc, the distance lost in the dust, dull warm light "
                  "and no blue in the sky.",
        "aloft": "A vast brown dust storm spreads across the land below.",
        "outfit": "a hiker with a scarf over the mouth and nose, sunglasses, a light jacket and boots"},
}
DEFAULT_WEATHER = "clear"


def weather_prompt(name: str, situation: dict | None = None) -> str:
    """The weather sentence for a prompt: from orbit or a satellite view, the weather seen from above; inside the
    cloud deck, only the cloud's tint."""
    weather = WEATHER[name]
    if situation and situation.get("over") == "cloud":  # no sky or ground in view, only the cloud's tint
        return weather.get("in_cloud", "")
    if situation and situation.get("over") in ("orbit", "high"):
        return weather.get("aloft", weather["prompt"])
    return weather["prompt"]
