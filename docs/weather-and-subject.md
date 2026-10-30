# Weather and your own character

*Branch `claude/weather-character-4wkecs`, 6 October 2026. Blocking checked on CPU (Blender 5.0 bpy, Cycles);
the klein and LTX steps still need the desktop GPU.*

## Weather

`--weather` (and the Render Blocking Video, Keyframe Prompt and Subject nodes) now offers eleven presets: the
original clear, partly cloudy, overcast, hazy, golden hour and thunderstorm, plus:

| Preset | In the Blender blocking | In the prompts |
|---|---|---|
| snow | white snow on ground flatter than ~40° above the middle of the height range, in patches; steep faces stay bare | fresh snow on the flats and ledges, bare cliffs, cold blue sky |
| fog | two sheets of low cloud in the low ground, 60% of the way from the lowest ground to the landmark's level, so the rim and high buttes stand clear (a cloud inversion) | a sea of fog in the canyon, ridges rising out of it like islands |
| dusk | sun 2° up and orange, pink-orange horizon, violet-blue zenith, less sky light | sky glowing orange and pink at the horizon, land in cool shadow |
| night | dim blue moonlight, navy sky with a star field | full moon, stars and the Milky Way, silver-blue land |
| sandstorm | orange-brown sky and sky light, haze at a third of the usual distance | thick orange-brown dust, a dust wall, a pale sun disc |

Partly cloudy, overcast, thunderstorm, golden hour and dusk now paint clouds on the blocking's sky
(`sky_clouds`), shrinking toward the horizon like a real cloud ceiling
(`review/weather-character/blocking_sky_clouds.jpg`). Before this the sky was a bare gradient and the prompt alone
had to supply the clouds; the Higgsfield examples nearly all have a textured sky.

Every preset now also has an `aloft` sentence for orbit and satellite keyframes (no lightning striking the far rim
when seen from 400 km up) and an `in_cloud` tint for the cloud-whiteout keyframe. The Earth zoom cloud-deck cover
moved into the presets. An invented subject is dressed for the weather (puffer jacket in snow, scarf over the mouth
in a sandstorm, head torch at night). Editing a preset's values now re-renders the blocking instead of reusing it.

Blocking previews (CPU, 640x360):

- `review/weather-character/blocking_weather_existing.jpg` and `blocking_weather_new.jpg`: 5 s rim reveal.
- `review/weather-character/earthzoom_blocking_weather_cpu.jpg`: Earth zoom, every 40th frame. The orbit and
  high frames use a flat placeholder globe (NASA GIBS is not reachable from the cloud machine), so only the last
  five columns mean anything.

Fog reads best: the inversion lies in the canyon below the overlook and the stand-in stands clear of it. Snow
whitens the whole South Rim plateau, which is right for a winter storm but leaves the trees to the prompt. Night is
dark enough that the stand-in is hard to see; LTX may struggle with it.

## Your own character: the subject sheet

With `--keyframes flux-klein`, the Earth zoom first paints one **subject sheet**: the person alone, full length,
facing the camera, on a plain grey wall. It is a klein edit of `--subject photo.jpg`, or painted from
`--subject-description` (default: a hiker dressed for the weather) when there is no photo. Every eye-level keyframe
gets the sheet as its subject reference instead of the raw photo. This is meant to fix three things:

1. The photo's own background and framing leaking in. klein copies reference images wholesale; this is the same
   failure the push-in hit.
2. A head-and-shoulders photo has no body or shoes to copy.
3. An invented person differing between keyframes: each painted its own hiker from the text.

The sheet is saved as `<prefix>_subject_*.png` (node 173). `--no-subject-sheet` goes back to the raw photo. Qwen
keyframes are unchanged.

In ComfyUI, **GeoBlock 7 · Subject** has `weather` and `subject_sheet` inputs and a `sheet prompt` output to feed
a klein edit.

## Fast checks: `--keyframes-only`

`geoblock-run ... --ltx --keyframes flux-klein --keyframes-only` stops after the keyframes and the sheet, with no LTX.
That is about a minute or two per weather instead of 12+ minutes, which makes it the way to compare looks.

`scripts/make_test_subject.py` paints a stand-in "user photo" with klein (a casual knees-up street photo, the hard
case), so `--subject` can be tested without a real person's picture.

## GPU test plan

1. Run keyframes only, with an invented subject, for each of snow, fog, dusk, night and sandstorm.
2. Run `make_test_subject.py`, then clear-weather keyframes only with `--subject`, once with the sheet and once with
   `--no-subject-sheet`. Check that the face and clothes match across the person keyframes and that the street
   doesn't leak in.
3. Render a full 10 s LTX 2.5 Earth zoom: fog, the test subject, and the sheet.

## Night and dust from orbit down (after the GPU run)

The first GPU run showed night and sandstorm changing at the top of the zoom. The orbit and cloud keyframes stayed
daylit and white, so the video flipped from day to night around f56. Fixed in three places:

- **Keyframe prompts.** A preset's `above` text replaces the daytime orbit, satellite and in-cloud descriptions.
  Night gets the night side of the Earth with moonlit clouds and city lights. Sandstorm gets a tan dust plume from
  orbit and a dive through orange-brown dust.
- **LTX prompt.** `whole_shot` says the weather holds the whole way down, and the "layer of white clouds" becomes
  moonlit cloud or dust.
- **Blender.**
  - At night the globe is dimmed (`globe_light`) and the limb glow is weaker.
  - The cloud shell seen from orbit and the deck flown through take the weather's colours (`shell_colour`,
    `deck_colour`): blue-grey at night, tan dust in a sandstorm.

CPU check: `review/weather-character/earthzoom_night_dust_top_cpu.jpg`, every 12th frame, with the placeholder globe.

## Night v3 (after the full night GPU run)

- **Lit square from orbit.** Fixed. At night the terrain patches now get the globe's dimming from 60 km up, with full
  moonlight below 8 km (a "Night dim" node keyframed with altitude). CPU before and after:
  `review/weather-character/earthzoom_night_dim_cpu.jpg`; the top row is before, the bottom row after.
- **Moon drifting against the Milky Way.** Each keyframe placed its own moon. The moon is now behind the camera and
  out of view, so only the stars are in the sky.
- **Headlamp hiding the face.** The invented night hiker's head torch is now switched off, and the face is lit by
  the moon.
- **Snow-like ground.** The moonlit ground no longer looks like snow ("no snow, pale ground dim grey-blue", e8db76c).
- **Not addressed.** The brightness pumping and ghosting in the swoop (f84–96) look like the general LTX swoop
  issue, not a weather problem.

## Night v4

The v3 GPU run fixed the lit square, the stars and the face. klein still painted a moon (f109–131), a lit head
torch, and a white, snowy-looking rim. Each of these came from words in the prompt: "no moon in the sky", "head
torch switched off", "no snow". The night prompt and the night outfit now name only what should be in the frame: the
Milky Way, dim cool blue night light, dark and muted ground, and a hiker in a dark jacket and beanie. The words moon,
torch and snow are gone from the ground-level prompt.
