# Where the clips are

One place per place, one obvious file per build. The answer to "where is the latest test run" is always:

```
outputs/touchdown/<place>/latest.mp4
```

| Where | What | In git? |
| --- | --- | --- |
| `outputs/touchdown/<place>/` | the build: every stage, the manifest, the logs, `report.md` and `latest.mp4` | no (ignored) |
| `app/demo-clips/` | the 720p copies with a title and a poster, which the app's Examples panel plays | yes |
| `docs/demo/` | the still images: the real-aerial-beside-the-clip proof, the contact sheets | yes |

`outputs/` holds the full-resolution clips (about 20 MB each) and everything else the build produced. It is
ignored by git, so the repository stays small and the media is not versioned.

## Making a clip

```bash
# a place from the catalogue: the manifest is generated, the closer ending is applied, and the build runs
python scripts/build_place.py canyonlands-grand-view

# the same, with the wide ending the place used to have
python scripts/build_place.py horseshoe-bend --ending full

# a custom point
python scripts/build_place.py custom --lat 36.0617 --lon -112.1078 --name mather-point
```

The build writes into `outputs/touchdown/<name>/`, and at the end copies the finished clip to `latest.mp4` there.

To put a finished clip in the app's Examples panel, and to refresh the gallery:

```bash
python scripts/make_demo_gallery.py --out app/demo-clips \
    --clip "Canyonlands, Grand View Point=outputs/touchdown/canyonlands-grand-view/latest.mp4"
```

The clip has to be served to the app from the gallery folder, so the gallery copies are the ones the app plays.

## The pieces of the demo

| File | What it is |
| --- | --- |
| `real-vs-made.jpg` | The real NAIP aerial photograph of Grand View Point (USDA, public domain) beside the same place in the clip: the proof that the ground is measured |
| `canyonlands-grand-view-contact-sheet.jpg` | One tile every 0.75 s of the Canyonlands build, the whole shot at a glance |
