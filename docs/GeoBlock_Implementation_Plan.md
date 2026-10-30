# GeoBlock SceneCompiler implementation plan

*Historical starting plan. It predates the Blender, Woodside, ComfyUI and H3 experiments. See [the project story](PROJECT_STORY.md) and [current pipeline status](programmatic-flyover.md) for what was actually built.*

This plan turns the supplied GeoBlock brief into an executable development sequence. The first deliverable is a verified pair of Worm’s Head terrain assets. The later deliverable is a matched comparison of generative video made from coarse global terrain and Welsh LiDAR terrain.

The project tests whether coarse geometry plus photographs can establish recognisable geography and controllable performance. Results remain open: global terrain may suffice, high resolution may matter, or both may need another approach. This document is a plan; no terrain, Blender scenes or generated comparison clips have yet been produced.

## Agreed decisions

| Decision | Working choice |
| --- | --- |
| Primary environment | Linux workstation with RTX 4090 |
| First video handoff | Manual upload to a service supporting reference video and images |
| Submerged walking surface | Identical, explicitly labelled synthetic seabed proxy in both variants |
| First geographic example | Worm’s Head, Rhossili, Gower |
| Initial area | Radius parameter 1,500 m, interpreted as a 3,000 × 3,000 m projected square |
| Duration and format | Proposed 12 seconds, 24 fps, 288 frames, 16:9 |
| Initial camera | Fixed camera, final position selected after terrain and photo inspection |
| Geometry scale | Metres, vertical exaggeration 1.0 |
| Product boundary | Core compiler, Blender editing, thin ComfyUI nodes |

The exact camera position, shoreline entry point and photograph set are deliberately selected after previews exist. A location search result provides a starting centre; it does not determine the cinematic viewpoint. Freeze the final area before generating the production A/B pair.

“Identical” means identical scene coordinates, bounds, water, camera and performance. The terrain geometry must differ, because terrain source and fidelity are the experimental variables.

## Verified sources and outstanding checks

OpenTopography distributes GEDTM30 as a roughly 30 m bare-earth terrain model under CC BY 4.0. Its published metadata identifies WGS84 horizontal coordinates and EGM2008 height referencing. The dataset is broad global land coverage, rather than guaranteed coverage of every location. [GEDTM30 metadata](https://doi.org/10.5069/G9BV7DT1)

GEDTM is accessible through OpenTopography’s global DEM API. The API requires an individual key. Keep the key outside workflows, exported packages and logs; users configure their own credentials. Dataset reuse and API access have separate conditions. [API documentation and conditions](https://opentopography.org/developers) · [GEDTM API availability](https://www.opentopography.org/news/new-machine-learning-derived-digital-terrain-models)

DataMapWales publishes downloadable Welsh LiDAR DTMs as tiles and mosaics, including a 32-bit terrain Cloud Optimized GeoTIFF. Its download map identifies OGL licensing, and its tile catalogue describes 1 m terrain data. Actual Worm’s Head coverage, raster CRS, height datum, scale/offset and tile dates still require inspection. A catalogue’s projection does not by itself establish every raster’s metadata. [Welsh download map](https://datamap.gov.wales/maps/lidar-data-download/) · [Tile catalogue metadata](https://datamap.gov.wales/layers/geonode%3Awelsh_government_lidar_tile_catalogue_2020_2023/metadata_detail)

MiniMax describes H3 as accepting multimodal context and demonstrates using video for movement and images for identity. This makes it a candidate for the manual test; the actual account interface and upload constraints must still be checked. No H3 API adapter or local 4090 execution is assumed. [Official H3 announcement](https://www.minimax.io/blog/minimax-h3)

ComfyUI documents custom datatypes for custom nodes. Use those interfaces, with runtime validation of objects passed through them. [ComfyUI datatypes](https://docs.comfy.org/custom-nodes/backend/datatypes)

## Architecture and dependency boundaries

Keep the supplied repository layout, adding a CLI, JSON schemas, cache handling and experiment validation inside the independent core. Create modules only when their milestone needs them; avoid empty model adapters suggesting implemented support.

| Component | Responsibility | Must remain independent of |
| --- | --- | --- |
| geoblock_core | Project specs, providers, raster processing, mesh generation, provenance, package validation, prompt and request preparation | bpy, ComfyUI, torch |
| CLI | Invoke core operations and produce machine-readable results | Blender UI and ComfyUI |
| Blender bridge | Import compiled assets, build stage, edit performance, render and export | Terrain download implementation |
| ComfyUI nodes | Validate inputs and invoke the same core or CLI | Custom GIS or animation logic |
| Adapter layer | Describe supported controls and prepare explicit requests | Assumptions that every model consumes every pass |

Use a dedicated Linux Python environment for GIS dependencies. Proposed stack: NumPy, Rasterio, pyproj, trimesh, Pillow, Pydantic, HTTP client, platformdirs and pytest. Pin a tested Python minor version, Blender release and ComfyUI revision during setup rather than depending on “latest”.

Run Blender as a separate executable. It consumes assets and JSON, so GIS libraries do not need installing inside Blender’s Python. Prefer an isolated CLI subprocess for Comfy nodes if GIS wheels conflict with the existing Comfy environment. Pass argument lists, return JSON and capture readable errors.

Use pyproject.toml as the dependency source of truth; generate a pinned environment specification rather than maintaining conflicting dependency lists. Record Blender, FFmpeg, Python and processing versions in export metadata.

Code licensing remains an implementation setup choice: a permissive independent core with a GPL-compatible Blender bridge is a candidate. Check distribution obligations before release, retain third-party notices, and treat dataset and reference-image rights separately.

## Project contracts

Define versioned contracts before implementing providers:

- GeoSceneSpec: stable project ID, confirmed WGS84 location, canonical metric area, projected CRS, horizontal origin, common vertical reference, radius interpretation and provider selection.
- TerrainAsset: raster and mesh paths, local bounds, nodata coverage, native and processed spacing, content hashes, transforms, quality report and source records.
- PerformanceSpec: per-frame character root and joint transforms, camera transforms, timing, lens, water and proxy geometry.
- ShotPackage: manifest and referenced assets validated against the shot schema.
- BackendRequest: prepared uploads, reference roles, compiled prompt, selected capabilities, settings and an explicit unsupported-control report.

Each contract carries its schema version. Reject unsupported major versions; preserve recognised provenance and optional fields when round-tripping.

The project owns the coordinate system. Providers cannot independently choose origins, crop dimensions or vertical offsets.

## Milestone 0 Source and environment check

Complete this small feasibility check before building the terrain pipeline.

1. Identify the Linux setup, Python environment and available Blender executable without changing the existing Comfy installation.
2. Confirm GEDTM30 API request parameters against the current API documentation and make one authenticated small request when a key is configured.
3. Inspect Welsh DTM metadata and coverage for the same provisional area. Prefer a windowed read of the published COG or a few intersecting tiles; save a local subset for LocalGeoTIFFProvider.
4. Record both horizontal and vertical references. Select and verify a supported conversion to one common height reference.
5. Check the chosen video service supports reference video plus separately identifiable reference images. Record duration, dimensions, file count, upload size and role limitations.

Do not build a Welsh national provider. Obtaining the example’s local raster is fixture preparation, followed by the ordinary LocalGeoTIFFProvider path.

Exit condition: both terrain sources are accessible and understood, or the missing credential, coverage or transform is precisely identified. Backend inspection informs later exports but does not block terrain development.

## Milestone 1 Matched terrain compiler

This is the first coding task and the only implementation scope initially.

### Location and shared coordinates

Accept coordinates immediately and a place query through a replaceable geocoder. Return candidates for ambiguous queries rather than silently selecting one. Cache the chosen location. The Worm’s Head fixture stores its confirmed coordinates; do not commit placeholder coordinates.

Create the canonical 3,000 m square in one projected metric CRS. Fetch a geographic envelope with enough margin for reprojection and interpolation, then crop to the canonical projected area. Both variants use the same origin and extents, even if their raster dimensions and sampling grids differ.

For this Welsh example, British National Grid is a proposed project CRS, subject to raster inspection. A future general resolver selects appropriate UTM or local projections, with explicit handling for polar areas, the antimeridian and invalid bounds. Unsupported cases return clear errors in the MVP.

Use explicit longitude/latitude axis order in transformations. Record complete CRS definitions, operation selection, origin, units and forward/inverse transforms. X and Y represent projected easting and northing offsets; retain grid convergence if needed for true-north orientation.

Normalize heights using an audited vertical operation with the required grids available. Record the operation and grid identities. Ordinary horizontal reprojection does not establish vertical compatibility. Unknown height references, missing grids or silent approximate conversions prevent a scientifically matched export. Any exploratory unaligned preview must be labelled accordingly.

Sea level is a later chosen elevation in the common height reference. Z = 0 is not automatically the tide level.

### Providers and caching

Implement OpenTopographyProvider configured for GEDTM30 and LocalGeoTIFFProvider. Provide coverage, metadata and fetch methods. A local provider accepts one raster or a declared local mosaic.

Implement a small resolver with explicit source pinning and AUTO preference. AUTO ranks configured, actually implemented providers; future regional and fallback adapters remain extension points. If GEDTM30 fails, do not silently replace A with another dataset. A deliberate fallback starts a newly labelled comparison.

Source metadata includes provider, dataset/version, source type such as DTM or DSM, horizontal and vertical references, nominal resolution, native pixel spacing, license, attribution, acquisition dates, retrieval timestamp, source URI and checksum.

Use separate raw and derived caches. Normalize bounds and parameters before hashing; derived keys also include input checksum, output CRS/grid, nodata policy, resampling, software version and mesh settings. Check for a cached raster that already covers the request before downloading overlapping areas. Use atomic writes and locking so concurrent executions reuse one successful acquisition. Offline reruns reuse cached material; failures do not become successful entries.

### Raster and mesh processing

Decode source nodata, raster scale/offset and units before resampling. Preserve missing-data masks. Do not replace unknown ocean or missing tiles with measured elevation zero.

Reproject elevations with a documented continuous-data interpolation method and conservative validity handling. Do not interpolate across large coastal gaps or triangulate holes into false land. Produce coverage diagnostics and fail if the intended landmark or walking region lacks usable land data.

Retain both the original cropped raster and the normalized metric GeoTIFF. Mesh sampling is separately configurable from source resolution. Finer mesh spacing must never be described as new geographic detail in the global source.

Proposed starting grids:

| Variant | Mesh spacing | Approximate full-square grid budget |
| --- | --- | --- |
| A GEDTM30 | 30 m | 10,201 vertices and 20,000 triangles |
| B Welsh LiDAR | 5 m initially | 361,201 vertices and 720,000 triangles |

These counts assume a complete regular grid including boundary nodes; nodata changes them. Preserve the 1 m source raster. Decide whether a finer B mesh is necessary from screen-space detail and measured rendering performance. Adaptive detail can follow if the uniform grid proves inadequate.

Define the boundary-node convention so both mesh extents are exactly equal. Fetch margins permit sampling the perimeter. Preserve elevation values, triangle winding and units.

Use the internal Z-up metre convention and explicitly map it to glTF’s Y-up convention. Record the basis transform, apply it exactly once and verify a Blender import round-trip later. [Blender glTF convention](https://docs.blender.org/manual/en/dev/addons/scene_gltf2.html)

### Outputs and acceptance

Each provider run produces terrain_source_crop.tif, terrain_metric.tif, terrain.glb, metadata.json, preview_topdown.png and ATTRIBUTION.md. Include a compact numeric mesh representation if it simplifies the Blender bridge.

Top-down previews show north, scale, footprint, elevation range and missing coverage. They are inspection aids, not substitutes for known-point checks.

Produce A_global and B_lidar plus a pair report. Require:

- Same project origin, common height reference and 3,000 m XY extents within a proposed 1 cm export tolerance.
- Coordinate round-trip error within the selected operation’s documented accuracy; internal local transform error under 1 mm.
- Mesh heights match valid processed raster samples within a proposed 1 cm numeric tolerance.
- Native and derived resolution, source dates and licensing survive every export.
- Several saved landmark checkpoints establish headland orientation and plausible relative elevations.
- A cached rerun and an offline local-raster run succeed without extra terrain requests.

Do not force A and B elevations to equal each other. Differences are measured and recorded. The report must distinguish transformation error from source accuracy.

Exit condition: the matched metric terrain assets pass numeric and visual inspection. Blender scene generation begins only after this gate.

## Milestone 2 Blender blocking stage

Build the specified collections for TERRAIN, WATER, BLOCKING, CAMERAS, REFERENCE_CAMERAS, GEO_CONTEXT and EXPORT. Set units to metres. Add separate LOCATION_REFS, CHARACTER_REFS and STYLE_REFS collections and corresponding reference records.

Provide an unmistakable rigged mannequin with a documented reusable asset license. Manual animation remains in Blender; no automatic motion synthesis is required.

Create one master PerformanceSpec and reuse it in both scenes. Store frame-by-frame character and camera samples so constraints and animation evaluation cannot silently diverge after import.

Use one fixed camera initially. Preserve lens, sensor dimensions and fit, shift, clipping, resolution, pixel aspect, world matrices and camera conventions. The package later records both Blender camera axes and a defined vision-camera basis.

Before freezing blocking, inspect terrain and photographs to choose an identifiable viewpoint and physically plausible shoreline approach. Do not choose a path that only works on one terrain. Plot root height and foot clearance against both variants. If no acceptable shared land path exists, adjust the shared staging and document the limitation.

Add the agreed identical seabed proxy as a separate object with provenance_type = synthetic. It must not alter or be merged into the surveyed terrain source. Its land join and submerged path must remain plausible in both scenes. Specify its dimensions, slope, elevation, author and checksum.

Use an opaque neutral water plane for the control render so it hides progressively immersed limbs. Keep the proxy hidden from appearance renders when it is underwater. It may appear in a diagnostic pass.

A proposed beat schedule is approach during seconds 0–4, shoreline entry around 4–6, immersion during 6–11, and fully hidden by 11–12. Actual timing is manually authored and then frozen. Treat these as proposed staging, not recovered historical motion.

Exit condition: both .blend files share exact water, camera, proxy, rig and evaluated performance hashes; terrain is the intentional difference. The immersion comparison evaluates adherence to staged controls, not verified seabed geography.

## Milestone 3 Shot Package export

One Blender action or headless command exports a valid version 0.1 package. Use Eevee for neutral RGB and verify pass availability in the pinned Blender version. Do not assume Workbench exports every required control pass.

Export clay.mp4 and blockout.mp4 with documented visual roles; they may initially contain the same neutral RGB render. Export first_frame.png, last_frame.png, depth, masks, camera.json, characters.json, scene.json, references, manifest.json and ATTRIBUTION.md.

Define numeric controls precisely:

- Depth: float32 EXR with metre units, documented camera-ray distance or camera-axis depth semantics, invalid/background handling and water-occlusion policy. Define this from an actual render calibration test.
- Optional depth PNG: 16-bit model-oriented derivative with one declared range shared across both variants and the whole clip. No independent per-frame normalization.
- Masks: lossless visible-pixel masks for character, terrain and water. Character visibility must shrink correctly below water. Record whether edges represent coverage or discrete labels.
- Camera: per-frame camera-to-world matrices, intrinsics, image dimensions, pixel aspect and basis conversion.
- Character: per-frame root transforms; joint/pose export remains optional until a backend needs it.

Opaque water means its surface is the first visible depth hit. If a later model needs underwater geometry or unoccluded character masks, export separately named supplementary passes rather than changing the required pass meaning.

Use binary shader or object-ID passes supported by the chosen renderer for segmentation. Test occlusion and silhouette edges. Blender documents that some numeric/index passes lack antialiasing; do not promise interchangeable mask semantics. [Blender render passes](https://docs.blender.org/manual/en/4.5/render/layers/passes.html)

Manifest asset entries carry relative paths, content hashes, media format, dimensions, frame count, start frame, FPS, colour space and pass conventions. Include project transforms, all source records, synthetic objects, renderer and tool versions. Store references by role with rights/attribution and checksums. Preserve the full master references even when an adapter prepares a smaller selection.

Render PNG or EXR sequences first; encode RGB with FFmpeg. Validate decoded video frame counts and duration, not just export settings. Disable motion blur and depth of field for baseline spatial controls.

Exit condition: a small calibration scene and the Worm’s Head package pass schema, file, frame, camera, mask and depth checks. Export time and memory use are measured on the target workstation.

## Milestone 4 ComfyUI wrappers

Implement the five small nodes from the brief:

| Node | Output |
| --- | --- |
| GeoBlock Location | Validated GEOBLOCK_SCENE |
| GeoBlock Terrain | Terrain asset, provenance and preview |
| GeoBlock Build Blender Scene | .blend path and scene metadata |
| GeoBlock Import Shot | Validated GEOBLOCK_SHOT |
| GeoBlock Prepare Model Request | Upload bundle, compiled prompt and supported-control report |

For the agreed manual route, the last node prepares the submission; it does not claim to generate video. Keep the adapter interface ready for a later explicitly implemented API adapter.

Workflow JSON must contain no credentials and no workstation-specific absolute paths. Let users configure the Blender executable, output folder and provider key outside the portable workflow.

Use file/spec hashes to invalidate cached node results when external rasters or Blender exports change. Cache large passes on disk and load batches lazily rather than keeping the full sequence in GPU memory.

Exit condition: a clean Comfy installation can run the example through terrain/stage creation and import a manually exported shot without bespoke scripting. Editing remains in Blender.

## Milestone 5 Manual video submission

Choose one accessible reference-video service after the capability check. Prepare an identical ordered image subset for A and B if the service cannot accept the full Location/Character/Style pack. Never silently drop, combine or relabel reference roles.

Compile prompts explaining that the control video defines camera, geometry, trajectory and timing; photographs define location appearance; character images define identity; style frames define rendering. Keep the creative instructions identical between variants. Capability declarations report unused depth or masks.

Record the exact service and model identifier, visible settings, reference order, prompt, submission time, run ID, output and seed if exposed. Unknown settings remain unknown. If the service cannot consume a reference video, it does not qualify for the primary test.

Exit condition: one real service accepts the prepared package and returns a recorded output. This proves an end-to-end manual handoff. Automated generation through a Comfy API node is a subsequent adapter milestone, not a claimed result of manual upload.

## Milestone 6 Matched experiment

Start with one A/B pair to check submission and control interpretation. If viable, extend to three paired attempts, keeping failures and avoiding selection of only the best outputs. This is a proposed small exploratory sample, not a statistically decisive trial. The first pair counts toward the three.

Pair seeds where available. Without seed control, alternate or randomize A/B submission order and record stochastic variation. Model/version changes invalidate a matched pair.

Package validation asserts equal canonical area, common vertical reference, water, proxy, camera, animation, references, prompt, export settings and backend settings. Provider records, terrain meshes and resulting control-video pixels are allowed differences.

Evaluate geography separately from visual attractiveness:

| Measure | Evidence |
| --- | --- |
| Landmark consistency | Silhouette and landmark checkpoints at fixed timestamps |
| Camera adherence | Horizon and static landmark screen positions across frames |
| Trajectory adherence | Manually annotated character centres against projected blocking path |
| Immersion adherence | First water entry and final disappearance time; visible character area |
| Appearance | Human comparison with supplied photos and style references |
| Fine geometry contribution | Whether terrain distinctions remain visible and useful after generation |

Use a small set of fixed checkpoints, for example seconds 0, 3, 6, 9 and 11. Store annotations and failures. Do not compare a character directly in pixels if its artistic silhouette changed; use consistent centres and note ambiguity.

A versus B changes both resolution and source methodology, dates and accuracy. If B appears materially better, add an optional cheap diagnostic render from B downsampled to 30 m before making a claim specifically about resolution. This secondary control follows the primary demo and is not part of the first coding task.

Present side-by-side terrain, identical blocking, generated results and a short findings report. Outcomes A, B and C in the supplied brief remain the decision framework. A useful result for Worm’s Head is evidence for this shot, not proof that every global location will work.

## Test coverage and release boundary

Core tests use small synthetic rasters with known planar slopes, holes, negative heights, scale/offset and coordinate transforms. Test provider success, credential errors, interrupted downloads, cache reuse, datum uncertainty, exact extents, mesh orientation and provenance persistence.

Blender integration tests use one tiny calibrated scene for import orientation, known depth, mask occlusion, camera matrices and frame counts. Test the real Worm’s Head assets visually and retain checkpoint evidence.

Provide a CLI recipe, a Comfy workflow, installation instructions, the schemas and a reproducible example configuration. Redistributable demo data and reference assets carry their licenses; private assets are referenced locally and excluded from the public release. No external repository or package publishing is included in this planning task.

Implement no Google geometry dependency, global LiDAR discovery, photo scraping, OSM, buildings, photogrammetry, Gaussian splats, automatic animation, bespoke 3D editor, narration analysis or cloud deployment during this validation sequence. Retain those as later milestones in the supplied order.

## First implementation handoff

Implement Milestone 1 after the source/environment check. Build the minimal CLI, project schema, coordinate and vertical handling, GEDTM30/local providers, cache, raster processing, grid mesh export and preview.

Run both providers against one confirmed Worm’s Head project. Deliver the paired terrain outputs, attribution and a numeric/visual comparison report. Stop scene generation until the matching-coordinate and geography checks pass.

The user’s next contributions are the Linux/Blender setup details when implementation begins, an individual terrain API key configured locally, and the location/character/style references before the video milestone. Final camera placement and water level are selected from the verified terrain previews; the chosen video service and generation budget are fixed before paid submissions.
