# Place catalog: where a fly-down can land

GeoBlock ships a catalog of 45 landing places (`src/geoblock_core/data/places.json`) for the Earth zoom and the other camera presets. Each place has a curated landing point (where the person stands) and, for some, a named view (what the camera looks at past them, such as the Matterhorn from Gornergrat). Everything else in an entry is measured from open data by `scripts/build_place_catalog.py`. Listing or choosing places downloads nothing: terrain, imagery and land cover are fetched only for the place a shot is built for, and only the window around it.

## Using it

```bash
uv run geoblock-run --list-places            # all 45, one line each
uv run geoblock-run --list-places norway     # filter by words in id, name, region, country or kind
uv run geoblock-run --place-info gornergrat  # terrain, lidar, imagery, land cover and licences for one place
uv run geoblock-run --place gornergrat --preset "Earth zoom" --duration 10 --ltx --keyframes flux-klein --ltx-version 2.5
```

Add `--landing-lidar auto` (US places) to land on USGS 3DEP 1 m lidar terrain instead of the 10 m DEM; see [1 m lidar at the landing](#1-m-lidar-at-the-landing).

`--place` takes an id or any words that match one place. It sets the coordinates, the camera heading (toward the named view, or toward the deepest measured drop when there is none), the place's palette (Grand Canyon places keep `grand-canyon`) and the measured look used in the keyframe and LTX prompts. An explicit `--heading` or `--palette` still wins.

On the ComfyUI canvas, **GeoBlock 2 · Place From Catalog** is a dropdown of the same places. Its `chosen place` output plugs into **Build Landform Scene** like the other place nodes; the scene then carries the view heading (used by **Camera Preset** when its heading is -1), the palette (used when the render and keyframe nodes are left on `default`) and the look.

## What each entry records

| Field | Source | Licence |
| --- | --- | --- |
| Terrain family, ground height, relief within 1 km, drop below the landing | USGS 3DEP 1/3 arc-second (10 m, bare earth, NAVD88) in the conterminous US; Copernicus DEM GLO-30 (30 m, surface model, EGM2008) everywhere else | Public domain; Copernicus DEM licence (free for any use with attribution) |
| US lidar surveys containing the landing point, with year, point count and density | 3DEP lidar Entwine Point Tile index (`usgs-lidar-public`) | Public domain |
| National lidar outside the US | Listed by country (England, Wales, Scotland, Switzerland, France, Norway, Ireland, New Zealand, Australia, Japan) | As listed; none of these are wired into scenes yet |
| Imagery | NASA HLS 30 m via GIBS (worldwide, draped with `--imagery`; the Earth zoom always drapes it); USDA NAIP 0.6 m in the US (listed, not draped yet) | Public domain |
| Land cover near the person (600 m) and across the view (8 km), and prompt words made from it | ESA WorldCover 10 m 2021 | CC BY 4.0 |
| `landing` (only when moved) | The highest ground within `snap_m` of the curated point, for cliff-top places where a 30 m surface model smears the brink; or the nearest point (up to 200 m) where the named view clears the foreground at eye level | — |
| `view_clearance_deg` | How far the named view's top stands above the nearer ground, seen by the Earth zoom's eye-level camera 2-12 m behind the person (worst case, and over ±15 m) | — |
| `dem1m` | USGS 3DEP 1 m lidar DEM projects with a tile under the landing (US), newest first | Public domain |
| `near_view_deg`, `lidar_landing` | On the 1 m lidar: how far below level the person sees toward the view, and for overlooks whose edge hides the drop, the brink (used only with `--landing-lidar`) | — |

Places that are not in the catalog get the same treatment at scene build time: the terrain family is chosen by location and the look is measured from WorldCover around the coordinates, so `--latitude/--longitude` and the search nodes now work outside the US too.

## What changed in the pipeline

- **Worldwide terrain.** `CopernicusDEMProvider` reads only the needed window of the public Copernicus GLO-30 COGs on AWS. Scenes outside the conterminous US use WGS 84 / UTM and EGM2008 heights. Open-sea tiles do not exist and are filled as sea level; 3DEP now does the same for missing offshore tiles, which the Earth zoom's 100 km regional terrain reaches at coastal places such as Big Sur.
- **Same family for the Earth zoom's regional terrain** as its local terrain, so both share one projected CRS when Blender lines them up.
- **HLS imagery picks late summer for the hemisphere** (end of March in the south) instead of always September.
- **Prompts.** A curated palette (grand-canyon) keeps its hand-written words; any other place uses its measured look, and a named view is added to the far description ("the Matterhorn stands out in the view beyond").

## Can the person see the view?

The first Gornergrat blocking render showed only the Matterhorn's tip over the summit hump, so the builder now checks every named view from where the Earth zoom's camera actually ends: eye height, 2 to 12 m behind the person, looking past them. The view's top (the highest ground within 400 m of it, plus `structure_m` for a built top such as the Golden Gate's towers) must clear the foreground by 4°. The foreground stops at 60% of the distance, so the target's own flanks are not counted, and the check takes the worst of the point and four points 15 m around it, so a single lucky cell doesn't pass. Merges are aligned to the source pixel grid, so the answer doesn't change with the window size.

If the view is hidden, the landing moves to the nearest point within 200 m that clears it, or stays and gets a warning when no point gains at least 2°. Results: Gornergrat moved 10 m (1.9° to 7.7°), Battery Spencer 10 m, Victoria Peak 20 m, Johnston Ridge 60 m, Sugarloaf 80 m and Crater Lake 82 m. Three keep a warning: Tunnel View (Half Dome clears by 2.8°), Aiguille du Midi (Mont Blanc 3.4°) and Rhossili (Worm's Head 0.8°). In the cloud Blender render after the move, the Matterhorn stands clear behind the stand-in.

## 1 m lidar at the landing

`--landing-lidar auto` (or the `landing_lidar` input on **Build Landform Scene**) adds an 800 m square of USGS 3DEP 1 m bare-earth DEM, built from lidar, around the person. Only the Earth zoom uses it. It is nested inside the 20 m local mesh: its outer 15% is blended into the local heights and the local faces under it are cut out, so there is no step at the seam. `auto` takes the newest project with a tile under the landmark and falls back to older ones; a project name picks that one. Where a project boundary clips the square, the missing part is filled from the 1/3 arc-second DEM (same NAVD88 heights), as long as the middle 200 m is all lidar; Mather Point needs this (14% filled). It is US only; national lidar elsewhere is listed but not wired.

The patch wears USDA NAIP aerial photography (about 0.6 m, public domain) from the USGS National Map image service instead of 30 m HLS. It keeps its own colours and fades into the surrounding HLS imagery over its outer 15%, so it does not show as a hard square. (Matching its colours to the HLS over the whole patch was tried first and crushed it to near-black olive, because the HLS there is in shadow.) On the first GPU run at Glacier Point, with HLS only, klein painted the granite overlook as a grassy slope: WorldCover reads the slabs there as grassland, and 30 m imagery carries nothing finer. If the NAIP service fails, the patch falls back to HLS.

The prompt words for the ground underfoot come from NAIP as well when `geoblock-run` uses the 1 m patch. WorldCover's share within 600 m says "forest, with grassland and meadow" at Glacier Point. Even with NAIP on the patch, klein still painted a meadow at eye level, where the blocking ground right in front of the camera is smooth. So `imagery.naip_ground` splits an 80 m square of NAIP around where the person stands into bare rock (bright and colourless), trees and shrubs (dark and green), grass (lighter green) and bare earth, leaving out deep shadow. It feeds those shares into the eye-level keyframe prompts and the LTX prompt ("standing at the viewpoint on ..."). On the Glacier Point patch it reads about 36% rock, 29% trees, 19% earth and 14% grass. Grass and lit tree tops overlap in RGB, so this is a rough split. The ComfyUI node path does not do this yet.

The 1 m terrain matters most at overlooks. On the 10 m DEM a cliff edge is a rounded slope, so a person standing 20 m back from the real brink still sees the drop in the blocking. On 1 m lidar the edge is sharp, and the first Glacier Point render showed flat ground hiding the valley. So the builder measures, on the 1 m lidar, how far below level the person sees toward the view (`near_view_deg`). If an overlook's edge hides the drop (less than 20° down), it walks up to 40 m toward the view, never down a face, to the first point that sees 25° down, and stores it as `lidar_landing`. That point is used only with `--landing-lidar`, because on the 10 m mesh it lies on the rounded slope. Glacier Point's brink is 15 m out (9° down becomes 25°, and the re-render shows the valley below the stand-in), and Crater Lake's is 18 m out. Tunnel View and Johnston Ridge have no brink within 40 m and carry a warning. Places whose heading is picked at render time (toward the deepest drop, such as Mather Point) are not checked for a brink yet.

1 m lidar also shows steps that the 10 m DEM smooths away. Mather Point's catalog point is on a rock step 2-3 m below the path behind it, so on the first 1 m render the figure stood on a slope and dropped out of the eye-level frame. Now the Earth zoom moves the figure, on the 1 m patch only, to the nearest point within 10 m that is level underfoot (under 0.4 m of rise across 1.2 m) and that the eye-level camera, 2 to 12 m behind, can see down to the chest (`zoom.standing_spot`; the route records it as `subject_local_m`). At Mather Point that is 2.5 m back, on the path, and the re-render shows the figure against the canyon. At Glacier Point's brink the point already passes, so nothing moves.

## Checks

Measured heights against published ones: Mount Whitney 4417 m (4421), Half Dome 2688 (2693), Mount Fuji 3767 (3776), Kilimanjaro 5866 (5895), Ben Nevis 1342 (1345), Table Mountain upper station 1055 (1067), Glacier Point 2201 (2199), Mather Point 2164 (2172). `tests/test_places.py` keeps six of these within 40 m.

Five landing points were moved onto the top by `snap_m`: Angels Landing (48 m), Aiguille du Midi (33 m), Preikestolen (94 m), Table Mountain (82 m) and Rhossili (752 m, onto the cliff top above the bay). Trolltunga was dropped: its listed coordinate reads as the lake 650 m below in the DEM. Uluru's landing is the flat car sunset viewing area about 3.6 km from the rock, so it carries a low-relief warning by design.

Earth zoom blocking was rendered in the cloud with Blender Cycles on the CPU for Gornergrat, Milford Sound, Victoria Peak, the Cliffs of Moher, Glacier Point (10 m, then 1 m with the brink) and Mather Point (1 m, then with the standing spot). The globe and imagery textures there were stand-ins, because NASA GIBS is not reachable from the cloud. On the 10 m terrain, geometry, heading and the stand-in look right everywhere, and Glacier Point on 1 m lidar with the brink is the best eye-level frame so far. Contact sheets are in `docs/review/place-catalog/`. None of the new places has been through keyframes and LTX on the GPU yet.

## Coordinates against Overture Maps

Wikidata, Nominatim and GeoNames are blocked from the cloud environment, but the Overture Maps release on AWS (2026-09-23.1) is reachable. Its named land features, infrastructure and places within about 3 km of every catalog point were read by row group, only where the bounding boxes overlap, and matched by name (`scripts/check_places_overture.py` reruns this). Five points were wrong by 500 m to 1.8 km and are now fixed:

| Place | Was off by | Now at |
| --- | --- | --- |
| Bryce Inspiration Point | 920 m north-west | Upper Inspiration Point viewpoint |
| Battery Spencer | 550 m north | the battery (the view check then moves it 10 m) |
| Rifugio Locatelli, Tre Cime | 1.1 km south-west | the hut |
| Xianggong Hill | 1.5 km south-east | the hilltop viewpoint |
| Uluru sunset viewing area | 1.8 km north-east | the car sunset viewing area |

Rhossili's landing was always the cliff top above the bay, but it was named "Rhossili Down", a hill 2.4 km away; it is now "Rhossili cliff top". After the fixes, 39 of 45 points are within 150 m of a feature with the same name. The other six were checked one by one: Bixby Bridge's point is 160 m from Overture's viewpoint, on the same pull-out; Table Mountain and Sugarloaf are within 100 m of their summit features once the name variants ("Pão de Açúcar") are counted; Fuji's summit point is on the crater rim, 400 m from Overture's mountain centroid; Crater Lake's landing is 80 m from Rim Village.

## Limits

- Coordinates were curated by hand, then checked against the DEM and against named features in Overture Maps (see above). Overture names come from OpenStreetMap and commercial sources and are not always right either.
- Copernicus GLO-30 is a surface model at 30 m: canopy and buildings are in the heights and sharp summits are lowered by up to about 150 m (Aiguille du Midi reads 3689 m against 3842 m).
- WorldCover's words are broad classes. The Grand Canyon's view reads as forest first, because the rim forest covers more of the 8 km circle than the bare canyon walls; that place keeps its curated palette.
- HLS imagery is stacked from up to six dates so swath gaps fill in; what none of them covers falls back to 500 m Blue Marble. Before this, the Glacier Point regional terrain showed a dark half with a diagonal swath edge right after the globe.
- NAIP (US only) has not been fetched from the cloud, where the USGS image service is blocked; the request and colour match are unit-tested with a stand-in image.
- Lidar is used only for the 800 m landing patch of the Earth zoom, and only in the US. The rest of the scene is the 10 m or 30 m DEM, and the descent passes over the 150 m regional mesh.
- The 1 m DEM is bare earth: trees, railings and buildings are not in it, so the brink is the ground's edge.
- The view and brink checks look along straight lines; they know nothing about paths, fences or closed areas.

## All places

| Id | Name | Country | Terrain | Ground (m) | Relief within 1 km (m) | Lidar | 1 m DEM at landing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grand-canyon-mather-point` | Mather Point, Grand Canyon | US | USGS 3DEP | 2164 | 1017 | AZ_GrandCanyonNP_1_2019 (2019, 14.8 pts/m²) | AZ_GrandCanyonNP_2019_B19 |
| `grand-canyon-desert-view` | Desert View Watchtower, Grand Canyon | US | USGS 3DEP | 2271 | 823 | AZ_GrandCanyonNP_1_2019 (2019, 14.8 pts/m²) | AZ_NorthCentral_B23, AZ_NorthEast_D23 |
| `horseshoe-bend` | Horseshoe Bend | US | USGS 3DEP | 1287 | 380 | none listed | AZ_NorthEast_D23, AZ_CentralCoconino_B22 |
| `monument-valley` | Monument Valley overlook | US | USGS 3DEP | 1695 | 378 | none listed | UT_WestEast_B22, UT_StatewideSouth_2020_A20 |
| `bryce-inspiration-point` | Inspiration Point, Bryce Canyon | US | USGS 3DEP | 2531 | 298 | USGS_LPC_UT_Southern_QL1_2018_LAS_2019 (2018, 19.9 pts/m²) | UT_StatewideKane_2020_A20 |
| `zion-angels-landing` | Angels Landing, Zion | US | USGS 3DEP | 1765 | 800 | none listed | none |
| `canyonlands-grand-view` | Grand View Point, Canyonlands | US | USGS 3DEP | 1913 | 427 | UT_WestEast_4_B22 (2022, 6.4 pts/m²) | UT_WestEast_B22 |
| `dead-horse-point` | Dead Horse Point | US | USGS 3DEP | 1813 | 552 | UT_WestEast_4_B22 (2022, 6.4 pts/m²) | UT_WestEast_B22 |
| `arches-delicate-arch` | Delicate Arch viewpoint, Arches | US | USGS 3DEP | 1445 | 181 | UT_FlamingGorge_5_2020 (2020, 9.3 pts/m²) | UT_FEMA_FS_FlamingGorge_2020_B20 |
| `yosemite-tunnel-view` | Tunnel View, Yosemite | US | USGS 3DEP | 1343 | 785 | none listed | CA_SouthernSierra_2020_B20, CA_YosemiteNP_2019_D19 |
| `yosemite-glacier-point` | Glacier Point, Yosemite | US | USGS 3DEP | 2201 | 1216 | none listed | CA_YosemiteNP_2019_D19 (brink 15 m) |
| `half-dome-summit` | Half Dome summit, Yosemite | US | USGS 3DEP | 2692 | 1385 | none listed | CA_YosemiteNP_2019_D19 |
| `mount-whitney` | Mount Whitney summit | US | USGS 3DEP | 4417 | 721 | CA_SierraNevada_14_B22 (2022, 30.6 pts/m²) | CA_SierraNevada_B22 |
| `death-valley-dantes-view` | Dantes View, Death Valley | US | USGS 3DEP | 1670 | 658 | none listed | CA_FEMAR9Southeast_D24 |
| `death-valley-zabriskie` | Zabriskie Point, Death Valley | US | USGS 3DEP | 201 | 163 | none listed | CA_FEMAR9Southeast_D24 |
| `big-sur-bixby-bridge` | Bixby Creek Bridge, Big Sur | US | USGS 3DEP | 123 | 254 | CA_FEMA_Z4_B1_2018 (2018, 5.7 pts/m²) | CA_AZ_FEMA_R9_Lidar_2017_D18 |
| `golden-gate-battery-spencer` | Battery Spencer, Golden Gate | US | USGS 3DEP | 144 | 263 | CA_SanFrancisco_1_B23 (2023, 62.1 pts/m²) | CA_SanFrancisco_B23 |
| `crater-lake-rim-village` | Rim Village, Crater Lake | US | USGS 3DEP | 2171 | 307 | none listed | OR_SouthEast_D22 (brink 18 m) |
| `mount-rainier-paradise` | Paradise, Mount Rainier | US | USGS 3DEP | 1656 | 417 | none listed | none |
| `mount-st-helens-johnston-ridge` | Johnston Ridge, Mount St Helens | US | USGS 3DEP | 1291 | 360 | none listed | WA_FEMAHQ_2018_D18 |
| `glacier-logan-pass` | Logan Pass, Glacier | US | USGS 3DEP | 2028 | 800 | MT_Statewide_P2_5_2020 (2020, 20.2 pts/m²) | MT_Statewide_Phase3_2021_B21, MT_Statewide_Phase2_2020_B20 |
| `smokies-clingmans-dome` | Clingmans Dome, Great Smoky Mountains | US | USGS 3DEP | 2025 | 501 | NC_Phase5_Swain_2017 (2017, 58.4 pts/m²) | NC_Phase5_2018_A18 |
| `acadia-cadillac-mountain` | Cadillac Mountain, Acadia | US | USGS 3DEP | 458 | 327 | ME_MidCoast_1_2021 (2021, 5.6 pts/m²) | ME_MidCoast_2021_B21 |
| `mount-washington` | Mount Washington summit | US | USGS 3DEP | 1914 | 515 | NH_Umbagog_2016 (2016, 7.4 pts/m²) | NH_Umbagog_LiDAR_2016_D16 |
| `haleakala-summit` | Haleakalā summit | US | Copernicus DEM | 3051 | 334 | HI_NOAAMauiOahu_2_B20 (2020, 26.6 pts/m²) | — |
| `gornergrat-matterhorn` | Gornergrat | CH | Copernicus DEM | 3116 | 690 | swisstopo swissALTI3D, 0.5 m (not wired) | — |
| `jungfraujoch-sphinx` | Jungfraujoch Sphinx | CH | Copernicus DEM | 3490 | 970 | swisstopo swissALTI3D, 0.5 m (not wired) | — |
| `aiguille-du-midi` | Aiguille du Midi | FR | Copernicus DEM | 3689 | 1112 | IGN LiDAR HD (not wired) | — |
| `tre-cime-di-lavaredo` | Rifugio Locatelli, Tre Cime di Lavaredo | IT | Copernicus DEM | 2409 | 655 | none listed | — |
| `preikestolen` | Preikestolen (Pulpit Rock) | NO | Copernicus DEM | 572 | 703 | Kartverket høydedata, 1 m (not wired) | — |
| `old-man-of-storr` | The Storr, Isle of Skye | GB | Copernicus DEM | 704 | 438 | Scottish Public Sector LiDAR (partial coverage) (not wired) | — |
| `ben-nevis` | Ben Nevis summit | GB | Copernicus DEM | 1342 | 683 | Scottish Public Sector LiDAR (partial coverage) (not wired) | — |
| `rhossili-worms-head` | Rhossili cliff top | GB | Copernicus DEM | 78 | 190 | Welsh Government LiDAR 2020-23, 1 m (not wired) | — |
| `cliffs-of-moher` | Cliffs of Moher | IE | Copernicus DEM | 156 | 188 | Geological Survey Ireland lidar (partial coverage) (not wired) | — |
| `santorini-oia` | Oia, Santorini | GR | Copernicus DEM | 129 | 139 | none listed | — |
| `table-mountain` | Table Mountain upper cableway | ZA | Copernicus DEM | 1055 | 816 | none listed | — |
| `kilimanjaro-uhuru` | Uhuru Peak, Kilimanjaro | TZ | Copernicus DEM | 5866 | 623 | none listed | — |
| `kala-patthar-everest` | Kala Patthar | NP | Copernicus DEM | 5606 | 635 | none listed | — |
| `mount-fuji` | Mount Fuji summit | JP | Copernicus DEM | 3767 | 767 | GSI 5 m DEM (partly lidar) (not wired) | — |
| `victoria-peak` | Victoria Peak, Hong Kong | HK | Copernicus DEM | 390 | 489 | none listed | — |
| `guilin-xianggong-hill` | Xianggong Hill, Li River karst | CN | Copernicus DEM | 285 | 284 | none listed | — |
| `uluru-sunset-view` | Uluru sunset viewing area | AU | Copernicus DEM | 527 | 37 | Geoscience Australia ELVIS lidar (partial coverage) (not wired) | — |
| `milford-sound` | Milford Sound foreshore | NZ | Copernicus DEM | 6 | 772 | LINZ national 1 m DEM (lidar), public AWS bucket nz-elevation (not wired) | — |
| `sugarloaf-rio` | Sugarloaf Mountain, Rio de Janeiro | BR | Copernicus DEM | 303 | 329 | none listed | — |
| `machu-picchu` | Machu Picchu guardhouse | PE | Copernicus DEM | 2455 | 1007 | none listed | — |
