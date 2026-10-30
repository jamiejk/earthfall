# GeoBlock source and environment check — 2026-10-03

*This is the source-check snapshot from 3 October. Later Blender and H3 work is recorded in [the project story](PROJECT_STORY.md); the unresolved Welsh height-datum finding below still applies.*

## Local environment

- Project environment: CPython 3.12.13 through `uv`; dependencies locked in `uv.lock`.
- System Python is 3.14.7; the project does not install GIS packages into that interpreter or into ComfyUI.
- Blender 5.2.2 LTS and FFmpeg 8.1.2 are available as separate executables.
- A configured `OPENTOPOGRAPHY_API_KEY` was not present. No authenticated GEDTM30 request has been made.

## Source A: GEDTM30

[OpenTopography's dataset metadata](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.082025.4326.1) identifies GEDTM30 v20250619 as a roughly 30 m DTM, WGS84 horizontal reference, EGM2008 height reference (EPSG:3855), CC BY 4.0, with source surveys from 2006–2015. Its [current OpenAPI specification](https://portal.opentopography.org/apidocs/openapi.json) lists `GEDTM30` for `demtype` and requires `south`, `north`, `west`, `east`, and `API_Key`; `outputFormat=GTiff` is supported. The provider implements that endpoint and preserves a successful response in a locked raw cache. Its authenticated path has not yet been exercised.

The [publisher's v20250619 COG](https://s3.opengeohub.org/global/edtm/filtered.dtm_edtm_m_30m_s_20060101_20151231_go_epsg.4326.3855_v20250619.tif) is publicly byte-range readable. Its local subset is `data/worms_head_gedtm30_v20250619_subset.tif`, with source URI and checksum in the adjacent JSON file. The raster is Int32 with a 0.1 scale from decimetres to metres. Exact-zero cells were conservatively treated as unknown because the source leaves large sea areas at zero without a nodata mask. This also removes any genuinely zero-height land cells. The A terrain asset has 18.31% valid coverage in the project square.

The newer v1.2 COG was inspected but not selected: it is Float32, its metadata PDF states metre units, and its raw values around Worm's Head are on the order of 80 m, while its TIFF header still reports a 0.1 scale. Applying that header reduces the local relief tenfold. The release needs a source clarification before use in a matched comparison.

## Source B: Welsh LiDAR DTM

[DataMapWales](https://datamap.gov.wales/maps/lidar-data-download/) publishes the 32-bit DTM COG under the map's OGL licence. The [tile catalogue](https://datamap.gov.wales/layers/geonode%3Awelsh_government_lidar_tile_catalogue_2020_2023/metadata_detail) describes 1 m terrain data collected in 2020–2023. A byte-range read of the [published COG](https://dmwproductionblob.blob.core.windows.net/cogs/lidar/wales_dtm_32bit_cog.tif) found:

| Property | Inspected value |
| --- | --- |
| CRS | EPSG:27700 horizontal only |
| Raster size | 191,007 × 233,000 pixels |
| Pixel size | 1 × 1 m |
| Nodata | -9999 |
| Scale, offset | 1, 0 |
| File size | 48,611,310,928 bytes |
| Provisional 3 km square valid fraction | 18.4% after projection; remaining area is mostly sea |
| Source ETag | `0x8DCE488A5BF1B2C` |

The local subset is `data/worms_head_lidar_subset.tif`, with acquisition details and checksum in `data/worms_head_lidar_subset.json`. The top-down previews and GLBs in `outputs/A_global/` and `outputs/B_lidar/` use a manually selected provisional centre at 51.565° N, 4.31° W. The landmark coordinate is 51.5643° N, 4.3241° W. Neither complete source raster was downloaded.

The COG and the cited catalogue do **not** declare a vertical datum. The [original 1 m DTM tile](https://dmwproductionblob.blob.core.windows.net/lidar-zips/2023-24/dtm/wg_del_35_239187_20220120dtm.tif) checked near the western headland also has EPSG:27700 horizontal coordinates but no vertical CRS tag. The catalogue records acquisition on 20 January 2022 for this area. An [NRW coastal study](https://naturalresources.wales/media/e3cee2t3/r4537_vol1_coastal-squeeze-methodology_final.pdf) uses the All Wales 1.0 m LiDAR (2020–2022) in a combined project DTM referenced to ODN, converting bathymetry from Chart Datum to ODN. This is strong supporting evidence for ODN, but the report does not explicitly certify the raw elevation values in this tile or state whether any vertical correction was applied in its workflow. The generated B asset therefore remains `exploratory_unaligned`.

To close this gate, obtain the National LiDAR Survey for Wales technical specification or the tile's survey/QA report from Welsh Government or Natural Resources Wales and confirm the delivered DTM height datum, units, geoid model, and any corrections for tile `wg_del_35_239187_20220120dtm.tif`. The [survey procurement notice](https://www.sell2wales.gov.wales/Search/Show/Search_View.aspx?ID=SEP310536) identifies the contract and independent quality-control work but does not publish the technical specification. Cross-check a few flat, stable, bare-ground locations against independent modern ODN control; calculate the signed residuals and check for a systematic offset. A shoreline is unsuitable as the datum test because its elevation depends on the tide, waves, and survey time, while LiDAR often has missing or false water returns. ODN zero is a fixed reference surface, not the instantaneous sea level. For a rendered water plane, choose a tidal stage separately, express it in ODN metres, and use the same stage for both terrain variants.

## Height conversion feasibility

[Ordnance Survey](https://www.ordnancesurvey.co.uk/geodesy-positioning/coordinate-transformations) documents OSTN15 and OSGM15 for British National Grid and ODN transformation. A PROJ operation from GEDTM30's WGS84 + EGM2008 reference to BNG + ODN was selected after downloading these [official PROJ grids](https://cdn.proj.org/):

| Grid | SHA-256 |
| --- | --- |
| `uk_os_OSGM15_GB.tif` | `bf0ac81f1c0c33332693d7d606f335b793c3c2995b8e69eeb1350bd94adcde3b` |
| `uk_os_OSTN15_NTv2_OSGBtoETRS.tif` | `5d6ed64d2119952c4c559fa1fccbc594b6520fc3ec3ef2fc10be13202c4384fa` |
| `us_nga_egm08_25.tif` | `4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a` |

The selected operation reports 1.151 m accuracy. A smoke transformation of 50.0 m EGM2008 height at the landmark returned 50.5939 m ODN. This verifies that a transformation can run with the grids available; it is not a validation of the LiDAR datum or absolute terrain accuracy. The grids used for the A asset are in ignored `data/proj/`. The processor selects the best available projected-to-geographic operation and samples GEDTM30 with that operation; the Welsh raster already uses the project horizontal CRS. The final package should record the grid checksums in its manifest, rather than relying only on this check report.

## Gate status

Both datasets were read from their published COGs, and provisional A/B terrain assets exist. The matched pair remains blocked by the Welsh DTM's undocumented vertical reference. The compiler's pair validator rejects the exploratory B height alignment. An individual key is still needed to exercise the OpenTopography API path; the public publisher COG supplied the A fixture independently. No verified matched pair, Blender scene or video result is claimed. A chosen video service's real account upload limits also remain unchecked, but this does not block terrain development.

The exploratory numeric comparison is in `outputs/exploratory_comparison.json`. It samples 1,531 overlapping 30 m cell centres and records three provisional land checkpoints. Its A-minus-B median is -0.78 m, with a 5th-to-95th percentile range of -15.30 to +4.41 m. Those differences combine source methodology, pixel spacing and unresolved Welsh height reference; they cannot yet support a terrain-fidelity conclusion.
