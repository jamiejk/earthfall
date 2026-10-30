from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GeoSceneSpec(BaseModel):
    """Canonical XY area shared by every terrain variant."""

    model_config = ConfigDict(extra="allow")
    schema_version: str = "1.0"
    project_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    projected_crs: str = "EPSG:27700"
    vertical_crs: str = "EPSG:5701"
    radius_m: float = Field(default=1500, gt=0)
    radius_interpretation: str = "half_width_of_projected_square"
    origin_easting_m: float
    origin_northing_m: float
    location_source: str
    origin_source: str = "projected location coordinate"

    @field_validator("schema_version")
    @classmethod
    def supported_version(cls, value: str) -> str:
        if value.split(".")[0] != "1":
            raise ValueError("unsupported GeoSceneSpec major version")
        return value

    @model_validator(mode="after")
    def check_contract(self):
        if self.radius_interpretation != "half_width_of_projected_square":
            raise ValueError("unsupported radius interpretation")
        return self

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        r = self.radius_m
        return (self.origin_easting_m-r, self.origin_northing_m-r,
                self.origin_easting_m+r, self.origin_northing_m+r)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    provider: str
    dataset: str
    source_type: str = "DTM"
    horizontal_crs: str
    vertical_crs: str | None = None
    vertical_reference_verified: bool = False
    nominal_resolution_m: float | None = None
    native_pixel_spacing: tuple[float, float] | None = None
    native_pixel_spacing_units: str | None = None
    nodata_policy: str = "source_mask_only"
    license: str | None = None
    attribution: str | None = None
    acquisition_dates: str | None = None
    source_uri: str
    retrieval_utc: str | None = None
    sha256: str | None = None
    checksum_scope: str = "acquired_local_raster"


class TerrainAsset(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "1.0"
    project_id: str
    variant: str
    source: SourceRecord
    project_crs_wkt: str
    horizontal_operation: str
    horizontal_operation_accuracy_m: float | None = None
    origin_easting_m: float
    origin_northing_m: float
    bounds_local_m: tuple[float, float, float, float]
    common_vertical_crs: str
    vertical_operation: str
    vertical_operation_accuracy_m: float | None = None
    height_alignment: str
    raster_spacing_m: float
    mesh_spacing_m: float
    mesh_vertices: int
    mesh_triangles: int
    coverage_fraction: float
    elevation_min_m: float | None = None
    elevation_max_m: float | None = None
    gltf_basis: str = "local (east,north,up) -> glTF (east,up,-north)"
    files_sha256: dict[str, str]

    @field_validator("schema_version")
    @classmethod
    def supported_version(cls, value: str) -> str:
        if value.split(".")[0] != "1":
            raise ValueError("unsupported TerrainAsset major version")
        return value
