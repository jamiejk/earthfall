"""Curated discovery index for geometry-capable GeoBlock source families.

Discovery is deliberately separate from shot qualification: a provider's LoD or
coverage claim cannot establish that the buildings in a particular view exist.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


CATEGORIES = ("All", "Catalogs", "3D city models", "Landform terrain")


@dataclass(frozen=True)
class Resource:
    name: str
    category: str
    geography: str
    geometry: str
    status: str
    url: str
    note: str
    search_terms: str = ""
    machine_url: str = ""


RESOURCES = (
    Resource(
        "3DBAG", "3D city models", "Netherlands", "LoD2.2 building roofs",
        "Backup demo candidate — inspect target tiles", "https://docs.3dbag.nl/en/",
        "Good roof geometry for urban context, but not a detailed facade/landmark mesh. Individual buildings can be missing, dated or simplified; pair with checked terrain.",
        "dutch holland buildings city",
    ),
    Resource(
        "Hamburg LoD3.0-HH", "3D city models", "Hamburg, Germany",
        "Detailed textured buildings on a 5 m terrain model", "Priority demo candidate — target tile unchecked",
        "https://www.govdata.de/suche/daten/3d-gebaudemodell-lod3-0-hh-hamburg",
        "CityGML with detailed roof forms and 20 cm facade textures; inspect the user-selected landmark and its visible corridor before choosing a shot. DL-DE BY 2.0 attribution.",
        "germany harbour harbor waterfront buildings city",
    ),
    Resource(
        "Helsinki 3D Mesh", "3D city models", "Helsinki, Finland",
        "Photogrammetry mesh of buildings and ground", "Priority demo candidate — target tile unchecked",
        "https://www.hel.fi/en/decision-making/information-on-helsinki/maps-and-geospatial-data/helsinki-3d",
        "Downloadable OBJ mesh under CC BY 4.0; inspect the selected tile for roof and facade fidelity. Semantic CityGML LoD2 is a fallback.",
        "finland finnish helsinki buildings city",
    ),
    Resource(
        "Open 3D City Model Catalog", "Catalogs", "Worldwide registry",
        "CityJSON/CityGML discovery metadata", "Discovery only — no shot approval",
        "https://catalog.open3d.city/",
        "Community STAC registry; its entries and licences may be incomplete. Follow an item to the source, then validate actual geometry.",
        "open3d city model catalog stac cities",
        "https://storage.googleapis.com/city3d-stac/catalog.json",
    ),
    Resource(
        "Welsh Government LiDAR DTM", "Landform terrain", "Wales",
        "Measured bare-earth relief", "Dataset lead — target tile unchecked",
        "https://datamap.gov.wales/maps/lidar-data-download/",
        "Suitable for natural landforms after tile, relief, and vertical-datum checks; does not supply building shapes.",
        "wales welsh landform coast topography",
    ),
)

_QUERY_FILLER = {"a", "an", "at", "can", "fly", "flyover", "for", "i", "in", "me", "of", "over",
                 "see", "show", "the", "to", "want"}


def _words(value: str) -> set[str]:
    # Treat Worm's/Worms and punctuation variants alike; this is an index filter,
    # not a geocoder or proof that a dataset covers a requested coordinate.
    normalized = re.sub(r"(?<=\w)['’](?=\w)", "", value.casefold())
    return set(re.findall(r"[\w]+", normalized))


def discover_resources(category: str = "All", search: str = "") -> tuple[list[Resource], str]:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown resource category: {category}")
    terms = _words(search) - _QUERY_FILLER
    matches = [
        resource for resource in RESOURCES
        if (category == "All" or resource.category == category)
        and terms.issubset(_words(" ".join((resource.name, resource.geography, resource.geometry,
                                           resource.note, resource.search_terms))))
    ]
    matches.sort(key=lambda resource: resource.name.casefold())
    header = f"GEOMETRY SOURCE BROWSER · {category} · query: {search.strip() or 'all'}"
    lines = [header, f"{len(matches)} matching resources (alphabetical)"]
    for resource in matches:
        lines.extend((
            f"\n{resource.name} [{resource.category}]",
            f"  Area: {resource.geography}",
            f"  Geometry: {resource.geometry}",
            f"  Status: {resource.status}",
            f"  {resource.note}",
            f"  Source: {resource.url}",
        ))
        if resource.machine_url:
            lines.append(f"  Machine-readable catalog: {resource.machine_url}")
    if not matches:
        lines.append("No curated source matches. Try a wider area or another category; no flat substitute will be used.")
    lines.append("\nA listed source is a lead, not proof of a renderable shot. Verify the exact area and visible geometry before blocking.")
    return matches, "\n".join(lines)
