"""Cache mapped pedestrian layout and convert it to local draped surface strips."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np


def fetch(latitude: float, longitude: float, cache: Path, radius: int = 180) -> dict:
    import httpx
    if cache.is_file():
        return json.loads(cache.read_text())
    query = (f'[out:json][timeout:25];(way(around:{radius},{latitude},{longitude})[highway];'
             f'way(around:{radius},{latitude},{longitude})[barrier];);out tags geom;')
    errors = []
    for endpoint in ('https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter'):
        try:
            response = httpx.get(endpoint, params={'data': query}, timeout=40,
                headers={'User-Agent': 'Touchdown/0.1 (https://github.com/jamiejk/geoblock-scenecompiler)'})
            response.raise_for_status()
            result = response.json()
            if 'elements' not in result:
                raise ValueError('No map elements returned')
            result['provenance'] = {'source': 'OpenStreetMap contributors', 'license': 'ODbL 1.0',
                'attribution_url': 'https://www.openstreetmap.org/copyright', 'endpoint': endpoint,
                'query': query, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat()}
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2) + '\n')
            return result
        except (httpx.HTTPError, ValueError) as error:
            errors.append(str(error))
    raise RuntimeError('Pedestrian layout discovery failed: ' + '; '.join(errors))


def project_layout(source: dict, metadata: dict, zone: int) -> dict:
    from pyproj import Transformer
    from geoblock_core.utm import projected_crs
    project = Transformer.from_crs('EPSG:4326', projected_crs(zone), always_xy=True)
    paths, barriers = [], []
    for element in source['elements']:
        tags = element.get('tags', {})
        geometry = element.get('geometry', [])
        if len(geometry) < 2:
            continue
        points = []
        for point in geometry:
            x, y = project.transform(point['lon'], point['lat'])
            points.append([x - metadata['x'], y - metadata['y']])
        if tags.get('barrier'):
            barriers.append({'id': element['id'], 'tags': tags, 'points': points})
        if tags.get('highway') not in ('footway', 'pedestrian', 'steps', 'path'):
            continue
        if any(abs(x) > 380 or abs(y) > 380 for x,y in points):
            continue  # do not clamp an out-of-window path onto the near DEM edge
        try:
            width = float(tags['width'].removesuffix(' m'))
            width_source = 'OSM width tag'
        except (KeyError, ValueError):
            width = 2 if tags['highway'] == 'steps' else 3
            width_source = 'estimated default; not measured from photograph'
        if not 0.5 <= width <= 12:
            continue
        paths.append({'id': element['id'], 'tags': tags, 'points': points,
                      'width_m': width, 'width_source': width_source})
    if not paths:
        raise ValueError('No mapped pedestrian surface available for a shared layout')
    return {'paths': paths, 'barriers': barriers, 'provenance': source.get('provenance'),
            'limitations': ['Estimated widths where untagged', 'Steps follow the DEM rather than measured risers',
                           'No railings reconstructed unless separately modelled from mapped geometry']}


def strip_mesh(points: list[list[float]], width: float, heights: np.ndarray) -> tuple[list, list]:
    dense = []
    for a, b in zip(points, points[1:]):
        delta = np.asarray(b) - a
        length = float(np.linalg.norm(delta))
        if length < 0.01:
            continue
        for fraction in np.linspace(0, 1, max(2, math.ceil(length * 2) + 1))[:-1]:
            dense.append(np.asarray(a) + delta * fraction)
    dense.append(np.asarray(points[-1]))
    vertices = []
    cross_samples = max(2, math.ceil(width * 2) + 1)
    for index, point in enumerate(dense):
        direction = dense[min(index + 1, len(dense) - 1)] - dense[max(0, index - 1)]
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        normal = np.array([-direction[1], direction[0]]) / length * width / 2
        for offset in np.linspace(1, -1, cross_samples):
            xy = point + normal * offset
            x, y = xy
            row, col = np.clip(200 - y / 2, 0, 400), np.clip(200 + x / 2, 0, 400)
            r, c = int(row), int(col)
            rr, cc = min(r + 1, 400), min(c + 1, 400)
            z = ((1 - (row-r)) * ((1-(col-c))*heights[r,c] + (col-c)*heights[r,cc])
                 + (row-r) * ((1-(col-c))*heights[rr,c] + (col-c)*heights[rr,cc]))
            vertices.append((float(x), float(y), float(z) + 0.08))
    faces = [(row*cross_samples+col, (row+1)*cross_samples+col,
              (row+1)*cross_samples+col+1, row*cross_samples+col+1)
             for row in range(len(vertices)//cross_samples-1) for col in range(cross_samples-1)]
    return vertices, faces


def fit_walking_surface(vertices: list, faces: list, target: tuple,
                        radius: float = 16, max_correction: float = 0.6) -> tuple[list, list, dict]:
    """Infer a modest planar walking surface, without extending across a drop.

    Shape remains the mapped strip. Corrections fade outside the landing area;
    excessive corrections invalidate faces rather than bridging missing terrain.
    This is inferred geometry, not a surveyed platform outline.
    """
    if not vertices:
        return vertices, faces, {'assessed': False, 'reason': 'no mapped paved surface'}
    xyz = np.asarray(vertices, dtype=float)
    distance = np.linalg.norm(xyz[:, :2] - target, axis=1)
    local = distance < 6
    if local.sum() < 6:
        return vertices, faces, {'assessed': False, 'reason': 'no mapped surface near target'}
    design = np.column_stack((xyz[local, :2] - target, np.ones(local.sum())))
    # Robust central elevation removes edge samples that fall down a cliff.
    median = np.median(xyz[local, 2])
    usable = np.abs(xyz[local, 2] - median) < 0.75
    if usable.sum() < 6:
        return vertices, faces, {'assessed': False, 'reason': 'insufficient locally consistent elevations'}
    plane = np.linalg.lstsq(design[usable], xyz[local, 2][usable], rcond=None)[0]
    slope = float(np.linalg.norm(plane[:2]))
    if slope > 0.05:
        plane[:2] *= 0.05 / slope
    desired = (xyz[:, :2] - target) @ plane[:2] + plane[2]
    correction = desired - xyz[:, 2]
    weight = np.clip((radius - distance) / (radius - 6), 0, 1)
    invalid = (weight > 0) & (np.abs(correction) > max_correction)
    xyz[:, 2] += np.where(invalid, 0, correction * weight)
    kept = [face for face in faces if not any(invalid[i] for i in face)]
    return xyz.tolist(), kept, {'assessed': True, 'inferred': True,
        'grade': float(np.linalg.norm(plane[:2])), 'max_correction_m': max_correction,
        'rejected_vertices': int(invalid.sum()), 'rejected_faces': len(faces)-len(kept),
        'radius_m': radius, 'outline_source': 'mapped path strip; estimated width where untagged'}
