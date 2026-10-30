# Touchdown server for the NAS (hosted demo): web app, JSON API, queue and files. No GPU work happens here; a worker
# on the GPU machine pulls jobs over HTTPS. Build from the repository root (BuildKit reads Dockerfile.dockerignore):
#   docker compose -f app/docker-compose.yml up -d --build
# python:3.12-slim is multi-arch (amd64, arm64), and every dependency here has wheels for both.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/srv/touchdown/app:/srv/touchdown/src TOUCHDOWN_REPO=/srv/touchdown TOUCHDOWN_DATA=/data \
    TOUCHDOWN_HOST=0.0.0.0 TOUCHDOWN_PORT=8000

# numpy is here because geoblock_core.lidar imports it at module level, so the custom-point check needs it
RUN pip install "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "python-multipart>=0.0.9,<1" "tomli-w>=1.0,<2" \
        "httpx>=0.27,<1" "Pillow>=11,<13" "pillow-heif>=1.8.1,<2" "pyproj>=3.7,<4" "numpy>=2.0,<3" \
    && mkdir -p /data

WORKDIR /srv/touchdown
COPY app/server app/server
COPY app/web app/web
COPY app/tested_places.json app/tested_places.json
# the place catalogue, and the 1 m lidar coverage check for custom points (geoblock_core.lidar: httpx + pyproj + numpy)
COPY src/geoblock_core/__init__.py src/geoblock_core/lidar.py src/geoblock_core/place_photos.py src/geoblock_core/
COPY src/geoblock_core/data/places.json src/geoblock_core/data/
# the data tier of the non-US catalogue places (the server rates them with it, the same as manifest_gen does)
COPY src/geoblock_core/data/place_data_support.json src/geoblock_core/data/
# the preset manifests: the server reads only each one's default [shot] person
COPY shots/flux3/monument-valley.toml shots/flux3/grand-canyon-mather.toml shots/flux3/yosemite-tunnel-view.toml shots/flux3/

# The container runs as root: Docker creates a missing bind-mount source (./data) as root-owned, so a container
# running as another user cannot write /data and the server dies on start. Root inside the container is a
# trade-off (it can write anywhere in the container's filesystem, not just /data); the image serves the app and
# holds nothing but the worker token in its environment.
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/config', timeout=4)"
CMD ["python", "-m", "server", "--server-only"]
